#!/usr/bin/env python
"""
Live commute-isochrone server: set ANY workplace address in the browser and the map
recomputes door-to-door times from a grid of SF origins.

The R5 network is loaded ONCE at startup and kept warm, so each recompute is just the
travel-time step (no network rebuild). Run:

    .venv/bin/python scripts/server.py        # then open http://127.0.0.1:8000

Env: GRID_M (default 200) trades detail for speed; EXACT_THREADS, WINDOW_MIN, PORT.

The workplace address lives only in the browser — this process never hardcodes or
persists it. Two distinct computations power the map (see CLAUDE.md/Issues.md):
  * map TIME  — /compute is a fast reverse one-to-many approximation; /compute_exact is
                the exact forward per-cell refine. (scripts/isochrone.py is the reference.)
  * route LEGS — /itinerary + /attribution use R5 recorded paths; hover computes one cell
                on demand, color-by-line builds the whole grid lazily on first toggle.
"""
import os, sys, json, copy, math, time, threading, datetime as dt
from collections import OrderedDict
# Cap numba's thread pool BEFORE numba is imported (via core.raptor_*). Bounds the parallel MC
# kernel's worker threads — fewer idle threads (lower RSS) and no oversubscription on a small box.
# Overridable via NUMBA_NUM_THREADS. (The MC kernel is also serialized below: numba's workqueue
# threading layer is NOT threadsafe, so two concurrent parallel kernels would abort the process.)
os.environ.setdefault("NUMBA_NUM_THREADS", str(min(4, os.cpu_count() or 4)))
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
from flask_limiter import Limiter

from core import config, feeds, geo              # lightweight JVM-free core (no r5py/pandas/geopandas)
from core.raptor_journey import reconcile_legs   # shared leg-rounding reconciliation (JVM-free)
config.load_dotenv()             # load .env (GEOCODER / GEOAPIFY_KEY) for the geocoder
# pandas / geopandas / shapely / Point / core.grid are imported LAZILY in the boot BUILD path only
# (below): the lean JVM-free boot loads a precomputed static bundle and never pulls the ~70 MB
# geo/pandas stack. They become module globals there, used by the R5 branches (unreachable lean).

# ---- Flags (parsed EARLY: they decide whether the in-process JVM starts at all) --------
# THE DEFAULT IS NOW THE JVM-FREE STACK: USE_RAPTOR + USE_WALK_GRAPH + arrive-by-09:00 + the
# service-noise overlay. R5/the JVM is no longer loaded by default. Opt back into the legacy R5
# path with USE_RAPTOR=0 (and/or USE_WALK_GRAPH=0 for the R5 walk matrix, RAPTOR_SEMANTIC=departafter).
USE_RAPTOR = os.environ.get("USE_RAPTOR", "1").lower() in ("1", "true", "yes", "on")
RAPTOR_SEMANTIC = os.environ.get("RAPTOR_SEMANTIC", "arriveby").lower()
RAPTOR_MC = os.environ.get("RAPTOR_MC", "1").lower() in ("1", "true", "yes", "on")
USE_WALK_GRAPH = os.environ.get("USE_WALK_GRAPH", "1").lower() in ("1", "true", "yes", "on")
# Safety net: the JVM-free walk stack needs its one-time bakes (walk graph + access_walk table).
# If they're absent (fresh checkout), fall back to the R5 walk matrix (re-enables the JVM) with a
# clear message instead of crashing. Bake them to go fully JVM-free (see below / setup.sh).
if USE_WALK_GRAPH and not (
        (config.DATA / "walk_graph.npz").exists()
        and any((config.DATA / "raptor_cache").glob("access_walk_*m_*.npz"))):
    print("[boot] USE_WALK_GRAPH on but the walk graph / access_walk table isn't baked — falling "
          "back to R5 walk (JVM). Bake the JVM-free stack: scripts/fetch_dem.sh && "
          "scripts/build_walk_graph.py && scripts/bake_walk_access.py")
    USE_WALK_GRAPH = False
# The walk graph is only reachable through the RAPTOR path (_WG loads inside `if USE_RAPTOR:`,
# and R5 serves everything under USE_RAPTOR=0) — force the flag off so /healthz can't report
# walk=graph for a graph that never loads.
if USE_WALK_GRAPH and not USE_RAPTOR:
    print("[boot] USE_WALK_GRAPH requires USE_RAPTOR; walk graph disabled (R5 walk)")
    USE_WALK_GRAPH = False
# Validate the semantic LOUDLY: a typo (e.g. "arrive-by") would otherwise silently serve the
# departafter branch — and load the JVM — while /healthz echoes the bogus string as healthy.
if RAPTOR_SEMANTIC not in ("arriveby", "departafter"):
    raise SystemExit(f"RAPTOR_SEMANTIC must be 'arriveby' or 'departafter', "
                     f"got {RAPTOR_SEMANTIC!r}")
# FULLY JVM-free when the map, breakdown, AND walk all come from RAPTOR + the walk graph — now for
# BOTH semantics: arrive-by traces a single 09:00 deadline tree, depart-after traces the window's
# per-T* trees (DepartAfterJourneyTree), both pure numpy/numba. R5 is still needed only off the
# RAPTOR path (fast approx / R5 walk matrix), i.e. USE_RAPTOR=0 or USE_WALK_GRAPH=0.
_NEED_R5 = not (USE_RAPTOR and USE_WALK_GRAPH and RAPTOR_SEMANTIC in ("arriveby", "departafter"))

network = None
MAX_INT32 = (2 ** 31) - 1                         # R5's "unreachable" sentinel (JVM-free fallback)
DEFAULT_MAX_RIDES = 8                             # R5's ride cap (rides = transfers + 1)
if _NEED_R5:
    # Cap the JVM heap BEFORE r5py starts it (small hosting box; over-reserve risks OOM). r5py
    # reads --max-memory; unset keeps its default (80% of RAM). Local rule: do NOT set it.
    _r5_mem = os.environ.get("R5_MAX_MEMORY")
    if _r5_mem and "--max-memory" not in sys.argv and "-m" not in sys.argv:
        sys.argv += ["--max-memory", _r5_mem]
    from core import network                      # imports r5py -> starts the in-process JVM
    from core.network import MAX_INT32, DEFAULT_MAX_RIDES
    import com.conveyal.r5

HERE = Path(__file__).resolve().parent
GRID_M = int(os.environ.get("GRID_M", str(config.GRID_M)))

# ---- Boot: feeds, model date, grid, warm R5 network (all once) -------------------------
GTFS = config.gtfs_paths()

# ---- Static page data: cells (GeoJSON), origin coords, line shapes, service date --------
# These are workplace-INDEPENDENT (only the feeds + neighborhoods determine them). We cache them to
# data/server_static.json so the JVM-free server boots WITHOUT geopandas/shapely/pandas (~70 MB).
# The R5 path — and the first JVM-free boot, or after a GTFS repull (fingerprint mismatch) — builds
# them via the geo stack and writes the bundle; subsequent JVM-free boots load it with json only.
_STATIC = config.DATA / "server_static.json"


def _gtfs_fp():
    import hashlib
    h = hashlib.sha256()
    for p in GTFS:
        st = Path(p).stat()
        h.update(f"{Path(p).name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


_fp = _gtfs_fp()
_bundle = None
if not _NEED_R5 and _STATIC.exists():
    try:
        _b = json.loads(_STATIC.read_text())
        if _b.get("gtfs_fp") == _fp:
            _bundle = _b
    except Exception:
        _bundle = None

if _bundle is not None:                          # LEAN boot — json only, no geopandas/shapely/pandas
    CELLS_GEOJSON = _bundle["cells"]; LINES = _bundle["lines"]
    ORIGIN_LL = {k: tuple(v) for k, v in _bundle["origin_ll"].items()}
    _SVC_DATE = dt.datetime.strptime(_bundle["svc_date"], "%Y%m%d").date()
    DEP = config.departure(_SVC_DATE)
    NET = SNAPPED_GRID = _TRANSIT_LAYER = None
    _EXACT_IDS, _EXACT_GEOMS = [], []
    print(f"[boot] modeling weekday {_SVC_DATE} @ {DEP:%-I:%M%p}".lower())
    print(f"[boot] ready: {len(ORIGIN_LL)} origins (JVM-free, lean static bundle — no geopandas). "
          f"Open http://127.0.0.1:8000")
else:                                            # BUILD path — needs the geo/pandas stack (one-time)
    import pandas as pd, geopandas as gpd, shapely      # noqa: F401 (module globals for R5 branches)
    from shapely.geometry import Point                  # noqa: F401
    from core import grid
    LINES = feeds.route_shapes(GTFS)             # GTFS line geometries for the overlay
    # Auto-pick a weekday with trips in ALL feeds (a hardcoded date silently breaks after a repull).
    _SVC_DATE = feeds.pick_service_date(GTFS)
    DEP = config.departure(_SVC_DATE)
    print(f"[boot] modeling weekday {_SVC_DATE} @ {DEP:%-I:%M%p}".lower())
    print(f"[boot] building grid @ {GRID_M}m" + (" + loading R5 network (once)..." if _NEED_R5
          else " (one-time geo build; caching static bundle for lean reboots)..."))
    NEIGH = grid.load_neighborhoods()
    GRID = grid.build_grid(NEIGH, GRID_M)[["id", "geometry"]]
    ORIGIN_LL = {r.id: (r.geometry.y, r.geometry.x) for r in GRID.itertuples()}
    _cells = gpd.GeoDataFrame({"id": GRID["id"].values},
                              geometry=grid.square_cells(GRID, GRID_M).values, crs=config.WGS)
    _cells = grid.attach_neighborhoods(_cells, NEIGH)
    CELLS_GEOJSON = json.loads(_cells.to_json())
    for f in CELLS_GEOJSON["features"]:
        f["properties"] = {"id": f["properties"]["id"], "n": f["properties"].get("name")}
    try:                                         # cache the bundle so the next JVM-free boot is lean
        _STATIC.write_text(json.dumps({
            "gtfs_fp": _fp, "svc_date": _SVC_DATE.strftime("%Y%m%d"),
            "origin_ll": {k: [v[0], v[1]] for k, v in ORIGIN_LL.items()},
            "cells": CELLS_GEOJSON, "lines": LINES}))
    except Exception as _e:
        print(f"[boot] (could not cache static bundle: {_e})")
    if _NEED_R5:
        NET = network.build_network(GTFS)
        SNAPPED_GRID = GRID.copy()
        SNAPPED_GRID["geometry"] = NET.snap_to_network(GRID.geometry)
        SNAPPED_GRID = SNAPPED_GRID[SNAPPED_GRID.geometry != shapely.Point()].reset_index(drop=True)
        _EXACT_IDS = list(SNAPPED_GRID.id)
        _EXACT_GEOMS = list(SNAPPED_GRID.geometry)
        _TRANSIT_LAYER = NET._transport_network.transitLayer
        print(f"[boot] ready: {len(GRID)} origins ({len(SNAPPED_GRID)} on-network). "
              f"Open http://127.0.0.1:8000")
    else:
        NET = SNAPPED_GRID = _TRANSIT_LAYER = None
        _EXACT_IDS, _EXACT_GEOMS = [], []
        print(f"[boot] ready: {len(GRID)} origins (JVM-free; bundle cached). "
              f"Open http://127.0.0.1:8000")

# Thread pool for the EXACT recompute. R5 routing is read-only against the shared warm
# network and each task clones its own Java RegionalTask, so per-origin routing parallelises
# safely in this one JVM. The Java RAPTOR releases the GIL, but Python-side result extraction
# does not, so the real speedup tops out ~4.8x near the physical-core count. We default the
# pool to (cores - 2) to leave the machine thermal/interactive headroom (the fan). Override
# with EXACT_THREADS=N.
_N_PHYS = os.cpu_count() or 8
EXACT_THREADS = int(os.environ.get("EXACT_THREADS", str(max(2, _N_PHYS - 2))))
_EXACT_POOL = ThreadPoolExecutor(max_workers=EXACT_THREADS, thread_name_prefix="r5-exact")


# ---- Bounded LRU for the workplace-keyed caches ----------------------------------------
class _BoundedLRU:
    """Thread-safe bounded LRU — the ONE eviction implementation behind every workplace-keyed
    cache below (result/egress/tree/MC; previously four hand-rolled lock+move_to_end+popitem
    copies). ``copy_mode`` preserves each cache's load-bearing copy policy (cache-poisoning
    fixes from a past audit — do NOT weaken when adding a cache):
      None      — hand out the stored object itself (caller must treat it as immutable,
                  e.g. the egress tuple of numpy arrays);
      'shallow' — store as-is, return ``dict(value)`` on get (callers can't reassign the
                  cached entry's keys; the values themselves stay shared);
      'deep'    — deepcopy on put AND get (full isolation for mutable JSON-able results).
    The lock is REENTRANT and exposed as ``.lock`` so composite check-then-act sections
    (the MC in-flight registry) and the tests' clear/pop-under-lock pattern can wrap several
    operations atomically without deadlocking on the methods' own acquisition."""

    def __init__(self, maxsize, copy_mode=None, lock=None):
        self.maxsize = int(maxsize)
        self._copy_mode = copy_mode
        self._od = OrderedDict()
        self.lock = lock if lock is not None else threading.RLock()

    def _out(self, value):
        if self._copy_mode == "deep":
            return copy.deepcopy(value)
        if self._copy_mode == "shallow":
            return dict(value)
        return value

    def get(self, key, default=None):
        with self.lock:
            if key in self._od:
                self._od.move_to_end(key)
                return self._out(self._od[key])
        return default

    def put(self, key, value):
        with self.lock:
            self._od[key] = copy.deepcopy(value) if self._copy_mode == "deep" else value
            self._od.move_to_end(key)
            while len(self._od) > self.maxsize:
                self._od.popitem(last=False)

    def pop(self, key, default=None):
        with self.lock:
            return self._od.pop(key, default)

    def clear(self):
        with self.lock:
            self._od.clear()

    def __contains__(self, key):
        with self.lock:
            return key in self._od

    def __len__(self):
        with self.lock:
            return len(self._od)


# ---- RAPTOR engine (flag-gated grid travel-times; flags parsed up top) ----------------
# When USE_WALK_GRAPH the walk-baked (slope-aware) access table + the JVM-free walk router serve
# the whole map path; otherwise R5 stays in-process for the walk matrix (and, under departafter,
# the R5 hover/color-by-line).
_RAPTOR = None
_RAPTOR_STOPS = None             # GeoDataFrame of stop coords keyed by gid (egress destinations)
_WG = None                       # WalkGraph (JVM-free) when USE_WALK_GRAPH
_WG_STOP_NODES = _WG_STOP_CONN = _WG_CELL_NODES = _WG_CELL_CONN = None
_WG_STOP_GIDS = None             # gids aligned to _WG_STOP_NODES rows
# ~24 workplace buckets x 3 numpy arrays (~1 MB/entry) — covers a small crowd of distinct
# workplaces without growing past a few tens of MB. No copy: the tuple + arrays are treated
# as immutable by every consumer (the engine reads, never writes them).
_EGRESS_CACHE_MAX = 24
_RAPTOR_EGRESS_CACHE = _BoundedLRU(_EGRESS_CACHE_MAX)  # coarse_key -> (egress_g, egress_w, purewalk)
if USE_RAPTOR:
    try:
        from core import raptor_engine
        import numpy as _np
        _acc_path = None
        if USE_WALK_GRAPH:                       # prefer the walk-baked (slope-aware) access table
            import core.raptor_build as _rb
            # _acc_fp, NOT _fp: the module-global _fp above is the static-bundle GTFS
            # fingerprint (stamped into server_static.json) — reusing the name here would
            # silently break the bundle's repull invalidation if the write ever moved later.
            _acc_fp = _rb._fingerprint(GTFS, _SVC_DATE.strftime("%Y%m%d"),
                                       _rb.band_seconds(), _rb.FOOTPATH_M)
            # exact name, NOT a glob: a leftover bake at another resolution would sort first
            # and silently swap the engine onto a different grid than the server's GRID_M
            _acc_path = config.DATA / "raptor_cache" / f"access_walk_{GRID_M}m_{_acc_fp}.npz"
            if not _acc_path.exists():
                raise FileNotFoundError(
                    f"USE_WALK_GRAPH set but {_acc_path.name} is missing "
                    f"(run scripts/build_walk_graph.py + bake_walk_access.py"
                    + (f" with GRID_M={GRID_M}" if GRID_M != config.GRID_M else "") + ")")
            # Staleness guard: the bake embeds the sha256 of the walk_graph.npz it was built
            # from. A graph rebuild without a rebake would silently mix access times from the
            # OLD graph with live egress/pure-walk from the NEW one — fail loudly instead.
            import hashlib as _hashlib
            _wg_sha = _hashlib.sha256((config.DATA / "walk_graph.npz").read_bytes()).hexdigest()
            with _np.load(_acc_path, allow_pickle=True) as _zacc:
                _baked_sha = (str(_zacc["walk_graph_sha"]) if "walk_graph_sha" in _zacc.files
                              else None)
            if _baked_sha is None:                # pre-fingerprint bake: warn, don't refuse
                print(f"[boot] ({_acc_path.name} predates the walk-graph fingerprint; rerun "
                      f"scripts/bake_walk_access.py to enable the staleness check)")
            elif _baked_sha != _wg_sha:
                raise RuntimeError(
                    f"{_acc_path.name} was baked against a DIFFERENT walk_graph.npz — "
                    f"rerun scripts/bake_walk_access.py")
        _RAPTOR = raptor_engine.RaptorEngine(GTFS, _SVC_DATE, access_path=_acc_path, verbose=True)
        _gl, _go = _RAPTOR.data["stop_lat"], _RAPTOR.data["stop_lon"]
        _gids = [g for g in range(_RAPTOR.data["n_stops"]) if not _np.isnan(_gl[g])]
        # id="S<gid>" so stop ids stay DISJOINT from cell ids ("0".."N") and the origin "w"
        # (R5's matrix returns 0 travel time when an origin id string == a destination id).
        # _RAPTOR_STOPS (a GeoDataFrame) is only used by the R5 egress walk matrix; in the lean
        # JVM-free path the walk router snaps stop coords from numpy below, so skip it (gpd/Point
        # aren't even imported in lean mode).
        if _NEED_R5:
            _RAPTOR_STOPS = gpd.GeoDataFrame(
                {"id": ["S" + str(g) for g in _gids]},
                geometry=[Point(_go[g], _gl[g]) for g in _gids], crs=config.WGS)
        # align purewalk to the engine's cell order (server grid == engine grid by id)
        _RAPTOR_CELL_POS = {c: i for i, c in enumerate(_RAPTOR.cell_ids)}
        if USE_WALK_GRAPH:
            from core import walk as _walkmod
            _WG = _walkmod.WalkGraph.load()
            _WG_STOP_GIDS = _np.asarray(_gids, dtype=_np.int32)
            _WG_STOP_NODES, _WG_STOP_CONN = _WG.snap(
                _np.column_stack(([_go[g] for g in _gids], [_gl[g] for g in _gids])))
            _cll = _np.array([[ORIGIN_LL[c][1], ORIGIN_LL[c][0]] for c in _RAPTOR.cell_ids])
            _WG_CELL_NODES, _WG_CELL_CONN = _WG.snap(_cll)
            print(f"[boot] RAPTOR engine ON (semantic={RAPTOR_SEMANTIC}); WALK GRAPH ON "
                  f"(JVM-free walk: {_acc_path.name}) — R5 walk not used")
        else:
            print(f"[boot] RAPTOR engine ON (semantic={RAPTOR_SEMANTIC}); R5 kept for hover only")
    except Exception as _e:                     # missing access table etc.
        if not _NEED_R5:
            # JVM-free mode: R5 was never imported, so there is NO fallback — fail loudly.
            raise RuntimeError(f"USE_WALK_GRAPH engine init failed and R5 is not loaded: {_e}")
        print(f"[boot] USE_RAPTOR requested but engine init failed ({_e}); using R5 path")
        USE_RAPTOR = False
        USE_WALK_GRAPH = False

# ---- Generation / cancel token --------------------------------------------------------
# Each new workplace (/compute) bumps _GENERATION. The long all-cores jobs (compute_exact,
# the lazy attribution prewarm) read the generation they started under and, between origin
# waves, bail the moment it changes — so rapidly retyping the address does not stack multiple
# ~14-32s exact bursts. A superseded /compute_exact returns HTTP 409 (the frontend keeps the
# fast approximation); a superseded prewarm just stops.
_GENERATION = 0
_GEN_LOCK = threading.Lock()


def _bump_generation():
    global _GENERATION
    with _GEN_LOCK:
        _GENERATION += 1
        return _GENERATION


def _current_generation():
    with _GEN_LOCK:
        return _GENERATION


class _Superseded(Exception):
    """Raised inside a heavy job when a newer workplace has been set (cancel token)."""


# Serialise heavy routing jobs (/compute_exact, the lazy /attribution build) against each
# other so one heavy request can use all cores without two thrashing them. Light /itinerary
# (one OD pair) and the fast /compute do NOT take this lock, so hover stays responsive while
# a heavy refine/attribution job runs.
_HEAVY_LOCK = threading.Lock()

# ---- Per-workplace itinerary + attribution caches -------------------------------------
# Recorded-path breakdowns, keyed by workplace. The full-grid build (color-by-line) and the
# per-cell on-demand build (hover) are cached separately so a later full build can replace the
# dest's dict wholesale without dropping hovered cells. Both are CLEARED on every new workplace
# (_reset_caches), so memory stays bounded to the current workplace instead of growing forever.
_ITIN_CACHE = {}                 # dest_key -> {cellId: itinerary dict}  (full-grid build)
_ITIN_INFLIGHT = {}              # dest_key -> threading.Event (full-grid build in progress)
_ITIN_CACHE_LOCK = threading.Lock()
_CELL_CACHE = {}                 # dest_key -> {cellId: itinerary dict}  (per-cell on demand)
_CELL_CACHE_LOCK = threading.Lock()
_LAST_DEST_KEY = None            # workplace of the last /compute; a CHANGE clears the caches above


def _dest_key(lat, lon, max_rides=DEFAULT_MAX_RIDES, speed=None):
    """Per-workplace cache key for the breakdown caches (_ITIN_CACHE/_CELL_CACHE/
    _ITIN_INFLIGHT) and the _LAST_DEST_KEY change-detector. A capped result is a DIFFERENT
    journey than the uncapped one, so the cap is part of the key — otherwise a maxrides=1
    breakdown would poison the uncapped cell (and vice-versa). The walk SPEED likewise changes
    the journey, so it's keyed too. At the R5 default + medium speed the key is the same 2-tuple
    as before, so the baseline (and existing key callers) is unchanged."""
    base = (round(float(lat), 5), round(float(lon), 5))
    if max_rides != DEFAULT_MAX_RIDES:
        base = base + (int(max_rides),)
    if speed and speed != DEFAULT_SPEED:
        base = base + (speed,)
    return base


def _reset_caches():
    """Drop cached breakdowns for previous workplaces (called on each new /compute). An
    in-flight build for an old workplace is already a different generation, so it bails
    between waves (returns {}) — and if it completes during its FINAL wave, the gen re-check
    in _itineraries_cached keeps it from re-populating the cache we just cleared. Either way
    it pops its own in-flight marker; we only clear the result dicts here."""
    with _ITIN_CACHE_LOCK:
        _ITIN_CACHE.clear()
    with _CELL_CACHE_LOCK:
        _CELL_CACHE.clear()


class _Busy(Exception):
    """Raised when a heavy full-grid build is needed but _HEAVY_LOCK is already held; the
    /attribution route turns this into a 503 instead of blocking behind the running job."""


def _itineraries_cached(dlat, dlon, *, nonblock=False, max_rides=DEFAULT_MAX_RIDES):
    """Full per-cell itinerary map for (dlat,dlon) -> {cellId: itin dict}, built LAZILY (only
    on demand from /attribution) and cached. If another thread is already building this
    destination, wait for its result (in-flight de-dup). The build takes _HEAVY_LOCK so it
    serialises against /compute_exact.

    With ``nonblock=True``: if THIS call is the one that would do the build and _HEAVY_LOCK is
    already held by another heavy job, raise _Busy (-> 503) instead of blocking. A call that
    is merely WAITING on another thread's in-flight build still waits (it isn't the one doing
    the work) — but if the owner finished WITHOUT caching anything (it hit _Busy or was
    superseded), the waiter raises _Busy too, so both callers get the same retryable 503
    instead of the waiter getting a successful-looking empty 200. A cache hit never blocks
    regardless. ``max_rides`` is part of the cache key so a capped build can't be served for
    an uncapped request."""
    key = _dest_key(dlat, dlon, max_rides)
    with _ITIN_CACHE_LOCK:
        if key in _ITIN_CACHE:
            return _ITIN_CACHE[key]
        event = _ITIN_INFLIGHT.get(key)
        if event is None:
            event = threading.Event()
            _ITIN_INFLIGHT[key] = event
            owner = True
        else:
            owner = False
    if not owner:                # someone else is building it; wait for their result
        event.wait()
        with _ITIN_CACHE_LOCK:
            res = _ITIN_CACHE.get(key)
        if res is None:          # owner cached nothing (busy/superseded/empty) -> retry later,
            raise _Busy()        # mirroring the 503 the owner's own request reported
        return res
    gen = _current_generation()  # cancel token: bail if a newer workplace is set mid-build
    locked = False
    try:
        if nonblock:
            if not _HEAVY_LOCK.acquire(blocking=False):
                raise _Busy()
        else:
            _HEAVY_LOCK.acquire()
        locked = True
        itins = prewarm_itineraries(dlat, dlon, gen, max_rides)
        # Never cache an empty result (a superseded/partial build would poison this workplace
        # until the next /compute). The gen re-check closes the last-wave race: _map_cancelable
        # only checks the generation BETWEEN waves, so a /compute that bumped the generation and
        # ran _reset_caches during the FINAL wave would otherwise see this stale full grid
        # written back into the freshly-cleared _ITIN_CACHE (megabytes kept until the next
        # workplace change).
        if itins and _current_generation() == gen:
            with _ITIN_CACHE_LOCK:
                _ITIN_CACHE[key] = itins
        return itins
    except _Superseded:
        return {}                # newer workplace set; abandon this stale full-grid build
    finally:
        if locked:
            _HEAVY_LOCK.release()
        with _ITIN_CACHE_LOCK:
            _ITIN_INFLIGHT.pop(key, None)
        event.set()


def _dominant_line(itin):
    """The "color by line" attribution for an itinerary: the transit leg carrying the most
    ride time, or 'walk only' when the trip has no transit legs."""
    rides = [l for l in itin["legs"] if l["mode"] != "walk"]
    if not rides:
        return "walk only"
    return max(rides, key=lambda l: l["min"])["line"]


def _attribution_from_cache(dlat, dlon, *, nonblock=False, max_rides=DEFAULT_MAX_RIDES):
    """{cellId: dominantLine} derived from the shared per-cell itinerary cache (building the
    full grid lazily if needed). ``nonblock`` propagates the non-blocking heavy-lock policy
    (raise _Busy -> 503 instead of queueing behind a running heavy job). ``max_rides`` keys
    the underlying itinerary cache."""
    itins = _itineraries_cached(dlat, dlon, nonblock=nonblock, max_rides=max_rides)
    return {cid: _dominant_line(it) for cid, it in itins.items()}


# ---- Coarse-coords RESULT cache for the heavy endpoints (A4) ---------------------------
# A bounded LRU caching the FINAL JSON-able results of /compute_exact and /attribution,
# keyed by (round(lat,3), round(lon,3)) (~110m). A hit returns the same result with no R5
# work and without taking _HEAVY_LOCK — so a repeated/nearby workplace (e.g. localStorage
# auto-restore, or panning back) is instant. This is SEPARATE from the per-workplace
# itinerary caches (_ITIN_CACHE/_CELL_CACHE), which key on the finer _dest_key and back
# /itinerary. copy_mode='deep' (put AND get) so callers can never mutate the store.
# 150 entries x two small JSON-able dicts (~100s of KB each) — generous because hits are the
# whole point (instant repeat workplaces) and the payloads are cheap relative to the tree/MC
# caches. The lock is SHARED by both caches (tests clear them together under it).
_RESULT_CACHE_MAX = 150
_RESULT_CACHE_LOCK = threading.RLock()
_EXACT_RESULT_CACHE = _BoundedLRU(_RESULT_CACHE_MAX, copy_mode="deep",
                                  lock=_RESULT_CACHE_LOCK)   # coarse_key -> {id: [best, real]}
_ATTR_RESULT_CACHE = _BoundedLRU(_RESULT_CACHE_MAX, copy_mode="deep",
                                 lock=_RESULT_CACHE_LOCK)    # coarse_key -> {cellId: dominantLine}


def _coarse_key(lat, lon, max_rides=DEFAULT_MAX_RIDES, speed=None):
    """~110m bucket for the heavy-result caches (_EXACT/_ATTR/_RAPTOR_TREE/_RAPTOR_MC). Like
    _dest_key, the transfer cap AND the walk speed are part of the key so a capped/slow result
    can't be served for an uncapped/medium request. At the R5 default + medium speed the key is
    the same 2-tuple as before, so the baseline cache behavior (and existing callers) is
    unchanged. (The egress/pure-walk cache stays speed-free: it's reference seconds, the engine
    applies the speed scalar.)"""
    base = (round(float(lat), 3), round(float(lon), 3))
    if max_rides != DEFAULT_MAX_RIDES:
        base = base + (int(max_rides),)
    if speed and speed != DEFAULT_SPEED:
        base = base + (speed,)
    return base


def _req_max_rides():
    """Parse the ``maxrides`` query param into an R5 ride cap. Absent/blank/invalid ->
    DEFAULT_MAX_RIDES (today's behavior). The frontend sends rides directly (transfers+1):
    0 transfers -> 1, 1 -> 2, 2 -> 3, "Any" -> 8. We clamp to [1, DEFAULT_MAX_RIDES] so a
    bogus value can't disable transit (rides must be >= 1) or exceed the model default."""
    raw = request.args.get("maxrides")
    if raw is None or not str(raw).strip():
        return DEFAULT_MAX_RIDES
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_RIDES
    return max(1, min(DEFAULT_MAX_RIDES, v))


# Walk-speed toggle (RAPTOR only): scalar = 4.8 / pace. The engine multiplies every WALK
# reference-second (access/egress/pure-walk, baked @4.8) by it. Default medium (the user is a
# fast walker but most aren't); the access table / egress stay reference seconds + cached once.
# The presets live in core.config (beside WALK_KMH, their reference speed).
WALK_SPEEDS = config.WALK_SPEEDS
DEFAULT_SPEED = config.DEFAULT_SPEED


def _req_speed():
    """(speed_name, walk_scalar) from ?speed=slow|med|fast; default medium (scalar 1.0)."""
    s = (request.args.get("speed") or "").lower()
    if s not in WALK_SPEEDS:
        return DEFAULT_SPEED, 1.0
    return s, config.WALK_KMH / WALK_SPEEDS[s]


app = Flask(__name__)

# ---- Rate limiting (Flask-Limiter, in-memory) -----------------------------------------
# Single-process app, so the default in-memory store is correct (no Redis needed). Per-IP
# limits are applied per endpoint below via @limiter.limit. Choosing the right IP source is
# load-bearing for the limits to be real:
#   - BOTH `CF-Connecting-IP` and `X-Forwarded-For` are forgeable by any client that reaches
#     the origin directly — a proxy header is only trustworthy when the proxy chain is
#     guaranteed. In production that guarantee comes from the cloudflare-ufw firewall
#     (80/443 restricted to CF CIDRs) plus TRUST_PROXY=1 in sfci.service; the headers
#     themselves prove nothing. So gate BOTH on TRUST_PROXY, preferring CF-Connecting-IP
#     (when traffic really transits Cloudflare it's the canonical real-client header,
#     whereas XFF can carry a client-supplied first hop that Caddy appends to rather than
#     replaces). Note: with TRUST_PROXY=1 a direct-to-origin attacker defeats per-IP limits
#     regardless of which header we read — the firewall, not this function, is the actual
#     abuse boundary.
#   - Without TRUST_PROXY, use the socket peer (correct for localhost dev / direct hits).
def _client_ip():
    if os.environ.get("TRUST_PROXY", "").lower() in ("1", "true", "yes"):
        cf = request.headers.get("CF-Connecting-IP", "").strip()
        if cf:
            return cf
        fwd = request.headers.get("X-Forwarded-For", "")
        if fwd:
            first = fwd.split(",")[0].strip()
            if first:
                return first
    return request.remote_addr or "127.0.0.1"


def _parse_ll(lat_raw, lon_raw):
    """Parse + validate lat/lon from query args — used by EVERY coordinate-taking endpoint.
    Rejects missing/non-numeric/NaN/inf coords with a friendly JSON 400 instead of a 500
    (NaN is a valid float() but never compares equal, so it would bypass every coord-keyed
    cache), and rejects out-of-SF-region coords so a coordinate the engine can't sensibly
    serve doesn't burn a full compute for a garbage result."""
    try:
        lat = float(lat_raw); lon = float(lon_raw)
    except (TypeError, ValueError):
        raise _BadRequest("lat/lon must be numeric")
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise _BadRequest("lat/lon must be finite")
    lo_min, la_min, lo_max, la_max = config.SF_VALID_BBOX   # loose box; see core/config.py
    if not (la_min <= lat <= la_max and lo_min <= lon <= lo_max):
        raise _BadRequest(f"lat/lon outside SF region: {lat:.4f},{lon:.4f}")
    return lat, lon


class _BadRequest(Exception):
    pass


@app.errorhandler(_BadRequest)
def _bad_request(e):
    return jsonify({"error": "bad_request", "detail": str(e)}), 400


limiter = Limiter(key_func=_client_ip, app=app, storage_uri="memory://")


# Dynamic-data endpoints whose response depends on workplace/speed/transfer params. Geocode and
# autocomplete are pure functions of the query string -> let the browser cache them. The page
# bundle (GET /) is workplace-agnostic and shipped once at boot -> stays cacheable + bfcache-
# eligible so back-navigation doesn't refetch the network and rerun /compute + /variance.
_NO_STORE_PATHS = frozenset({"/compute", "/compute_exact", "/itinerary", "/attribution", "/variance"})


@app.after_request
def _no_cache(resp):
    """Disable browser/bf-cache for the dynamic API endpoints only. Without no-store, a heuristic
    cache hit on /itinerary or /variance (URL identical across runs) would silently hide a
    server-side fix and could serve a stale response after a workplace/speed change."""
    if request.path in _NO_STORE_PATHS:
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.errorhandler(429)
def _ratelimit_json(e):
    """Flask-Limiter's default 429 page is HTML, so the frontend's r.json() throws and the user
    just sees 'error' in the tooltip. Return JSON so callers can handle it (retry / show toast)."""
    return jsonify({"error": "rate_limited", "detail": str(getattr(e, "description", e))}), 429


# ---- RAPTOR grid travel-times (flag-gated) --------------------------------------------
def _raptor_egress_purewalk(lat, lon):
    """Per-workplace inputs for the RAPTOR engine, via ONE-origin R5 WALK matrices:
      egress_g/egress_w — W->stop walk seconds (gid-keyed), capped at the access cap;
      purewalk          — W->cell walk seconds (cell order), capped at MAX_MIN.
    The only R5 use on the RAPTOR map path (one light walk tree, not the heavy per-cell pass).
    Cached per ~110m workplace bucket."""
    import numpy as _np
    ckey = _coarse_key(lat, lon)
    cached = _RAPTOR_EGRESS_CACHE.get(ckey)
    if cached is not None:
        return cached
    if USE_WALK_GRAPH:                              # JVM-free hill-aware walk router (no R5)
        Wll = (lon, lat)
        ecap = _RAPTOR.access_cap_min * 60          # reference seconds (engine applies speed scalar)
        # egress = stop -> W (alight->work) and pure-walk = cell -> W (home->work): both rooted at
        # W on the TRANSPOSED graph so uphill/downhill is correct for the actual walk direction.
        eg = _WG.one_to_many(Wll, _WG_STOP_NODES, _WG_STOP_CONN, ecap, reverse=True)
        fin = _np.isfinite(eg)
        egress_g = _WG_STOP_GIDS[fin].astype(_np.int32)
        egress_w = _np.rint(eg[fin]).astype(_np.int64)
        pw = _WG.one_to_many(Wll, _WG_CELL_NODES, _WG_CELL_CONN, config.MAX_MIN * 60, reverse=True)
        purewalk = _np.where(_np.isfinite(pw), _np.rint(pw), -1).astype(_np.int64)
        res = (egress_g, egress_w, purewalk)
        _RAPTOR_EGRESS_CACHE.put(ckey, res)
        return res
    # Legacy R5 walk-matrix branch. Parsing is SHARED with raptor_oracle.egress_purewalk via
    # core.r5_extract (min-dedup, minutes->seconds, -1 sentinel) so the goldens the engine
    # validates against can't drift from what this live path computes.
    from core import r5_extract
    W = gpd.GeoDataFrame({"id": ["w"]}, geometry=[Point(lon, lat)], crs=config.WGS)
    cap = _RAPTOR.access_cap_min
    # egress: W -> stops (walk); our stop ids are "S<gid>" so the id bridge is just the strip
    e = pd.DataFrame(network.walk_time_matrix(NET, W, _RAPTOR_STOPS, DEP, cap))
    egress_g, egress_w = r5_extract.egress_from_ttm(e, lambda to: int(to[1:]), dtype=_np.int64)
    # pure walk: W -> cells, aligned to the engine's cell order
    pw_ttm = pd.DataFrame(network.walk_time_matrix(NET, W, SNAPPED_GRID, DEP, config.MAX_MIN))
    purewalk = r5_extract.purewalk_from_ttm(pw_ttm, _RAPTOR_CELL_POS,
                                            len(_RAPTOR.cell_ids), dtype=_np.int64)
    res = (egress_g, egress_w, purewalk)
    _RAPTOR_EGRESS_CACHE.put(ckey, res)
    return res


# Per-workplace traced tree (Phase 2): one arrive-by tree serves the MAP (actual commute),
# the hover breakdown, AND color-by-line — so they are guaranteed consistent (hover == map).
# 8 entries: each holds a full JourneyTree whose lazy full-grid trace memo runs to tens of MB,
# so this (like the MC cache below) stays an order of magnitude smaller than the result LRUs.
# copy_mode='shallow': callers get dict(entry) so they can't reassign the cached entry's keys
# (tree/cells/dom values themselves are shared and treated as immutable).
_TREE_CACHE_MAX = 8
_RAPTOR_TREE_CACHE = _BoundedLRU(_TREE_CACHE_MAX, copy_mode="shallow")
# (coarse_key incl. rides+speed) -> {tree, cells, dom, geom}

# ---- Hover route GEOMETRY (the drawn journey) ------------------------------------------
# Walk-leg paths come from PathTree predecessor chains over the SAME walk graph the times
# use. The expensive piece is the workplace-rooted REVERSE tree (one ~75-min-cap Dijkstra
# serving every cell's egress + pure-walk path), so it's cached per ~110m workplace bucket;
# entries are ~10 MB (k=4 dist+pred over the 215k-node graph), hence the small bound. Walk
# paths are SPEED-INVARIANT (the scalar multiplies every edge uniformly — route choice and
# node chains don't change), so the key carries no speed/rides.
_WALKPATH_TREE_CACHE = _BoundedLRU(4)        # coarse (lat,lon) -> walk.PathTree (reverse, W-rooted)
# Assembled per-cell geometry responses are cached INSIDE the tree-cache entry ("geom":
# {ci: response dict}) so they live and die with the traced tree they were derived from.
# The entry dict's values are shared across the LRU's shallow copies; writes go through
# this lock (the documented rule for mutating a cached entry's values).
_GEOM_LOCK = threading.Lock()


class _JourneyGeomProvider:
    """Walk-leg geometry provider for ``JourneyTree.itinerary(ci, geom_provider=...)``.
    Real street paths when the walk graph is loaded (_WG): the workplace-rooted reverse
    PathTree serves egress + pure-walk legs (cached per workplace, warm = predecessor-chain
    walking only), per-cell forward trees serve the access leg, and tiny capped trees serve
    the <=250m synthesized transfer footpaths. Without the graph (R5-walk boot) every walk
    leg degrades to a straight 2-point segment marked approx=True. All methods return
    ([[lat, lon], ...], approx) with the EXACT endpoint coords prepended/appended (cell
    center / stop / workplace), so the drawn legs visibly connect."""

    _TRANSFER_CAP_REF = 900            # ref-sec Dijkstra cap for <=250m footpaths (street detours)

    def __init__(self, dlat, dlon):
        self.dlat, self.dlon = float(dlat), float(dlon)
        self._wtree = None             # lazy: workplace-rooted reverse PathTree
        self._cell_trees = {}          # ci -> forward PathTree (per-request; results are cached)

    # -- coordinate helpers ----------------------------------------------------------------
    def _stop_ll(self, g):
        la = float(_RAPTOR.data["stop_lat"][g]); lo = float(_RAPTOR.data["stop_lon"][g])
        if math.isnan(la) or math.isnan(lo):
            return None
        return (la, lo)

    def _cell_ll(self, ci):
        return ORIGIN_LL[_RAPTOR.cell_ids[ci]]

    @staticmethod
    def _straight(a, b):
        """Fallback 2-point segment (off-graph snap / missing coords / no walk graph)."""
        pts = []
        for p in (a, b):
            if p is None:
                continue
            q = [round(p[0], 6), round(p[1], 6)]
            if not pts or pts[-1] != q:
                pts.append(q)
        return pts, True

    @staticmethod
    def _seal(pts, start_ll, end_ll):
        """Pin the path's ends to the exact off-graph endpoints (the snap connector is a
        straight offset, so the chain starts/ends at the nearest NODE otherwise)."""
        out = [[round(start_ll[0], 6), round(start_ll[1], 6)]] + pts + \
              [[round(end_ll[0], 6), round(end_ll[1], 6)]]
        return out, False

    # -- trees -------------------------------------------------------------------------------
    def _w_tree(self):
        if self._wtree is None and _WG is not None:
            ckey = _coarse_key(self.dlat, self.dlon)
            t = _WALKPATH_TREE_CACHE.get(ckey)
            if t is None:
                t = _WG.path_tree((self.dlon, self.dlat), config.MAX_MIN * 60, reverse=True)
                _WALKPATH_TREE_CACHE.put(ckey, t)
            self._wtree = t
        return self._wtree

    def _cell_tree(self, ci):
        t = self._cell_trees.get(ci)
        if t is None and _WG is not None:
            la, lo = self._cell_ll(ci)
            # 2x the access cap: the table's times are <= cap, but an R5-baked table's path
            # can run longer on OUR graph — generous so extraction never starves the cap.
            t = _WG.path_tree((lo, la), _RAPTOR.access_cap_min * 60 * 2)
            self._cell_trees[ci] = t
        return t

    # -- the four walk-leg kinds (JourneyTree._geometry contract) ----------------------------
    def access(self, ci, s_star):
        cl = self._cell_ll(ci); sl = self._stop_ll(s_star)
        t = self._cell_tree(ci)
        if t is None or sl is None:
            return self._straight(cl, sl)
        pts = t.path_points((sl[1], sl[0]))
        return self._straight(cl, sl) if pts is None else self._seal(pts, cl, sl)

    def egress(self, s):
        sl = self._stop_ll(s); W = (self.dlat, self.dlon)
        t = self._w_tree()
        if t is None or sl is None:
            return self._straight(sl, W)
        pts = t.path_points((sl[1], sl[0]))
        return self._straight(sl, W) if pts is None else self._seal(pts, sl, W)

    def purewalk(self, ci):
        cl = self._cell_ll(ci); W = (self.dlat, self.dlon)
        t = self._w_tree()
        if t is None:
            return self._straight(cl, W)
        pts = t.path_points((cl[1], cl[0]))
        return self._straight(cl, W) if pts is None else self._seal(pts, cl, W)

    def transfer(self, s, j):
        sl = self._stop_ll(s); jl = self._stop_ll(j)
        if _WG is None or sl is None or jl is None:
            return self._straight(sl, jl)
        t = _WG.path_tree((sl[1], sl[0]), self._TRANSFER_CAP_REF)
        pts = t.path_points((jl[1], jl[0]))
        return self._straight(sl, jl) if pts is None else self._seal(pts, sl, jl)


def _raptor_tree(lat, lon, max_rides=DEFAULT_MAX_RIDES, speed=DEFAULT_SPEED, walk_scalar=1.0):
    """Build (or fetch) the cached JourneyTree + per-cell map values + dominant lines for this
    workplace (the Phase-2 path layer). Both semantics serve the map, hover breakdown, and
    color-by-line from ONE cached tree so hover==map by construction and JVM-free:

      arrive-by   -> a single-deadline ``JourneyTree``; ``cells[c]`` = [commute, commute] (the
                     actual door-to-door commute, p5==p50 since there's one tree).
      depart-after-> a ``DepartAfterJourneyTree`` over the depart-after window; ``cells[c]`` =
                     [p5, p50] (the SAME painted percentiles ``engine.departafter`` returns), with
                     the MAP COLOR = p50 (== ``DepartAfterJourneyTree.commute()`` -> the itinerary
                     total, so itinerary==map). The map color uses ``commute()`` (NO trees traced);
                     color-by-line uses ``dominant()`` (lazy per-T* trees, shared with hover).

    Tracing/painting is ~0.1-0.2s, done once per workplace bucket (keyed by transfer cap + walk
    speed). The entry shape ({tree, cells, dom, geom}) is identical for both semantics so every
    downstream reader (compute/itinerary/attribution) is semantic-agnostic."""
    key = _coarse_key(lat, lon, max_rides, speed)
    hit = _RAPTOR_TREE_CACHE.get(key)
    if hit is not None:
        return hit
    egress_g, egress_w, purewalk = _raptor_egress_purewalk(lat, lon)
    if RAPTOR_SEMANTIC == "departafter":
        # Depart-after p50 map color comes straight from the selection kernel (commute(), no trees);
        # p5 from a second cheap painted percentile pass. The dominant line is NOT computed here:
        # dominant() would trace the window's ~20 per-T* trees (~0.9s), and only color-by-line needs
        # it — so it's built lazily on the first /attribution (cached inside the tree). Keeping it off
        # this path is what holds the /compute + hover build to ~80ms.
        tree = _RAPTOR.journey_tree_departafter(egress_g, egress_w, purewalk, percentile=50.0,
                                                max_rounds=int(max_rides), walk_scalar=walk_scalar)
        commute = tree.commute()                          # p50 painted minute, == itinerary total
        # p5 (the "best-case" of the depart-after window) — same painted-percentile source as
        # engine.departafter's p5, so cells[c] = [p5, p50] matches the served /compute shape.
        p5 = _RAPTOR.departafter(egress_g, egress_w, purewalk, percentiles=(5,),
                                 max_rounds=int(max_rides), walk_scalar=walk_scalar)
        cells = {c: ([int(p5[c][0]) if p5[c][0] is not None else int(commute[i]),
                      int(commute[i])] if commute[i] >= 0 else [None, None])
                 for i, c in enumerate(_RAPTOR.cell_ids)}
        domd = None                                       # lazy; filled by _raptor_attribution
    else:
        tree = _RAPTOR.journey_tree(egress_g, egress_w, purewalk, max_rounds=int(max_rides),
                                    walk_scalar=walk_scalar)
        commute, dom = tree.commute_and_dominant()        # arrive-by: one tree, ~0.1s — eager is fine
        cells = {c: ([int(commute[i]), int(commute[i])] if commute[i] >= 0 else [None, None])
                 for i, c in enumerate(_RAPTOR.cell_ids)}
        domd = {c: dom[i] for i, c in enumerate(_RAPTOR.cell_ids) if dom[i] is not None}
    # "geom": per-cell hover-route geometry responses (ci -> /itinerary dict incl. "geom"),
    # filled lazily by /itinerary under _GEOM_LOCK; shared across the LRU's shallow copies
    # so it lives exactly as long as the tree it was traced from.
    entry = {"tree": tree, "cells": cells, "dom": domd, "geom": {}}
    _RAPTOR_TREE_CACHE.put(key, entry)
    return entry


def compute_raptor(lat, lon, max_rides=DEFAULT_MAX_RIDES, speed=DEFAULT_SPEED, walk_scalar=1.0):
    """Grid travel-times via the RAPTOR engine -> {id: [best, real]} (same shape as compute).
    BOTH semantics now serve from the cached traced tree so map==refine==hover (JVM-free):
    arrive-by uses the single-deadline tree (actual commute); depart-after uses the
    DepartAfterJourneyTree's painted [p5, p50] (the p50 map color == the breakdown total)."""
    return _raptor_tree(lat, lon, max_rides, speed, walk_scalar)["cells"]


def _nearest_raptor_cell(olat, olon):
    """Index of the nearest on-grid cell to an off-grid hover point (RAPTOR can only trace grid
    cells, since the access table is per-cell). Good enough for the occasional off-grid hover."""
    best_i, best_d = None, 1e30
    for i, cid in enumerate(_RAPTOR.cell_ids):
        la, lo = ORIGIN_LL.get(cid, (None, None))
        if la is None:
            continue
        d = (la - olat) ** 2 + (lo - olon) ** 2
        if d < best_d:
            best_d = d; best_i = i
    return best_i


def _raptor_attribution(dlat, dlon, max_rides=DEFAULT_MAX_RIDES, speed=DEFAULT_SPEED,
                        walk_scalar=1.0):
    """{cellId: dominant line} from the cached tree (color-by-line, no R5). Arrive-by populates
    ``dom`` eagerly at build time (one tree, ~0.1s). Depart-after leaves it None and traces it
    LAZILY here on the first color-by-line request (``dominant()`` traces the window's ~20 per-T*
    trees, ~0.9s — kept OFF the /compute + hover path); the result is cached ON THE TREE (not the
    entry, which is a shallow copy), and the tree is shared by reference across cache copies, so
    repeats are free."""
    entry = _raptor_tree(dlat, dlon, max_rides, speed, walk_scalar)
    dom = entry["dom"]
    if dom is None:                                       # depart-after: trace on demand (tree-cached)
        dl = entry["tree"].dominant()
        dom = {c: dl[i] for i, c in enumerate(_RAPTOR.cell_ids) if dl[i] is not None}
    return dom


# Per-workplace service-noise Monte-Carlo (realistic + fragility + alt-lines), JVM-free, lazy +
# cached. Keyed like the other heavy caches (coarse bucket + transfer cap), bounded LRU.
# 8 entries like the tree cache: each MC result is a pair of full-grid dicts and each build
# costs seconds of serialized numba kernel — small bound, high per-entry value.
# copy_mode='shallow': dict(entry) on get, callers can't reassign the cached entry's keys.
_MC_CACHE_MAX = 8
_RAPTOR_MC_CACHE = _BoundedLRU(_MC_CACHE_MAX, copy_mode="shallow")
# (coarse_key incl. rides+speed) -> {realistic, variance, alt_bundle, alt_chips, alt_geom, dlat,
#  dlon}. The alt_* keys back /itinerary's drawn alternative routes: alt_bundle is the engine's
# dominance-window geometry handle ({alt_stop: per-cell {line: access_stop}}, the route traced from
# the SAME cached primary tree via itinerary_via_stop — no perturbed re-trace), alt_chips[ci] the
# served alt-line chip set per cell (same primary-exclusion + 4-cap /variance applies), alt_geom[ci]
# the assembled per-cell alt response. They live and die with the MC entry (same coarse key:
# workplace + rides + SPEED — alt times depend on the speed-scaled walk, so a slow-walk hover can
# never read a medium-walk bundle).
_RAPTOR_MC_INFLIGHT = {}                     # key -> threading.Event (MC build in progress)
_RAPTOR_MC_LOCK = threading.Lock()           # guards the in-flight registry; held AROUND the
# cache-miss check + owner registration so "miss => exactly one owner" stays atomic (the
# cache's own RLock nests inside; nothing acquires them in the opposite order)
# One MC build at a time, NON-blocking like /compute_exact's _HEAVY_LOCK: the parallel MC kernel
# is serialized inside core/raptor.montecarlo_commute_committed (numba's workqueue threading
# layer isn't threadsafe), so without this guard every concurrent distinct-key /variance pins a
# waitress worker thread queued on that kernel lock — eight of them wedge the whole server
# (/compute, /itinerary, /healthz included). Contention raises _Busy -> 503 + Retry-After; the
# frontend's loadVariance already retries once. Same-key concurrency is de-duped separately via
# _RAPTOR_MC_INFLIGHT (waiters block briefly on the owner's Event, then re-read the cache).
_MC_BUSY = threading.Lock()
# Lazy alt-route building mutates the cached MC entry's shared value (alt_geom) on the hover path,
# so writes go through this lock (the documented rule for mutating a cached entry's values,
# mirroring _GEOM_LOCK for the primary route geometry).
_ALT_LOCK = threading.Lock()
# Per-route TYPICAL (committed-plan p50/frag) for a PINNED cell mutates the cached MC entry's shared
# value (typ) — same locked-mutation rule as _ALT_LOCK/_GEOM_LOCK. Only the /itinerary?pin=1 path
# touches it (never a plain hover), so the compare card's numbers all follow the metric selector.
_TYP_LOCK = threading.Lock()


def _raptor_mc(dlat, dlon, max_rides=DEFAULT_MAX_RIDES, speed=DEFAULT_SPEED, walk_scalar=1.0):
    """{"realistic": {id: min}, "variance": {id: {frag, std, stuck, alt}}} for this workplace.
    realistic = MC p50 (clamped >= perfect); frag = p90-p50 bad-day delta; stuck = fraction of
    draws hitting the cap; alt = lines that become dominant under delays (EXCLUDING the cell's
    normal line). Reachability follows the perfect map (unreachable cells are omitted).

    Concurrency (mirrors /compute_exact + _itineraries_cached): the first arriver for a key
    owns the build; same-key arrivals wait on its Event then re-read the cache; a build needed
    while ANOTHER key's build holds _MC_BUSY raises _Busy (-> 503 + Retry-After) instead of
    pinning a waitress thread queued on the serialized numba kernel."""
    key = _coarse_key(dlat, dlon, max_rides, speed)
    with _RAPTOR_MC_LOCK:
        hit = _RAPTOR_MC_CACHE.get(key)          # shallow copy: callers can't reassign cache keys
        if hit is not None:
            return hit
        event = _RAPTOR_MC_INFLIGHT.get(key)
        if event is None:
            event = threading.Event()
            _RAPTOR_MC_INFLIGHT[key] = event
            owner = True
        else:
            owner = False
    if not owner:                # same key already building: wait, then re-read the cache
        event.wait()
        res = _RAPTOR_MC_CACHE.get(key)          # shallow copy (or None)
        if res is None:          # owner hit _MC_BUSY (or failed) and cached nothing ->
            raise _Busy()        # surface the same retryable 503 the owner's request got
        return res
    try:
        if not _MC_BUSY.acquire(blocking=False):
            raise _Busy()        # another workplace's MC is running -> 503, don't queue
        try:
            return _raptor_mc_build(key, dlat, dlon, max_rides, speed, walk_scalar)
        finally:
            _MC_BUSY.release()
    finally:
        with _RAPTOR_MC_LOCK:
            _RAPTOR_MC_INFLIGHT.pop(key, None)
        event.set()


def _raptor_mc_build(key, dlat, dlon, max_rides, speed, walk_scalar):
    """The actual MC build + cache write for _raptor_mc (callers go through the guard above)."""
    import numpy as _np
    import hashlib as _hl
    entry = _raptor_tree(dlat, dlon, max_rides, speed, walk_scalar)   # perfect map + dom (cached)
    cells = entry["cells"]; dom = entry["dom"] or {}
    ids = _RAPTOR.cell_ids
    perfect = _np.array([(cells[c][0] if cells[c][0] is not None else -1) for c in ids], _np.int32)
    egress_g, egress_w, purewalk = _raptor_egress_purewalk(dlat, dlon)
    # deterministic per-workplace seed -> the realistic numbers are stable across reboots/reloads
    seed = int(_hl.sha256(f"{round(dlat,5)},{round(dlon,5)},{int(max_rides)},{speed}"
                          .encode()).hexdigest()[:8], 16)
    mc = _RAPTOR.montecarlo(egress_g, egress_w, purewalk, perfect=perfect, seed=seed,
                            walk_scalar=walk_scalar, max_rounds=int(max_rides),
                            tree=entry.get("tree"))   # reuse the cached arrive-by trace (no re-trace)
    realistic, variance = {}, {}
    alt_all = mc["alt"]
    alt_chips = {}                                       # ci -> [chip lines] (drawn-route source)
    for i, c in enumerate(ids):
        if perfect[i] < 0:                              # follow the perfect map's reachability
            continue
        realistic[c] = int(mc["realistic"][i])
        v = {"frag": int(mc["frag"][i]), "std": int(mc["std"][i]),
             "stuck": round(float(mc["stuck"][i]), 2)}
        a = alt_all[i] if alt_all else None
        if a:
            domc = dom.get(c)
            # alt is the dominance window {line: min_minutes}, already sorted closest-first; drop
            # the cell's PRIMARY line (its own chip is redundant) and keep the 4 closest.
            a = {k: vv for k, vv in a.items() if k != domc}
            if a:
                a = dict(list(a.items())[:4])
                v["alt"] = a
                alt_chips[i] = list(a.keys())          # SAME chips /itinerary will draw
        variance[c] = v
    # alt_bundle (per-cell {line: access_stop}) + the matching per-cell chips back /itinerary's
    # drawn alternative routes (traced from the cached primary tree); alt_geom fills lazily on the
    # first alt hover (under _ALT_LOCK).
    out = {"realistic": realistic, "variance": variance,
           "alt_bundle": mc.get("alt_bundle"), "alt_chips": alt_chips,
           "alt_geom": {}, "typ": {}, "dlat": float(dlat), "dlon": float(dlon)}
    _RAPTOR_MC_CACHE.put(key, out)
    return out


def _mc_peek(dlat, dlon, max_rides, speed):
    """Return the cached MC entry for this workplace+rides+speed, or None — WITHOUT ever
    triggering a build. /itinerary uses this so a hover never pays the ~1s MC cost; the alt
    routes simply stay [] until the frontend's /variance fetch lands (after which the next
    hover finds the entry). A shallow dict(entry) copy, so the returned wrapper is private but
    the shared alt_geom dict is the live cache value (mutated under _ALT_LOCK)."""
    return _RAPTOR_MC_CACHE.get(_coarse_key(dlat, dlon, max_rides, speed))


def _itinerary_alts(ci, dlat, dlon, max_rides, speed, provider=None):
    """The drawn alternative routes for cell ``ci``: for each alt chip line the MC overlay surfaced
    for this cell (the dominance-window alts, after the SAME primary-exclusion + 4-cap /variance
    applies), trace that line's journey from the access stop the window picked (the bundle's per-cell
    ``alt_stop`` map) on the SAME (unperturbed) cached tree the primary route uses, via
    ``JourneyTree.itinerary_via_stop``, and assemble the geometry through the SAME
    _JourneyGeomProvider (so alt walk legs get real street paths too). Returns
    [{line, min, legs:[...geom legs...]}], or [] if the MC build hasn't run yet (we never trigger it
    here) or the cell has no alt chips.

    ``provider`` lets the caller pass the primary route's _JourneyGeomProvider so the per-cell access
    walk-tree (one Dijkstra) is shared rather than recomputed for the alts; absent, a fresh one is
    built (only reached when the cell's alt_geom isn't already cached).

    Determinism: the alts are a deterministic window over the (deterministic) tree, so the same
    request yields identical alts run-to-run AND the same set at any walk speed within the window.
    Performance: warm (alt_geom cached) returns immediately; the tree is the one /compute already
    built + cached for this workplace, so no extra RAPTOR work."""
    mc = _mc_peek(dlat, dlon, max_rides, speed)
    if mc is None:
        return []                                       # variance not built yet -> no alts
    chips = mc.get("alt_chips", {}).get(ci)
    bundle = mc.get("alt_bundle")
    if not chips or not bundle:
        return []
    geom_cache = mc["alt_geom"]
    cached = geom_cache.get(ci)
    if cached is not None:
        return cached
    # Each chip line -> the access stop the window picked for it (the bundle's per-cell alt_stop).
    alt_stop = bundle.get("alt_stop")
    cell_alt_stop = (alt_stop[ci] if alt_stop and ci < len(alt_stop) else None) or {}
    if not cell_alt_stop:
        return []
    walk_scalar = config.WALK_KMH / WALK_SPEEDS.get(speed, config.WALK_KMH)
    tree = _raptor_tree(dlat, dlon, max_rides, speed, walk_scalar)["tree"]  # cache hit (same tree)
    if provider is None:
        provider = _JourneyGeomProvider(dlat, dlon)      # per-workplace walk paths (shared trees)
    out = []
    for line in chips:                                   # preserve the chip (closest-first) order
        s = cell_alt_stop.get(line)
        if s is None:
            continue
        it = tree.itinerary_via_stop(ci, int(s), geom_provider=provider)
        if it is None:
            continue
        out.append({"line": line, "min": it["total"], "legs": it["geom"]})
    with _ALT_LOCK:                                      # cache the assembled alts for this cell
        geom_cache[ci] = out
    return out


def _itinerary_alt_typicals(ci, dlat, dlon, max_rides, speed, alts):
    """Per-ROUTE committed-plan TYPICAL (p50 minutes) + FRAGILITY (p90-p50) for a PINNED cell ``ci``,
    one (real, frag) per route: the PRIMARY (the cell's selected journey) then each alt in ``alts``
    (the /itinerary drawn alternatives, closest-first). Every route is scored by the SAME committed
    Monte-Carlo as the served ``realistic`` map (``RaptorEngine.route_typicals`` -> one shared kernel
    call), so the compare-list numbers are directly comparable and the alt's typical never reads
    faster than the primary's just because it sat on the best-case metric.

    Returns {"prim": (real, frag) | None, "alts": [(real, frag) | None, ...]} aligned to ``alts``;
    None for an unreachable route (the frontend then falls back to that route's best-case). Lazy +
    cached per pinned cell in the MC entry (``typ``) under _TYP_LOCK; ONLY reached on /itinerary?pin=1
    (never a plain hover), and only AFTER /variance has built the MC for this workplace."""
    mc = _mc_peek(dlat, dlon, max_rides, speed)
    if mc is None:
        return None                                     # variance not built yet -> no typicals
    typ_cache = mc["typ"]
    cached = typ_cache.get(ci)
    if cached is not None:
        return cached
    walk_scalar = config.WALK_KMH / WALK_SPEEDS.get(speed, config.WALK_KMH)
    entry = _raptor_tree(dlat, dlon, max_rides, speed, walk_scalar)   # cache hit (same tree)
    tree = entry["tree"]
    s_star, _aw, _lh, is_walk = tree._select(ci)
    # PRIMARY first, then each alt's access stop (from the MC alt bundle, by line name). A walk-only
    # primary has no transit access stop -> we feed a stop=-1 row (committed_legs_via_stops yields
    # kind 0 -> route_typicals returns None for it); the frontend then shows the authoritative
    # REAL[id] for the primary, so the row only keeps the alt alignment.
    bundle = mc.get("alt_bundle") or {}
    alt_stop = bundle.get("alt_stop")
    cell_alt_stop = (alt_stop[ci] if alt_stop and ci < len(alt_stop) else None) or {}
    cells = entry["cells"]
    prim_best = cells.get(_RAPTOR.cell_ids[ci], [None, None])[0]
    stops = [int(s_star) if (not is_walk and s_star >= 0) else -1]
    floors = [prim_best]
    for a in alts:
        stops.append(int(cell_alt_stop.get(a["line"], -1)))
        floors.append(a.get("min"))
    egress_g, egress_w, _purewalk = _raptor_egress_purewalk(dlat, dlon)
    import hashlib as _hl
    # SAME per-workplace seed as _raptor_mc_build's /variance MC: route_typicals draws the same-shaped
    # (nR, T) delta arrays, so the PRIMARY route's committed p50 here is byte-identical to that cell's
    # served `realistic` (REAL[id]) -> the primary compare strip matches the headline exactly.
    seed = int(_hl.sha256(f"{round(dlat,5)},{round(dlon,5)},{int(max_rides)},{speed}"
                          .encode()).hexdigest()[:8], 16)
    pairs = _RAPTOR.route_typicals(tree, ci, stops, egress_g, egress_w,
                                   perfect_route_mins=floors, seed=seed,
                                   walk_scalar=walk_scalar, max_rounds=int(max_rides))
    out = {"prim": pairs[0] if pairs else None, "alts": pairs[1:] if len(pairs) > 1 else []}
    with _TYP_LOCK:                                      # cache the typicals for this pinned cell
        typ_cache[ci] = out
    return out


# ---- Map TIME: fast reverse approximation + exact forward refine ----------------------
def compute(lat, lon, max_rides=DEFAULT_MAX_RIDES):
    """Door-to-door times from every grid cell TO (lat, lon), as {id: [best, real]}.

    FAST APPROXIMATION (reverse one-to-many): R5 is dramatically faster computing one routing
    tree from a single point to many destinations than many one-origin trees. So we route a
    single tree FROM the workplace TO every (pre-snapped) cell (~0.2s) and treat that as the
    cell's commute time. Morning (cell->work) and reverse (work->cell) aren't perfectly
    symmetric, so this has a small error (measured MAE ~2 min vs the exact method); the exact
    forward pass below refines it, and scripts/isochrone.py is the offline reference."""
    origin = gpd.GeoDataFrame({"id": ["w"]}, geometry=[Point(lon, lat)], crs=config.WGS)
    ttm = pd.DataFrame(network.travel_time_matrix(NET, origin, SNAPPED_GRID, DEP,
                                                  max_rides=max_rides))
    cells = {}
    for _, r in ttm.iterrows():
        b, rl = r["travel_time_p5"], r["travel_time_p50"]
        cells[str(r["to_id"])] = [None if pd.isna(b) else int(b),
                                  None if pd.isna(rl) else int(rl)]
    return cells


def compute_exact(lat, lon, gen, max_rides=DEFAULT_MAX_RIDES):
    """EXACT (forward) door-to-door: route every grid cell -> workplace (one tree per origin),
    the slow accurate direction. r5py's TravelTimeMatrix runs these trees serially; here we
    drive R5's TravelTimeComputer directly from the thread pool (each thread clones the Java
    RegionalTask), which is bit-exact to the serial matrix but ~4.7x faster. ``gen`` is the
    cancel token (see _map_cancelable). Returns {id: [best, real]}."""
    dest = gpd.GeoDataFrame({"id": ["d"]}, geometry=[Point(lon, lat)], crs=config.WGS)
    template = network.routing_template(NET, dest, DEP, max_rides=max_rides)

    def _one(i):
        req = copy.copy(template)        # clones the underlying Java RegionalTask
        req.origin = _EXACT_GEOMS[i]
        # travelTimes.getValues() -> [percentile_index][destination_index]; 1 destination
        vals = com.conveyal.r5.analyst.TravelTimeComputer(req, NET).computeTravelTimes().travelTimes.getValues()
        b = int(vals[0][0]); rl = int(vals[1][0])
        return _EXACT_IDS[i], [None if b == MAX_INT32 else b,
                               None if rl == MAX_INT32 else rl]

    return {cid: pair for cid, pair in _map_cancelable(_one, range(len(_EXACT_IDS)), gen)}


# Wave size for the cancelable pool map: big enough to keep the pool saturated, small enough
# that a superseded job stops within a fraction of a second.
_CANCEL_WAVE = 64


def _map_cancelable(fn, indices, gen):
    """Like _EXACT_POOL.map(fn, indices) but submitted in waves; between waves, if the global
    generation has moved past ``gen`` (a newer workplace was set), raise _Superseded so the
    stale all-cores job stops. Yields fn(i) results in order within each wave."""
    indices = list(indices)
    for start in range(0, len(indices), _CANCEL_WAVE):
        if _current_generation() != gen:
            raise _Superseded()
        wave = indices[start:start + _CANCEL_WAVE]
        for res in _EXACT_POOL.map(fn, wave, chunksize=8):
            yield res


@app.route("/compute")
@limiter.limit("60/minute")
def _compute():
    global _LAST_DEST_KEY
    lat, lon = _parse_ll(request.args.get("lat"), request.args.get("lon"))
    max_rides = _req_max_rides()
    speed, walk_scalar = _req_speed()
    # speed only changes the result when the RAPTOR engine consumes the walk scalar; the legacy
    # R5 compute() ignores it, so keep it out of the key there — otherwise toggling the
    # (ineffective) speed control would bump the generation, cancel an in-flight exact pass,
    # and reset the breakdown caches for byte-identical results.
    key = _dest_key(lat, lon, max_rides, speed if USE_RAPTOR else None)
    # Compare-and-set under _GEN_LOCK: without it, two concurrent first-time /computes for
    # different workplaces could both see _LAST_DEST_KEY = None, both bump generation, both reset
    # caches, and a third request for the displaced workplace would re-fire the reset path.
    with _GEN_LOCK:
        changed = (key != _LAST_DEST_KEY)
        if changed:
            _LAST_DEST_KEY = key
    if changed:
        # Genuinely new workplace (or a changed transfer cap — or, under RAPTOR, a changed walk
        # speed — different colored times, so treat it as new): bump the generation (cancels any
        # in-flight exact/attribution job for the PREVIOUS key between waves) and drop its
        # breakdown caches. Re-submitting the same address + params (e.g. the localStorage
        # auto-restore on refresh) keeps caches.
        gen = _bump_generation()
        _reset_caches()
    else:
        gen = _current_generation()
    t0 = dt.datetime.now()
    cells = (compute_raptor(lat, lon, max_rides, speed, walk_scalar) if USE_RAPTOR
             else compute(lat, lon, max_rides))
    ms = (dt.datetime.now() - t0).total_seconds() * 1000
    tag = "raptor" if USE_RAPTOR else "approx"
    print(f"[compute:{tag}] ({lat:.4f},{lon:.4f}) rides={max_rides} speed={speed} "
          f"{ms:.0f}ms gen={gen}")
    return jsonify({"dest": [lat, lon], "cells": cells, "ms": round(ms)})


@app.route("/compute_exact")
@limiter.limit("12/minute")
def _compute_exact():
    lat, lon = _parse_ll(request.args.get("lat"), request.args.get("lon"))
    max_rides = _req_max_rides()
    # With RAPTOR ON, /compute already returned the near-exact answer; the "exact refine" is
    # the SAME engine, so return it instantly (no heavy per-cell R5 pass). map == refine, so
    # the old fast-vs-exact contradiction disappears entirely.
    if USE_RAPTOR:
        speed, walk_scalar = _req_speed()
        cells = compute_raptor(lat, lon, max_rides, speed, walk_scalar)
        return jsonify({"dest": [lat, lon], "cells": cells, "ms": 0})
    ckey = _coarse_key(lat, lon, max_rides)
    # CACHE HIT (~110m): return the same result instantly — no R5 work, no lock, and the
    # generation/supersede dance is irrelevant since there's nothing to cancel.
    cached = _EXACT_RESULT_CACHE.get(ckey)
    if cached is not None:
        print(f"[exact] ({lat:.4f},{lon:.4f}) rides={max_rides} cached -> {len(cached)} cells")
        return jsonify({"dest": [lat, lon], "cells": cached, "ms": 0})
    t0 = dt.datetime.now()
    gen = _current_generation()           # cancel token: abort if a newer workplace is set
    # Non-blocking: if a heavy job is already running, tell the client to retry rather than
    # queueing behind a ~30s burst. Release the lock in finally so a crash can't wedge it.
    if not _HEAVY_LOCK.acquire(blocking=False):
        print(f"[exact] ({lat:.4f},{lon:.4f}) busy -> 503")
        return jsonify({"busy": True}), 503, {"Retry-After": "4"}
    try:
        cells = compute_exact(lat, lon, gen, max_rides)   # one heavy job at a time; hover stays free
    except _Superseded:
        print(f"[exact] ({lat:.4f},{lon:.4f}) superseded -> 409")
        return jsonify({"error": "superseded"}), 409
    finally:
        _HEAVY_LOCK.release()
    _EXACT_RESULT_CACHE.put(ckey, cells)
    ms = (dt.datetime.now() - t0).total_seconds() * 1000
    print(f"[exact] ({lat:.4f},{lon:.4f}) rides={max_rides} {ms:.0f}ms")
    return jsonify({"dest": [lat, lon], "cells": cells, "ms": round(ms)})


# ---- Route LEGS: recorded-path breakdowns --------------------------------------------
def _path_route_name(raw):
    """R5 records a path leg's route as "NAME (route_id)" where NAME is resolved within the
    leg's own feed (Muni '8' but BART 'Red-N (8)'). Take the part before " (", falling back to
    the raw string if the format is unexpected."""
    s = str(raw)
    i = s.rfind(" (")
    return s[:i] if i > 0 else s


def _walk_only(p50):
    """Breakdown for a cell reachable on foot only (no transit legs)."""
    return {"total": int(p50), "xfers": 0,
            "legs": [{"mode": "walk", "line": None, "min": int(p50)}]}


_TINY_HOP_MIN = 2.0   # suppress sub-2-min transit hops (fold into adjacent walk)


def _build_itin(p50, itin_map, route_name_fn):
    """Turn an R5 recorded-path result for ONE cell into the /itinerary JSON breakdown.

    ``p50`` is the cell's authoritative realistic travel time (minutes) — the SAME value
    compute_exact colors the map with — and ``itin_map`` is that cell's path-result multimap.
    We (1) pick the recorded iteration whose total best matches p50 (so the breakdown's total
    == the cell's exact map color); (2) read EXACT per-leg components from R5 (ride time from
    rideTimesSeconds — TransitLeg.inVehicleTime is unreliable — wait from the iteration's
    waitTimes, plus access/egress and transfer walk); (3) lay them out as walk -> (wait+ride)
    per leg -> walk so legs sum to round(p50); (4) FOLD any sub-2-min transit hop into the
    surrounding walk; (5) reconcile rounding residual into a walk leg.
    Returns {total, xfers, legs:[{mode, line, min, wait?}]}."""
    total = float(p50)
    walk_only = {"total": round(total), "xfers": 0,
                 "legs": [{"mode": "walk", "line": None, "min": round(total)}]}
    cands = []
    for e in list(itin_map.entrySet()):
        rseq = e.getKey()
        for it in e.getValue():
            cands.append((it.totalTime, rseq, it))
    if not cands:
        return walk_only
    # iteration whose total best matches the authoritative p50 (tie -> faster trip). Among
    # equally-good iterations, break the tie deterministically by route-name signature so the
    # chosen path (and thus "color by line") is stable across server reboots -- R5's
    # path-template iteration order is not.
    def _score(c):
        return (abs(round(c[0] / 60.0) - p50), c[0])
    best = min(cands, key=_score)
    ties = [c for c in cands if _score(c) == _score(best)]
    if len(ties) > 1:
        # Among equally-fast iterations, prefer FEWER transit legs (fewer transfers) — the
        # route a human would actually take — then route-name signature for deterministic,
        # reboot-stable tie-breaking. (Doesn't change the time/color, only which equal-time
        # path is shown. A genuinely faster multi-transfer route still wins on _score.)
        def _simplicity(c):
            legs = list(c[1].transitLegs(_TRANSIT_LAYER))
            return (len(legs), tuple(route_name_fn(l.route) for l in legs))
        best = min(ties, key=_simplicity)
    _, rseq, it = best
    ss = rseq.stopSequence
    rt = ss.rideTimesSeconds
    rides = [rt.get(i) / 60.0 for i in range(rt.size())] if rt is not None else []
    n = len(rides)
    if n == 0:                                  # reachable on foot only
        return walk_only
    legs_meta = list(rseq.transitLegs(_TRANSIT_LAYER))
    waits = [it.waitTimes.get(i) / 60.0 for i in range(it.waitTimes.size())]
    acc = (ss.access.time / 60.0) if ss.access else 0.0
    egr = (ss.egress.time / 60.0) if ss.egress else 0.0
    xfer = ss.transferTime(it) / 60.0
    # walk buckets around the n rides: walk[0]=access ... walk[n]=egress; the aggregate
    # transfer-walk time is split evenly across the n-1 inter-ride gaps.
    walk = [0.0] * (n + 1)
    walk[0] = acc
    walk[n] += egr
    if n > 1:
        per = xfer / (n - 1)
        for i in range(1, n):
            walk[i] += per
    # fold tiny rides (and their wait) into the preceding walk bucket
    kept = []
    for i in range(n):
        if rides[i] < _TINY_HOP_MIN:
            walk[i] += rides[i] + waits[i]
        else:
            kept.append(i)
    legs = []

    def push_walk(m):
        if m <= 0:
            return
        if legs and legs[-1]["mode"] == "walk":
            legs[-1]["min"] += m
        else:
            legs.append({"mode": "walk", "line": None, "min": m})

    cursor = 0
    for i in kept:
        push_walk(sum(walk[cursor:i + 1]))
        legs.append({"mode": "transit", "line": route_name_fn(legs_meta[i].route),
                     "min": rides[i], "wait": waits[i]})
        cursor = i + 1
    push_walk(sum(walk[cursor:n + 1]))
    # round each leg; then reconcile so the shown legs sum EXACTLY to round(total) —
    # core.raptor_journey.reconcile_legs is the ONE shared implementation (RAPTOR _format
    # uses the same), so the two breakdowns can't drift.
    for l in legs:
        l["min"] = round(l["min"])
        if "wait" in l:
            l["wait"] = round(l["wait"])
    return reconcile_legs(legs, round(total))


def _recorded_itin(req):
    """Run a path-recording RegionalTask ``req`` (origin already set) and return its
    /itinerary breakdown dict, or None if the workplace is unreachable from this origin.
    Shared by the full-grid prewarm and the single-cell on-demand fallback so a cold lookup
    reads identically to a cached one and its total matches the map color."""
    result = com.conveyal.r5.analyst.TravelTimeComputer(req, NET).computeTravelTimes()
    p50 = result.travelTimes.getValues()[1][0]      # realistic (matches compute_exact)
    if p50 == MAX_INT32:
        return None
    paths = result.paths
    itin_map = paths.iterationsForPathTemplates[0].asMap() if paths is not None else None
    if itin_map is None:
        return _walk_only(p50)
    return _build_itin(p50, itin_map, _path_route_name)


def prewarm_itineraries(dlat, dlon, gen, max_rides=DEFAULT_MAX_RIDES):
    """Full exact itinerary for EVERY grid cell -> workplace via R5 recorded paths. These are
    the same forward journeys behind compute_exact, so each itinerary's total equals that
    cell's exact map color. Built lazily on the first /attribution request; ``gen`` is the
    cancel token. Returns {cellId: itin dict}."""
    dest = gpd.GeoDataFrame({"id": ["d"]}, geometry=[Point(dlon, dlat)], crs=config.WGS)
    template = network.routing_template(NET, dest, DEP, paths=True, max_rides=max_rides)

    def _one(i):
        req = copy.copy(template)
        req.origin = _EXACT_GEOMS[i]
        return _EXACT_IDS[i], _recorded_itin(req)

    itins = {}
    for cid, itin in _map_cancelable(_one, range(len(_EXACT_IDS)), gen):
        if itin is not None:
            itins[cid] = itin
    return itins


def fastest_itin(olat, olon, dlat, dlon, max_rides=DEFAULT_MAX_RIDES):
    """On-demand single-OD breakdown — the fallback for /itinerary cache misses (off-grid
    points, or before any prewarm). Mirrors exactly what the prewarm caches."""
    o = NET.snap_to_network(gpd.GeoSeries([Point(olon, olat)], crs=config.WGS)).iloc[0]
    if o.is_empty:
        return None
    dest = gpd.GeoDataFrame({"id": ["d"]}, geometry=[Point(dlon, dlat)], crs=config.WGS)
    req = network.routing_template(NET, dest, DEP, paths=True,    # fresh, single-use; no clone needed
                                  max_rides=max_rides)
    req.origin = o
    return _recorded_itin(req)


@app.route("/itinerary")
@limiter.limit("120/minute")
def _itinerary():
    cid = request.args.get("id")
    if cid is not None and cid in ORIGIN_LL:
        olat, olon = ORIGIN_LL[cid]
    else:  # olat/olon are only required when no (valid) cell id resolves the origin
        olat, olon = _parse_ll(request.args.get("olat"), request.args.get("olon"))
    dlat, dlon = _parse_ll(request.args.get("dlat"), request.args.get("dlon"))
    # Same transfer cap the map used, so the breakdown total matches the cell's colored time
    # (and so it hits the SAME _dest_key-bucketed cache as the matching /compute).
    max_rides = _req_max_rides()
    speed, walk_scalar = _req_speed()
    # RAPTOR path (Phase 2): trace the cell's journey from the cached tree -> breakdown total
    # EQUALS the cell's map value by construction (no R5), for BOTH semantics — arrive-by from the
    # single-deadline JourneyTree, depart-after from the DepartAfterJourneyTree's per-T* tracer.
    # Off-grid points snap to nearest cell. (The R5 fallback below now serves only USE_RAPTOR=0.)
    if USE_RAPTOR and RAPTOR_SEMANTIC in ("arriveby", "departafter"):
        ci = _RAPTOR.cell_index.get(cid) if cid is not None else None
        if ci is None:
            ci = _nearest_raptor_cell(olat, olon)
        res = None
        provider = None
        if ci is not None:
            entry = _raptor_tree(dlat, dlon, max_rides, speed, walk_scalar)
            res = entry["geom"].get(ci)          # warm: assembled geometry cached per cell
            if res is None:
                # Geometry from the SAME traced journey the breakdown shows (the hover==map
                # invariant extends to the drawn route — never recomputed via another path).
                provider = _JourneyGeomProvider(dlat, dlon)
                res = entry["tree"].itinerary(ci, geom_provider=provider)
                if res is not None:
                    with _GEOM_LOCK:             # mutating a cached entry's value -> locked
                        entry["geom"][ci] = res
        res = dict(res) if res else {"error": "no route"}
        # Drawn alternative routes (under-delays re-routes): for the SAME chip lines the MC
        # overlay surfaced for this cell, the alt journey traced from a captured perturbed draw.
        # Empty until /variance has built the MC for this workplace — we never trigger that build
        # here, so the hover stays cheap; the frontend re-hovers after /variance lands. Reuse the
        # provider the primary geom just built: its per-cell access walk-tree (one Dijkstra) is
        # memoized, so the alt routes for the same cell don't redo it (warm walk legs).
        res["alts"] = (_itinerary_alts(ci, dlat, dlon, max_rides, speed, provider=provider)
                       if ci is not None and "error" not in res else [])
        # PINNED cells (?pin=1) also get a per-ROUTE committed-plan typical + fragility so the
        # compare card can show every strip on the SAME metric as the selector (the primary's typical
        # was already best-case-vs-typical; the alts now carry their own). Lazy + cached per pinned
        # cell, scored by the same committed MC; NEVER computed on a plain hover (the gate below).
        if (request.args.get("pin") == "1" and ci is not None and "error" not in res
                and res.get("alts") is not None):
            typ = _itinerary_alt_typicals(ci, dlat, dlon, max_rides, speed, res["alts"])
            if typ is not None:
                prim = typ.get("prim")
                if prim is not None:
                    res["real"], res["frag"] = prim[0], prim[1]
                for a, p in zip(res["alts"], typ.get("alts", [])):
                    if p is not None:
                        a["real"], a["frag"] = p[0], p[1]
        res["olat"], res["olon"] = round(olat, 5), round(olon, 5)
        return jsonify(res)
    dkey = _dest_key(dlat, dlon, max_rides)
    res = None
    # FAST PATH 1: the full-grid cache (present once color-by-line has been triggered).
    if cid is not None:
        with _ITIN_CACHE_LOCK:
            full = _ITIN_CACHE.get(dkey)
        if full is not None and cid in full:
            res = full[cid]
    # FAST PATH 2: the per-cell on-demand cache (this cell was hovered before).
    if res is None and cid is not None:
        with _CELL_CACHE_LOCK:
            res = (_CELL_CACHE.get(dkey) or {}).get(cid)
    # FALLBACK: first touch of this cell (or an off-grid point) -> compute ONE cell on demand
    # (~100ms) and cache it. The common flow thus computes only the cells actually hovered.
    if res is None:
        gen = _current_generation()   # don't repopulate a cache _reset_caches cleared mid-call
        res = fastest_itin(olat, olon, dlat, dlon, max_rides) or {"error": "no route"}
        if cid is not None and "error" not in res and _current_generation() == gen:
            with _CELL_CACHE_LOCK:
                _CELL_CACHE.setdefault(dkey, {})[cid] = res
    res = dict(res)                       # don't mutate the cached object with olat/olon
    res.setdefault("geom", None)          # legacy R5 path: no traced geometry — frontend
    #                                       falls back gracefully (text breakdown only)
    res.setdefault("alts", [])            # alt routes are a RAPTOR-only feature; legacy path = none
    res["olat"], res["olon"] = round(olat, 5), round(olon, 5)
    return jsonify(res)


@app.route("/attribution")
@limiter.limit("12/minute")
def _attribution():
    """The "color by line" map: dominant transit line per cell. The ONLY trigger for the
    full-grid itinerary build (lazy here, on the user's first toggle). Cached + in-flight
    de-duped, so /itinerary also benefits once it has run."""
    dlat, dlon = _parse_ll(request.args.get("dlat"), request.args.get("dlon"))
    max_rides = _req_max_rides()
    speed, walk_scalar = _req_speed()
    # RAPTOR path (Phase 2): dominant line per cell from the same cached tree as the map (no R5,
    # no _HEAVY_LOCK / fan spike, deterministic), for BOTH semantics — depart-after reads the
    # DepartAfterJourneyTree.dominant() populated into the entry by _raptor_tree. (The R5
    # color-by-line below now serves only USE_RAPTOR=0.)
    if USE_RAPTOR and RAPTOR_SEMANTIC in ("arriveby", "departafter"):
        attr = _raptor_attribution(dlat, dlon, max_rides, speed, walk_scalar)
        print(f"[attr:raptor] ({dlat:.4f},{dlon:.4f}) rides={max_rides} speed={speed} "
              f"-> {len(attr)} cells")
        return jsonify(attr)
    ckey = _coarse_key(dlat, dlon, max_rides)
    # CACHE HIT (~110m): return the same attribution dict instantly — no R5 work, no lock.
    cached_res = _ATTR_RESULT_CACHE.get(ckey)
    if cached_res is not None:
        print(f"[attr] ({dlat:.4f},{dlon:.4f}) rides={max_rides} cached -> {len(cached_res)} cells")
        return jsonify(cached_res)
    t0 = dt.datetime.now()
    with _ITIN_CACHE_LOCK:
        cached = _dest_key(dlat, dlon, max_rides) in _ITIN_CACHE
    # Non-blocking: a build needed while another heavy job runs -> 503 (don't queue ~48s).
    try:
        attr = _attribution_from_cache(dlat, dlon, nonblock=True, max_rides=max_rides)
    except _Busy:
        print(f"[attr] ({dlat:.4f},{dlon:.4f}) busy -> 503")
        return jsonify({"busy": True}), 503, {"Retry-After": "4"}
    # Only cache a real result. A superseded build (workplace changed mid-build) returns {};
    # caching that would poison this ~110m bucket and make color-by-line return {} forever.
    if attr:
        _ATTR_RESULT_CACHE.put(ckey, attr)
    ms = (dt.datetime.now() - t0).total_seconds() * 1000
    print(f"[attr] ({dlat:.4f},{dlon:.4f}) {ms:.0f}ms -> {len(attr)} cells"
          f"{' (cached)' if cached else ''}")
    return jsonify(attr)


@app.route("/variance")
@limiter.limit("20/minute")
def _variance():
    """Service-noise overlay (RAPTOR arrive-by only): realistic (MC p50) + per-cell fragility +
    alternative lines. Lazy + cached; fetched by the frontend AFTER /compute paints the perfect
    map (progressive refinement, like /compute_exact). Empty when the flag/semantic is off so the
    frontend simply keeps the perfect map."""
    if not (USE_RAPTOR and RAPTOR_SEMANTIC == "arriveby" and RAPTOR_MC):
        return jsonify({"realistic": {}, "variance": {}})
    dlat, dlon = _parse_ll(request.args.get("dlat"), request.args.get("dlon"))
    max_rides = _req_max_rides()
    speed, walk_scalar = _req_speed()
    t0 = dt.datetime.now()
    # Non-blocking like /compute_exact: another workplace's MC running -> retryable 503
    # (the frontend's loadVariance retries once and otherwise keeps the perfect map).
    try:
        out = _raptor_mc(dlat, dlon, max_rides, speed, walk_scalar)
    except _Busy:
        print(f"[variance:raptor] ({dlat:.4f},{dlon:.4f}) busy -> 503")
        return jsonify({"busy": True}), 503, {"Retry-After": "4"}
    ms = (dt.datetime.now() - t0).total_seconds() * 1000
    print(f"[variance:raptor] ({dlat:.4f},{dlon:.4f}) rides={max_rides} speed={speed} {ms:.0f}ms "
          f"-> {len(out['realistic'])} cells")
    # Pick the JSON-able keys explicitly: the MC entry also carries the internal alt-route
    # plumbing (alt_bundle's numpy arrays, the lazy JourneyTree cache) that backs /itinerary's
    # drawn alternatives — never serialize those.
    return jsonify({"dest": [dlat, dlon], "realistic": out["realistic"],
                    "variance": out["variance"], "ms": round(ms)})


@app.route("/geocode")
@limiter.limit("60/minute")
def _geocode():
    q = request.args.get("q", "")
    if not q.strip():
        return jsonify({"error": "empty"}), 400
    try:
        lat, lon, label = geo.geocode(q, cache=False)   # don't pollute the dest cache
    except LookupError:
        return jsonify({"error": "not found"}), 404
    except (OSError, ValueError, KeyError):
        return jsonify({"error": "geocoding failed"}), 502
    return jsonify({"lat": lat, "lon": lon, "label": label})


@app.route("/autocomplete")
@limiter.limit("120/minute")
def _autocomplete():
    """Type-ahead suggestions for the workplace box. {"results":[{"label","lat","lon"}, ...]}
    with at most 6 entries. Blank/too-short q -> {"results":[]} (don't hit the geocoder for a
    single character). Upstream failures degrade to an empty list rather than an error so the
    box stays usable."""
    q = request.args.get("q", "")
    if len(q.strip()) < 2:
        return jsonify({"results": []})
    try:
        results = geo.autocomplete(q, limit=6)
    except (OSError, ValueError, KeyError):
        return jsonify({"results": []})
    return jsonify({"results": results})


# ---- Page (built once at boot from the template + shared viz.js) -----------------------
def _build_page():
    html = (HERE / "templates" / "index.html").read_text()
    viz = (HERE / "assets" / "viz.js").read_text()
    # Tell the page which semantic it's serving so it can drop the (now meaningless) R5 "refine"
    # affordance and frame everything as an arrive-by-09:00 estimate. arrive-by => map == the
    # engine result, there is no separate exact pass to refine to.
    _arriveby = USE_RAPTOR and RAPTOR_SEMANTIC == "arriveby"
    # speedtoggle: the walk-speed control is only fully wired under RAPTOR arrive-by (map AND
    # breakdown both apply the scalar). Under departafter the map would shift but the R5
    # breakdown routes at fixed 4.8 km/h — inconsistent numbers — so the frontend hides it.
    cfg = {"raptor": USE_RAPTOR, "arriveby": _arriveby, "speedtoggle": _arriveby,
           "timephrase": ("arriving by ~9:00am" if _arriveby
                          else f"leaving ~{DEP:%-I:%M%p}".lower())}
    # No default workplace is EVER injected into the page (privacy invariant): the .env
    # DEFAULT_ADDRESS / DEST_LAT/LON would otherwise put a real personal address into the
    # served HTML. A first-time visitor sees the onboarding prompt and types their own; a
    # returning visitor's own previously-typed workplace is restored from localStorage by
    # the frontend. DEFAULT_ADDRESS in .env/sfci.env no longer affects the public page.
    cfg["default_wp"] = None

    def _js(o):
        # json.dumps does NOT escape "<", so a "</script>" (or "<!--") inside third-party data
        # (geocoder label, GTFS route names, DataSF neighborhood names) would terminate the
        # inline <script> and inject markup. "<" is a valid JSON escape, so the parsed
        # values are byte-identical — only the embedding is hardened.
        return json.dumps(o).replace("<", "\\u003c")

    # __CFG__ is replaced LAST: it carries the only live string (the geocoder label), so if
    # that ever contained a literal "__CELLS__"/"__LINES__" token it stays literal instead of
    # expanding into the multi-MB JSON blobs.
    return (html.replace("/*__VIZ__*/", viz)
                .replace("__CELLS__", _js(CELLS_GEOJSON))
                .replace("__LINES__", _js(LINES))
                .replace("__CFG__", _js(cfg)))


PAGE_HTML = _build_page()
_BOOT_TS = time.time()


@app.route("/")
def _index():
    return PAGE_HTML


@app.route("/healthz")
def _healthz():
    """Cheap, non-rate-limited liveness/readiness probe. Returns the engine + walk router state
    + the modeled service date, so a monitor can alarm on (a) the engine never initializing or
    (b) the GTFS feed having drifted past its validity window (svc_date won't update until the
    server restarts after a feed re-pull)."""
    return jsonify({
        "ok": _RAPTOR is not None or USE_RAPTOR is False,
        "engine": "raptor" if USE_RAPTOR else "r5",
        "semantic": RAPTOR_SEMANTIC,
        "walk": "graph" if USE_WALK_GRAPH else "r5",
        "svc_date": _SVC_DATE.isoformat() if _SVC_DATE else None,
        "uptime_s": int(time.time() - _BOOT_TS),
    })


if __name__ == "__main__":
    # In production, prefer waitress (waitress-serve) over Flask's dev server: clean SIGTERM,
    # no "WARNING: development server" noise, no surprise per-request thread spawn. Falls back
    # to Flask's threaded dev server if waitress isn't installed (so local dev still works).
    PORT = int(os.environ.get("PORT", "8000"))
    try:
        from waitress import serve
        print(f"[boot] waitress serving on 127.0.0.1:{PORT}")
        serve(app, host="127.0.0.1", port=PORT, threads=8, _quiet=False, channel_timeout=120)
    except ImportError:
        print(f"[boot] waitress not installed — falling back to Flask dev server on 127.0.0.1:{PORT}")
        app.run(host="127.0.0.1", port=PORT, threaded=True)
