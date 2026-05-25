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
import os, sys, json, copy, threading, datetime as dt
from collections import OrderedDict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, geopandas as gpd, shapely
from shapely.geometry import Point
from flask import Flask, request, jsonify
from flask_limiter import Limiter

# Cap the JVM heap BEFORE r5py starts the JVM (imported via core.network below). r5py
# defaults -Xmx to 80% of TOTAL system RAM — fine locally, but on a small hosting box that
# over-reserves and risks OOM. Set R5_MAX_MEMORY (e.g. "1200M" or "70%") to cap it; unset
# keeps r5py's default. r5py reads this from its --max-memory option.
_r5_mem = os.environ.get("R5_MAX_MEMORY")
if _r5_mem and "--max-memory" not in sys.argv and "-m" not in sys.argv:
    sys.argv += ["--max-memory", _r5_mem]

from core import config, feeds, grid, network, geo
from core.network import MAX_INT32, DEFAULT_MAX_RIDES
import com.conveyal.r5            # JVM already started by core.network's r5py import

config.load_dotenv()             # load .env (GEOCODER / GEOAPIFY_KEY) for the geocoder

HERE = Path(__file__).resolve().parent
GRID_M = int(os.environ.get("GRID_M", str(config.GRID_M)))

# ---- Boot: feeds, model date, grid, warm R5 network (all once) -------------------------
GTFS = config.gtfs_paths()
LINES = feeds.route_shapes(GTFS)                  # GTFS line geometries for the overlay
# Auto-pick a weekday with trips in ALL feeds, so a future data repull (whose feed window
# may not cover a hardcoded date) keeps working instead of silently returning no service.
_SVC_DATE = feeds.pick_service_date(GTFS)
DEP = config.departure(_SVC_DATE)
print(f"[boot] modeling weekday {_SVC_DATE} @ {DEP:%-I:%M%p}".lower())

print(f"[boot] building grid @ {GRID_M}m + loading R5 network (once)...")
NEIGH = grid.load_neighborhoods()
GRID = grid.build_grid(NEIGH, GRID_M)[["id", "geometry"]]
ORIGIN_LL = {r.id: (r.geometry.y, r.geometry.x) for r in GRID.itertuples()}

# cell squares (sent to the browser once) + neighborhood label per cell
_cells = gpd.GeoDataFrame({"id": GRID["id"].values},
                          geometry=grid.square_cells(GRID, GRID_M).values, crs=config.WGS)
_cells = grid.attach_neighborhoods(_cells, NEIGH)
CELLS_GEOJSON = json.loads(_cells.to_json())
for f in CELLS_GEOJSON["features"]:
    f["properties"] = {"id": f["properties"]["id"], "n": f["properties"].get("name")}

NET = network.build_network(GTFS)

# Pre-snap the grid to the street network ONCE so every /compute can route a single tree
# FROM the workplace TO every grid cell (the fast reverse approximation) without re-snapping
# thousands of points per request.
SNAPPED_GRID = GRID.copy()
SNAPPED_GRID["geometry"] = NET.snap_to_network(GRID.geometry)
SNAPPED_GRID = SNAPPED_GRID[SNAPPED_GRID.geometry != shapely.Point()].reset_index(drop=True)
print(f"[boot] ready: {len(GRID)} origins ({len(SNAPPED_GRID)} on-network). "
      f"Open http://127.0.0.1:8000")

# Pre-extract origin ids/geometries once (used by the threaded exact router below).
_EXACT_IDS = list(SNAPPED_GRID.id)
_EXACT_GEOMS = list(SNAPPED_GRID.geometry)
# The TransitLayer materialises a recorded path's transit legs (route names, stops). It's
# read-only and shared across threads like the rest of NET.
_TRANSIT_LAYER = NET._transport_network.transitLayer

# Thread pool for the EXACT recompute. R5 routing is read-only against the shared warm
# network and each task clones its own Java RegionalTask, so per-origin routing parallelises
# safely in this one JVM. The Java RAPTOR releases the GIL, but Python-side result extraction
# does not, so the real speedup tops out ~4.8x near the physical-core count. We default the
# pool to (cores - 2) to leave the machine thermal/interactive headroom (the fan). Override
# with EXACT_THREADS=N.
_N_PHYS = os.cpu_count() or 8
EXACT_THREADS = int(os.environ.get("EXACT_THREADS", str(max(2, _N_PHYS - 2))))
_EXACT_POOL = ThreadPoolExecutor(max_workers=EXACT_THREADS, thread_name_prefix="r5-exact")

# ---- RAPTOR engine (flag-gated grid travel-times) -------------------------------------
# USE_RAPTOR=1 swaps the grid coloring from the R5 reverse-approx + per-cell exact refine to
# the self-built reverse range-RAPTOR (near-exact vs R5: MAE ~0.75, max ~7; ~0.05-0.14s/req,
# no heavy per-visitor R5 pass). R5 stays in-process ONLY for the on-demand hover breakdown
# (/itinerary) + color-by-line (/attribution) — Phase 2 moves those to RAPTOR too. Default
# OFF: the live app is byte-identical until the flag is flipped. RAPTOR_SEMANTIC selects
# 'departafter' (matches today's depart-8:35 model + the R5 hover, so map~=hover within RAPTOR
# error) or 'arriveby' (the arrive-by-09:00 product semantic, surfaced for review).
USE_RAPTOR = os.environ.get("USE_RAPTOR", "").lower() in ("1", "true", "yes", "on")
RAPTOR_SEMANTIC = os.environ.get("RAPTOR_SEMANTIC", "arriveby").lower()
# RAPTOR_MC=1 (default ON under arrive-by): the service-noise Monte-Carlo "realistic + fragility
# + alt-lines" layer, served by /variance (lazy, cached, ~1s once per workplace, off the hover
# path). The perfect-timing map (/compute) is unchanged; realistic is a refinement overlay.
RAPTOR_MC = os.environ.get("RAPTOR_MC", "1").lower() in ("1", "true", "yes", "on")
_RAPTOR = None
_RAPTOR_STOPS = None             # GeoDataFrame of stop coords keyed by gid (egress destinations)
_RAPTOR_EGRESS_CACHE = OrderedDict()   # coarse_key -> (egress_g, egress_w, purewalk)
_RAPTOR_EGRESS_LOCK = threading.Lock()
if USE_RAPTOR:
    try:
        from core import raptor_engine
        _RAPTOR = raptor_engine.RaptorEngine(GTFS, _SVC_DATE, verbose=True)
        import numpy as _np
        _gl, _go = _RAPTOR.data["stop_lat"], _RAPTOR.data["stop_lon"]
        _gids = [g for g in range(_RAPTOR.data["n_stops"]) if not _np.isnan(_gl[g])]
        # id="S<gid>" so stop ids stay DISJOINT from cell ids ("0".."N") and the origin "w"
        # (R5's matrix returns 0 travel time when an origin id string == a destination id).
        _RAPTOR_STOPS = gpd.GeoDataFrame(
            {"id": ["S" + str(g) for g in _gids]},
            geometry=[Point(_go[g], _gl[g]) for g in _gids], crs=config.WGS)
        # align purewalk to the engine's cell order (server grid == engine grid by id)
        _RAPTOR_CELL_POS = {c: i for i, c in enumerate(_RAPTOR.cell_ids)}
        print(f"[boot] RAPTOR engine ON (semantic={RAPTOR_SEMANTIC}); R5 kept for hover only")
    except Exception as _e:                     # missing access table etc. -> degrade to OFF
        print(f"[boot] USE_RAPTOR requested but engine init failed ({_e}); using R5 path")
        USE_RAPTOR = False

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


def _dest_key(lat, lon, max_rides=DEFAULT_MAX_RIDES):
    """Per-workplace cache key for the breakdown caches (_ITIN_CACHE/_CELL_CACHE/
    _ITIN_INFLIGHT) and the _LAST_DEST_KEY change-detector. A capped result is a DIFFERENT
    journey than the uncapped one, so the cap is part of the key — otherwise a maxrides=1
    breakdown would poison the uncapped cell (and vice-versa). At the R5 default the key is
    the same 2-tuple as before, so the baseline (and existing key callers) is unchanged."""
    base = (round(float(lat), 5), round(float(lon), 5))
    return base if max_rides == DEFAULT_MAX_RIDES else base + (int(max_rides),)


def _reset_caches():
    """Drop cached breakdowns for previous workplaces (called on each new /compute). An
    in-flight build for an old workplace is already a different generation, so it returns {}
    and pops its own in-flight marker; we only clear the result dicts here."""
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
    the work). A cache hit never blocks regardless. ``max_rides`` is part of the cache key so
    a capped build can't be served for an uncapped request."""
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
        return _ITIN_CACHE.get(key, {})
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
        if itins:                    # never cache an empty result (a superseded/partial build
            with _ITIN_CACHE_LOCK:   # would otherwise poison this workplace until the next /compute)
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
# /itinerary. The cached values are deep-copied on read so callers can't mutate the store.
_RESULT_CACHE_MAX = 150
_EXACT_RESULT_CACHE = OrderedDict()       # coarse_key -> {id: [best, real]} (exact cells)
_ATTR_RESULT_CACHE = OrderedDict()        # coarse_key -> {cellId: dominantLine}
_RESULT_CACHE_LOCK = threading.Lock()


def _coarse_key(lat, lon, max_rides=DEFAULT_MAX_RIDES):
    """~110m bucket for the heavy-result cache (_EXACT_RESULT_CACHE/_ATTR_RESULT_CACHE).
    Like _dest_key, the transfer cap is part of the key so a capped result can't be served
    for an uncapped request. At the R5 default the key is the same 2-tuple as before, so the
    baseline cache behavior (and existing callers) is unchanged."""
    base = (round(float(lat), 3), round(float(lon), 3))
    return base if max_rides == DEFAULT_MAX_RIDES else base + (int(max_rides),)


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


def _result_cache_get(store, key):
    with _RESULT_CACHE_LOCK:
        if key in store:
            store.move_to_end(key)
            return copy.deepcopy(store[key])   # don't hand out the cached object
    return None


def _result_cache_put(store, key, value):
    with _RESULT_CACHE_LOCK:
        store[key] = copy.deepcopy(value)      # store our own copy
        store.move_to_end(key)
        while len(store) > _RESULT_CACHE_MAX:
            store.popitem(last=False)


app = Flask(__name__)

# ---- Rate limiting (Flask-Limiter, in-memory) -----------------------------------------
# Single-process app, so the default in-memory store is correct (no Redis needed). Per-IP
# limits are applied per endpoint below via @limiter.limit. The key func prefers the FIRST
# hop in X-Forwarded-For (the real client when behind a proxy) and falls back to the socket
# peer when the header is absent/blank — so it's safe both behind a proxy and run directly.
def _client_ip():
    fwd = request.headers.get("X-Forwarded-For", "")
    if fwd:
        first = fwd.split(",")[0].strip()
        if first:
            return first
    return request.remote_addr or "127.0.0.1"


limiter = Limiter(key_func=_client_ip, app=app, storage_uri="memory://")


# ---- RAPTOR grid travel-times (flag-gated) --------------------------------------------
def _raptor_egress_purewalk(lat, lon):
    """Per-workplace inputs for the RAPTOR engine, via ONE-origin R5 WALK matrices:
      egress_g/egress_w — W->stop walk seconds (gid-keyed), capped at the access cap;
      purewalk          — W->cell walk seconds (cell order), capped at MAX_MIN.
    The only R5 use on the RAPTOR map path (one light walk tree, not the heavy per-cell pass).
    Cached per ~110m workplace bucket."""
    import numpy as _np
    ckey = _coarse_key(lat, lon)
    cached = None
    with _RAPTOR_EGRESS_LOCK:
        if ckey in _RAPTOR_EGRESS_CACHE:
            _RAPTOR_EGRESS_CACHE.move_to_end(ckey)
            cached = _RAPTOR_EGRESS_CACHE[ckey]
    if cached is not None:
        return cached
    W = gpd.GeoDataFrame({"id": ["w"]}, geometry=[Point(lon, lat)], crs=config.WGS)
    cap = _RAPTOR.access_cap_min
    # egress: W -> stops (walk), gid-keyed
    e = pd.DataFrame(network.walk_time_matrix(NET, W, _RAPTOR_STOPS, DEP, cap))
    ec = "travel_time" if "travel_time" in e.columns else \
        [c for c in e.columns if c.startswith("travel_time")][0]
    egr = {}
    for to, v in zip(e["to_id"].astype(str), e[ec]):
        if pd.isna(v):
            continue
        g = int(to[1:])                       # strip the "S" prefix -> gid
        sec = int(round(float(v) * 60))
        if g not in egr or sec < egr[g]:
            egr[g] = sec
    egress_g = _np.array(sorted(egr), dtype=_np.int32)
    egress_w = _np.array([egr[g] for g in egress_g], dtype=_np.int64)
    # pure walk: W -> cells, aligned to the engine's cell order
    pw_ttm = pd.DataFrame(network.walk_time_matrix(NET, W, SNAPPED_GRID, DEP, config.MAX_MIN))
    pc = "travel_time" if "travel_time" in pw_ttm.columns else \
        [c for c in pw_ttm.columns if c.startswith("travel_time")][0]
    purewalk = _np.full(len(_RAPTOR.cell_ids), -1, dtype=_np.int64)
    for to, v in zip(pw_ttm["to_id"].astype(str), pw_ttm[pc]):
        if pd.isna(v):
            continue
        i = _RAPTOR_CELL_POS.get(to)
        if i is not None:
            purewalk[i] = int(round(float(v) * 60))
    res = (egress_g, egress_w, purewalk)
    with _RAPTOR_EGRESS_LOCK:
        _RAPTOR_EGRESS_CACHE[ckey] = res
        while len(_RAPTOR_EGRESS_CACHE) > 64:
            _RAPTOR_EGRESS_CACHE.popitem(last=False)
    return res


# Per-workplace traced tree (Phase 2): one arrive-by tree serves the MAP (actual commute),
# the hover breakdown, AND color-by-line — so they are guaranteed consistent (hover == map).
_RAPTOR_TREE_CACHE = OrderedDict()           # (coarse_key, max_rides) -> {tree, cells, dom}
_RAPTOR_TREE_LOCK = threading.Lock()


def _raptor_tree(lat, lon, max_rides=DEFAULT_MAX_RIDES):
    """Build (or fetch) the cached arrive-by JourneyTree + per-cell map values (actual commute)
    + dominant lines for this workplace (the Phase-2 path layer). Tracing every cell is ~0.14s,
    done once per workplace bucket."""
    key = _coarse_key(lat, lon, max_rides)
    with _RAPTOR_TREE_LOCK:
        if key in _RAPTOR_TREE_CACHE:
            _RAPTOR_TREE_CACHE.move_to_end(key)
            return _RAPTOR_TREE_CACHE[key]
    egress_g, egress_w, purewalk = _raptor_egress_purewalk(lat, lon)
    tree = _RAPTOR.journey_tree(egress_g, egress_w, purewalk, max_rounds=int(max_rides))
    commute, dom = tree.commute_and_dominant()
    cells = {c: ([int(commute[i]), int(commute[i])] if commute[i] >= 0 else [None, None])
             for i, c in enumerate(_RAPTOR.cell_ids)}
    domd = {c: dom[i] for i, c in enumerate(_RAPTOR.cell_ids) if dom[i] is not None}
    entry = {"tree": tree, "cells": cells, "dom": domd}
    with _RAPTOR_TREE_LOCK:
        _RAPTOR_TREE_CACHE[key] = entry
        while len(_RAPTOR_TREE_CACHE) > 16:
            _RAPTOR_TREE_CACHE.popitem(last=False)
    return entry


def compute_raptor(lat, lon, max_rides=DEFAULT_MAX_RIDES):
    """Grid travel-times via the RAPTOR engine -> {id: [best, real]} (same shape as compute).
    arrive-by uses the traced tree (actual commute, pairs with the RAPTOR breakdown so hover==map);
    depart-after uses the R5-VALIDATED vectorized p5/p50 (Phase 1; pairs with the R5 breakdown)."""
    if RAPTOR_SEMANTIC == "arriveby":
        return _raptor_tree(lat, lon, max_rides)["cells"]
    egress_g, egress_w, purewalk = _raptor_egress_purewalk(lat, lon)
    res = _RAPTOR.departafter(egress_g, egress_w, purewalk, max_rounds=int(max_rides))
    return {cid: [p[0], p[1]] for cid, p in res.items()}


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


def _raptor_attribution(dlat, dlon, max_rides=DEFAULT_MAX_RIDES):
    """{cellId: dominant line} from the cached arrive-by tree (color-by-line, no R5)."""
    entry = _raptor_tree(dlat, dlon, max_rides)
    if entry["dom"] is None:                  # depart-after map -> derive dominant on demand
        _, dom = entry["tree"].commute_and_dominant()
        entry["dom"] = {c: dom[i] for i, c in enumerate(_RAPTOR.cell_ids) if dom[i] is not None}
    return entry["dom"]


# Per-workplace service-noise Monte-Carlo (realistic + fragility + alt-lines), JVM-free, lazy +
# cached. Keyed like the other heavy caches (coarse bucket + transfer cap), bounded LRU.
_RAPTOR_MC_CACHE = OrderedDict()             # (coarse_key, max_rides) -> {realistic, variance}
_RAPTOR_MC_LOCK = threading.Lock()


def _raptor_mc(dlat, dlon, max_rides=DEFAULT_MAX_RIDES):
    """{"realistic": {id: min}, "variance": {id: {frag, std, stuck, alt}}} for this workplace.
    realistic = MC p50 (clamped >= perfect); frag = p90-p50 bad-day delta; stuck = fraction of
    draws hitting the cap; alt = lines that become dominant under delays (EXCLUDING the cell's
    normal line). Reachability follows the perfect map (unreachable cells are omitted)."""
    import numpy as _np
    import hashlib as _hl
    key = _coarse_key(dlat, dlon, max_rides)
    with _RAPTOR_MC_LOCK:
        if key in _RAPTOR_MC_CACHE:
            _RAPTOR_MC_CACHE.move_to_end(key)
            return _RAPTOR_MC_CACHE[key]
    entry = _raptor_tree(dlat, dlon, max_rides)         # perfect map + dominant lines (cached)
    cells = entry["cells"]; dom = entry["dom"] or {}
    ids = _RAPTOR.cell_ids
    perfect = _np.array([(cells[c][0] if cells[c][0] is not None else -1) for c in ids], _np.int32)
    egress_g, egress_w, purewalk = _raptor_egress_purewalk(dlat, dlon)
    # deterministic per-workplace seed -> the realistic numbers are stable across reboots/reloads
    seed = int(_hl.sha256(f"{round(dlat,5)},{round(dlon,5)},{int(max_rides)}"
                          .encode()).hexdigest()[:8], 16)
    mc = _RAPTOR.montecarlo(egress_g, egress_w, purewalk, perfect=perfect, seed=seed)
    realistic, variance = {}, {}
    alt_all = mc["alt"]
    for i, c in enumerate(ids):
        if perfect[i] < 0:                              # follow the perfect map's reachability
            continue
        realistic[c] = int(mc["realistic"][i])
        v = {"frag": int(mc["frag"][i]), "std": int(mc["std"][i]),
             "stuck": round(float(mc["stuck"][i]), 2)}
        a = alt_all[i] if alt_all else None
        if a:
            domc = dom.get(c)
            a = {k: vv for k, vv in a.items() if k != domc}   # only OTHER lines (not the normal one)
            if a:
                v["alt"] = dict(list(a.items())[:4])
        variance[c] = v
    out = {"realistic": realistic, "variance": variance}
    with _RAPTOR_MC_LOCK:
        _RAPTOR_MC_CACHE[key] = out
        while len(_RAPTOR_MC_CACHE) > 16:
            _RAPTOR_MC_CACHE.popitem(last=False)
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
    lat = float(request.args["lat"]); lon = float(request.args["lon"])
    max_rides = _req_max_rides()
    key = _dest_key(lat, lon, max_rides)
    if key != _LAST_DEST_KEY:
        # Genuinely new workplace (or a changed transfer cap — different colored times, so
        # treat it as new): bump the generation (cancels any in-flight exact/attribution job
        # for the PREVIOUS key between waves) and drop its breakdown caches. Re-submitting the
        # same address + cap (e.g. the localStorage auto-restore on refresh) keeps the caches.
        _LAST_DEST_KEY = key
        gen = _bump_generation()
        _reset_caches()
    else:
        gen = _current_generation()
    t0 = dt.datetime.now()
    cells = compute_raptor(lat, lon, max_rides) if USE_RAPTOR else compute(lat, lon, max_rides)
    ms = (dt.datetime.now() - t0).total_seconds() * 1000
    tag = "raptor" if USE_RAPTOR else "approx"
    print(f"[compute:{tag}] ({lat:.4f},{lon:.4f}) rides={max_rides} {ms:.0f}ms gen={gen}")
    return jsonify({"dest": [lat, lon], "cells": cells, "ms": round(ms)})


@app.route("/compute_exact")
@limiter.limit("12/minute")
def _compute_exact():
    lat = float(request.args["lat"]); lon = float(request.args["lon"])
    max_rides = _req_max_rides()
    # With RAPTOR ON, /compute already returned the near-exact answer; the "exact refine" is
    # the SAME engine, so return it instantly (no heavy per-cell R5 pass). map == refine, so
    # the old fast-vs-exact contradiction disappears entirely.
    if USE_RAPTOR:
        cells = compute_raptor(lat, lon, max_rides)
        return jsonify({"dest": [lat, lon], "cells": cells, "ms": 0})
    ckey = _coarse_key(lat, lon, max_rides)
    # CACHE HIT (~110m): return the same result instantly — no R5 work, no lock, and the
    # generation/supersede dance is irrelevant since there's nothing to cancel.
    cached = _result_cache_get(_EXACT_RESULT_CACHE, ckey)
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
    _result_cache_put(_EXACT_RESULT_CACHE, ckey, cells)
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
    # round each leg; then reconcile so the shown legs sum EXACTLY to round(total)
    for l in legs:
        l["min"] = round(l["min"])
        if "wait" in l:
            l["wait"] = round(l["wait"])
    legs = [l for l in legs if not (l["mode"] == "walk" and l["min"] <= 0)]
    rides_kept = sum(1 for l in legs if l["mode"] != "walk")
    cur = sum(l["min"] for l in legs) + sum(l.get("wait", 0) for l in legs)
    diff = round(total) - cur
    if diff != 0:
        walks = [l for l in legs if l["mode"] == "walk"]
        if walks:
            walks[-1]["min"] = max(0, walks[-1]["min"] + diff)
        elif diff > 0:
            legs.append({"mode": "walk", "line": None, "min": diff})
        legs = [l for l in legs if not (l["mode"] == "walk" and l["min"] <= 0)]
    return {"total": round(total), "xfers": max(0, rides_kept - 1), "legs": legs}


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
    else:
        olat, olon = float(request.args["olat"]), float(request.args["olon"])
    dlat, dlon = float(request.args["dlat"]), float(request.args["dlon"])
    # Same transfer cap the map used, so the breakdown total matches the cell's colored time
    # (and so it hits the SAME _dest_key-bucketed cache as the matching /compute).
    max_rides = _req_max_rides()
    # RAPTOR path (Phase 2, arrive-by only): trace the cell's journey from the cached tree ->
    # breakdown total EQUALS the cell's map value by construction (no R5). depart-after keeps the
    # R5-validated breakdown below. Off-grid points snap to nearest cell.
    if USE_RAPTOR and RAPTOR_SEMANTIC == "arriveby":
        ci = _RAPTOR.cell_index.get(cid) if cid is not None else None
        if ci is None:
            ci = _nearest_raptor_cell(olat, olon)
        res = _raptor_tree(dlat, dlon, max_rides)["tree"].itinerary(ci) if ci is not None else None
        res = dict(res) if res else {"error": "no route"}
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
        res = fastest_itin(olat, olon, dlat, dlon, max_rides) or {"error": "no route"}
        if cid is not None and "error" not in res:
            with _CELL_CACHE_LOCK:
                _CELL_CACHE.setdefault(dkey, {})[cid] = res
    res = dict(res)                       # don't mutate the cached object with olat/olon
    res["olat"], res["olon"] = round(olat, 5), round(olon, 5)
    return jsonify(res)


@app.route("/attribution")
@limiter.limit("12/minute")
def _attribution():
    """The "color by line" map: dominant transit line per cell. The ONLY trigger for the
    full-grid itinerary build (lazy here, on the user's first toggle). Cached + in-flight
    de-duped, so /itinerary also benefits once it has run."""
    dlat = float(request.args["dlat"]); dlon = float(request.args["dlon"])
    max_rides = _req_max_rides()
    # RAPTOR path (Phase 2, arrive-by only): dominant line per cell from the same traced tree as
    # the map (no R5, ~0.14s, no _HEAVY_LOCK / fan spike, deterministic). depart-after keeps the
    # R5 color-by-line below.
    if USE_RAPTOR and RAPTOR_SEMANTIC == "arriveby":
        attr = _raptor_attribution(dlat, dlon, max_rides)
        print(f"[attr:raptor] ({dlat:.4f},{dlon:.4f}) rides={max_rides} -> {len(attr)} cells")
        return jsonify(attr)
    ckey = _coarse_key(dlat, dlon, max_rides)
    # CACHE HIT (~110m): return the same attribution dict instantly — no R5 work, no lock.
    cached_res = _result_cache_get(_ATTR_RESULT_CACHE, ckey)
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
        _result_cache_put(_ATTR_RESULT_CACHE, ckey, attr)
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
    dlat = float(request.args["dlat"]); dlon = float(request.args["dlon"])
    max_rides = _req_max_rides()
    t0 = dt.datetime.now()
    out = _raptor_mc(dlat, dlon, max_rides)
    ms = (dt.datetime.now() - t0).total_seconds() * 1000
    print(f"[variance:raptor] ({dlat:.4f},{dlon:.4f}) rides={max_rides} {ms:.0f}ms "
          f"-> {len(out['realistic'])} cells")
    return jsonify({"dest": [dlat, dlon], **out, "ms": round(ms)})


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
    return (html.replace("/*__VIZ__*/", viz)
                .replace("__CELLS__", json.dumps(CELLS_GEOJSON))
                .replace("__LINES__", json.dumps(LINES)))


PAGE_HTML = _build_page()


@app.route("/")
def _index():
    return PAGE_HTML


if __name__ == "__main__":
    # threaded=True so a slow /compute_exact or /attribution can't freeze hover (/itinerary)
    # or /compute. R5 routing is thread-safe (read-only shared network; per-task Java request
    # clones) and jpype auto-attaches Python threads to the JVM; heavy jobs are additionally
    # serialised by _HEAVY_LOCK while the light single-OD /itinerary never takes it.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8000")), threaded=True)
