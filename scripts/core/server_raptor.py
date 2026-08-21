#!/usr/bin/env python
"""RAPTOR server glue — the JVM-free engine integration extracted from scripts/server.py.

This module owns the RAPTOR engine state (the loaded engine, walk-graph snap tables, the
per-workplace tree / Monte-Carlo / egress caches + their locks) and the builders that turn a
workplace into the served map, hover breakdown, color-by-line, variance overlay, and drawn
route geometry. The thin Flask route handlers stay in scripts/server.py and call into here.

IMPORT BOUNDARY (no cycle): this module imports ONLY from ``core`` (config, geo helpers, the
RAPTOR engine, the journey tracer) — NEVER from ``server``. ``server.py`` imports THIS module,
parses the boot context (GTFS, grid, optional R5 network), and calls ``init(...)`` once to hand
the resolved engine + walk-graph + R5 deps over. After ``init`` the builders read this module's
own globals.

JVM-FREE INVARIANT: this module must NOT import r5py / jpype / com.conveyal. The only R5 use
on the RAPTOR map path (the USE_WALK_GRAPH=0 fallback in ``raptor_egress_purewalk``) goes
through the ``network`` module reference INJECTED by ``init`` — server.py imports r5py there,
not here.

The shared flags (USE_RAPTOR / RAPTOR_SEMANTIC / USE_WALK_GRAPH / RAPTOR_MC / DEFAULT_MAX_RIDES
/ DEFAULT_SPEED / WALK_SPEEDS) are parsed HERE at import time (pure env reads, JVM-free) so they
have ONE source of truth; server.py re-exports them. ``USE_RAPTOR`` / ``USE_WALK_GRAPH`` can be
toggled off by server.py's boot (missing bakes / engine-init failure) via ``set_flags`` — the
final values flow back so /healthz and the page reflect what actually loaded.
"""
import os
import copy
import json
import math
import sys
import threading
import time
from collections import OrderedDict

from . import config


# Request handlers pass an ordinary short-lived dict when benchmark phase telemetry is enabled.
# Keeping this helper here (rather than a module-level recorder) is deliberate: production has no
# timer allocations and concurrent requests can never mix measurements.  Values ending in ``_ms``
# are durations; the same dict can also carry small integer counts for X-Perf-Phases.
def _perf_add(perf, name, started):
    if perf is not None and started is not None:
        perf[name] = round((time.perf_counter() - started) * 1000.0, 3)

# ---- Shared flags (parsed EARLY: they decide whether the in-process JVM starts at all) ----
# THE DEFAULT IS THE JVM-FREE STACK: USE_RAPTOR + USE_WALK_GRAPH + DEPART-AFTER scheduled + the
# service-noise overlay. R5/the JVM is no longer loaded by default. The default SEMANTIC flipped
# arrive-by -> depart-after on 2026-06-17. The served depart-after read is now anchored on the first
# vehicle you choose to board, so controllable pre-board wait is not counted and walk speed is
# monotone by construction. Opt INTO arrive-by-09:00 with RAPTOR_SEMANTIC=arriveby. Opt back into
# the legacy R5 path with USE_RAPTOR=0 (and/or USE_WALK_GRAPH=0 for the R5 walk matrix).
USE_RAPTOR = os.environ.get("USE_RAPTOR", "1").lower() in ("1", "true", "yes", "on")
RAPTOR_SEMANTIC = os.environ.get("RAPTOR_SEMANTIC", "departafter").lower()
RAPTOR_MC = os.environ.get("RAPTOR_MC", "1").lower() in ("1", "true", "yes", "on")
USE_WALK_GRAPH = os.environ.get("USE_WALK_GRAPH", "1").lower() in ("1", "true", "yes", "on")

DEFAULT_MAX_RIDES = 8                             # R5's ride cap (rides = transfers + 1)
RAPTOR_ALT_CHIP_CAP = int(os.environ.get("RAPTOR_ALT_CHIP_CAP", "6"))
RAPTOR_ALT_TRACE_CAP = int(os.environ.get("RAPTOR_ALT_TRACE_CAP",
                                          str(max(RAPTOR_ALT_CHIP_CAP, RAPTOR_ALT_CHIP_CAP * 3))))
RAPTOR_BRANCH_WINDOW_MIN = float(os.environ.get(
    "RAPTOR_BRANCH_WINDOW_MIN", os.environ.get("RAPTOR_PLANNED_ALT_WINDOW_MIN", "10")))
ROUTE_FAMILY_NEAR_TIE_MIN = 3

# Walk-speed toggle (RAPTOR only): scalar = bake reference / selected product pace. Medium is the
# calibrated typical pace; Fast is brisk. The access table and egress stay in reference seconds.
WALK_SPEEDS = config.WALK_SPEEDS
DEFAULT_SPEED = config.DEFAULT_SPEED
DEFAULT_WALK_SCALAR = config.WALK_KMH / WALK_SPEEDS[DEFAULT_SPEED]


def _resolve_walk_speed(speed=DEFAULT_SPEED, walk_scalar=None):
    """Normalize a product pace and derive its graph-reference scalar when omitted.

    Flask normally supplies both values, but these helpers are also public Python entry points.
    Deriving here prevents ``speed="fast"`` from being cached under the fast key while silently
    routing with Medium's scalar. An explicit scalar remains available to oracle/reference tests.
    """
    speed = str(speed or "").lower()
    if speed not in WALK_SPEEDS:
        speed = DEFAULT_SPEED
    if walk_scalar is None:
        walk_scalar = config.WALK_KMH / WALK_SPEEDS[speed]
    return speed, float(walk_scalar)


def set_flags(**kw):
    """Let server.py's boot push back the FINAL flag values (e.g. USE_WALK_GRAPH forced off
    because the bakes are missing, or USE_RAPTOR off after an engine-init failure) so this
    module's builders + the re-exported copies stay consistent with what actually loaded."""
    g = globals()
    for k, v in kw.items():
        g[k] = v


# ---- Bounded LRU for the workplace-keyed caches ----------------------------------------
class BoundedLRU:
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

    def __init__(self, maxsize, copy_mode=None, lock=None, *, maxbytes=None, weight_fn=None):
        self.maxsize = int(maxsize)
        self._copy_mode = copy_mode
        self.maxbytes = None if maxbytes is None else max(0, int(maxbytes))
        self._weight_fn = weight_fn
        self._od = OrderedDict()
        self._weights = {}
        self._bytes = 0
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
            stored = copy.deepcopy(value) if self._copy_mode == "deep" else value
            weight = (max(0, int(self._weight_fn(stored)))
                      if self._weight_fn is not None else 0)
            if key in self._od:
                self._bytes -= self._weights.pop(key, 0)
            self._od[key] = stored
            self._weights[key] = weight
            self._bytes += weight
            self._od.move_to_end(key)
            while (len(self._od) > self.maxsize
                   or (self.maxbytes is not None and self._bytes > self.maxbytes)):
                old_key, _old_value = self._od.popitem(last=False)
                self._bytes -= self._weights.pop(old_key, 0)

    def pop(self, key, default=None):
        with self.lock:
            if key not in self._od:
                return default
            self._bytes -= self._weights.pop(key, 0)
            return self._od.pop(key)

    def clear(self):
        with self.lock:
            self._od.clear()
            self._weights.clear()
            self._bytes = 0

    def __contains__(self, key):
        with self.lock:
            return key in self._od

    def __len__(self):
        with self.lock:
            return len(self._od)

    @property
    def nbytes(self):
        """Current caller-supplied payload weight (bookkeeping objects excluded)."""
        with self.lock:
            return self._bytes


class BoundedCellCache(dict):
    """Insertion-ordered bounded mapping for lazy per-cell response fragments.

    Reads deliberately do not refresh order: Flask can read these without taking the mutation
    locks, while every write already happens under ``_GEOM_LOCK``, ``_ALT_LOCK``, or ``_TYP_LOCK``.
    FIFO eviction is deterministic and bounds adversarial citywide poking just as effectively as an
    LRU; the browser separately keeps the actively selected route in its own protected LRU.
    """

    def __init__(self, maxsize):
        super().__init__()
        self.maxsize = max(1, int(maxsize))

    def __setitem__(self, key, value):
        if key not in self and len(self) >= self.maxsize:
            del self[next(iter(self))]
        super().__setitem__(key, value)


def _owned_payload_nbytes(root, *, borrowed_root_ids=frozenset()):
    """Conservatively estimate bytes retained below an owned cache root.

    This is intentionally a *shadow telemetry* estimator, not an eviction weight.  It runs only
    when the opt-in benchmark health payload asks for it, so normal request/cache mutation paths
    pay no traversal cost.  Identity de-duplication keeps shared geometry aliases from being
    charged twice within one estimate.  ``borrowed_root_ids`` is an explicit set of identities
    owned elsewhere (the process-static engine/feed/walk graph, for example); excluded roots and
    everything reachable only through them contribute zero.

    NumPy arrays are detected without importing NumPy at module import time.  An owning array is
    charged at least payload + a conservative header; a view is charged its own header and walks
    its ``base`` chain.  Thus distinct views onto one allocation contribute their individual
    headers but the backing allocation only once, while two aliases of the same array object are
    also counted once.  This keeps the aggregate useful as a real retained-memory budget instead
    of charging one shared ndarray buffer for every cached view.
    JSON-like containers, strings/bytes/scalars, ordinary ``__dict__`` objects, and slotted known
    payload objects (notably ``MonteCarloScenario``) are all handled.  Unknown opaque objects are
    charged their interpreter-reported size only.
    """
    borrowed = frozenset(int(ident) for ident in borrowed_root_ids)
    seen = set()

    def visit(item):
        ident = id(item)
        if ident in borrowed or ident in seen:
            return 0
        seen.add(ident)
        try:
            shallow = max(0, int(sys.getsizeof(item)))
        except (TypeError, ValueError, OverflowError):
            shallow = 0

        module = getattr(item.__class__, "__module__", "")
        nbytes = None
        if module.startswith("numpy"):
            try:
                nbytes = getattr(item, "nbytes", None)
            except (AttributeError, TypeError, ValueError):
                nbytes = None
        if nbytes is not None:
            try:
                payload = max(0, int(nbytes))
            except (TypeError, ValueError, OverflowError):
                payload = 0
            base = getattr(item, "base", None)
            if base is not None:
                # NumPy views often expose another ndarray view as ``base``.  Walk that chain so
                # every distinct view keeps its small header charge, but the final allocation is
                # reached (and identity-deduped) exactly once.  If the base is explicitly
                # process-borrowed, ``visit`` returns zero and this owned view still contributes
                # its header -- precisely the retained object this cache owns.
                return shallow + visit(base)
            return max(shallow, payload + 128)

        if isinstance(item, memoryview):
            return shallow + max(0, int(getattr(item, "nbytes", 0)))
        if isinstance(item, dict):
            try:
                pairs = list(item.items())
            except RuntimeError:
                # A health read can overlap a lazy nested-cache insertion. The outer cache roots
                # were snapshotted under their locks; if a nested mapping changes during this
                # best-effort shadow walk, retain the safe container charge and try again next poll.
                return shallow
            return shallow + sum(visit(key) + visit(value) for key, value in pairs)
        if isinstance(item, (list, tuple, set, frozenset)):
            try:
                values = list(item)
            except RuntimeError:
                return shallow
            return shallow + sum(visit(value) for value in values)
        if isinstance(item, (str, bytes, bytearray, bool, int, float, complex, type(None))):
            return shallow

        try:
            attrs = getattr(item, "__dict__", None)
        except (AttributeError, TypeError, ValueError):
            attrs = None
        if attrs is not None:
            return shallow + visit(attrs)

        # Slotted objects do not have a __dict__. Walk slots across the MRO so the retained MC
        # scenario's arrays are visible without giving special weight semantics to its class.
        total = shallow
        for cls in getattr(item.__class__, "__mro__", ()):
            slots = getattr(cls, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for name in slots:
                if name in ("__dict__", "__weakref__"):
                    continue
                try:
                    value = getattr(item, name)
                except (AttributeError, TypeError):
                    continue
                total += visit(value)
        return total

    return max(0, int(visit(root)))


def _benchmark_borrowed_root_ids():
    """Identities of process-static roots excluded from cache-owned benchmark estimates.

    Keep this list explicit: shadow telemetry must not discover ownership by type or by chasing
    arbitrary globals.  The engine/feed data, access tables, grid identity, walk graph, and snap
    tables are boot-owned.  Per-workplace egress, trace, geometry, and MC objects are deliberately
    absent and therefore charged to their cache roots.
    """
    roots = [
        _RAPTOR, _WG, _RAPTOR_STOPS, _RAPTOR_CELL_POS,
        _WG_STOP_NODES, _WG_STOP_CONN, _WG_CELL_NODES, _WG_CELL_CONN, _WG_STOP_GIDS,
        ORIGIN_LL, _NETWORK, _NET, _SNAPPED_GRID,
    ]
    engine = _RAPTOR
    if engine is not None:
        for name in (
                "data", "access_off", "access_to", "access_w", "cell_ids", "cell_index",
                "cell_deps", "dep_grid", "Tgrid", "Tgrid_planned", "Tgrid_mc",
                "_walk_scaled_data"):
            roots.append(getattr(engine, name, None))
        data = getattr(engine, "data", None)
        if isinstance(data, dict):
            # JourneyTree retains a few feed arrays/tables directly as convenience aliases. Add
            # the explicit top-level data values so those aliases remain borrowed even though the
            # containing ``data`` dict itself is excluded from traversal.
            roots.extend(data.values())
    # Do not exclude interned primitive identities globally (a feed scalar ``1`` must not make
    # every independently-owned ``1`` disappear from the estimate). Borrowed ownership starts at
    # containers/arrays/objects; primitive descendants are modest and conservatively re-counted.
    primitive = (str, bytes, bytearray, bool, int, float, complex, type(None))
    return frozenset(id(root) for root in roots if not isinstance(root, primitive))


def _array_tuple_weight(value):
    """Payload bytes for tuples/lists/dicts made primarily of numpy arrays."""
    seen = set()

    def visit(item):
        ident = id(item)
        if ident in seen:
            return 0
        seen.add(ident)
        nbytes = getattr(item, "nbytes", None)
        if nbytes is not None and item.__class__.__module__.startswith("numpy"):
            return int(nbytes) + 128
        if isinstance(item, dict):
            return 64 + sum(visit(k) + visit(v) for k, v in item.items())
        if isinstance(item, (list, tuple)):
            return 64 + sum(visit(v) for v in item)
        return 64

    return visit(value)


def _walkpath_tree_weight(tree):
    """Owned predecessor/distance payload only; the referenced WalkGraph is process-global."""
    return 1024 + sum(int(getattr(getattr(tree, name, None), "nbytes", 0))
                      for name in ("_snodes", "_dist", "_pred"))


class Busy(Exception):
    """Raised when a heavy full-grid build is needed but its lock is already held; the route
    turns this into a 503 instead of blocking behind the running job."""


# ---- RAPTOR engine state (populated by init() at boot) --------------------------------
# When USE_WALK_GRAPH the walk-baked (slope-aware) access table + the JVM-free walk router serve
# the whole map path; otherwise R5 stays in-process for the walk matrix (and, under departafter,
# the R5 hover/color-by-line).
_RAPTOR = None
_RAPTOR_STOPS = None             # GeoDataFrame of stop coords keyed by gid (egress destinations)
_RAPTOR_CELL_POS = None          # cell id -> engine cell index (R5 egress alignment)
_WG = None                       # WalkGraph (JVM-free) when USE_WALK_GRAPH
_WG_STOP_NODES = _WG_STOP_CONN = _WG_CELL_NODES = _WG_CELL_CONN = None
_WG_STOP_GIDS = None             # gids aligned to _WG_STOP_NODES rows

# Boot context injected by server.py (the geo/R5 deps used only by the USE_WALK_GRAPH=0
# fallback in raptor_egress_purewalk — server.py imports r5py, not this module).
ORIGIN_LL = None                 # {cellId: (lat, lon)}
_NEED_R5 = True                  # mirrors server's _NEED_R5 (true unless fully JVM-free)
_NETWORK = None                  # core.network module (r5py) — only on the R5 fallback path
_NET = None                      # the warm R5 network
_SNAPPED_GRID = None             # snapped grid GeoDataFrame (R5 pure-walk matrix)
_DEP = None                      # model departure datetime


def init(*, raptor, raptor_stops, raptor_cell_pos, wg, wg_stop_nodes, wg_stop_conn,
         wg_cell_nodes, wg_cell_conn, wg_stop_gids, origin_ll, need_r5, network=None,
         net=None, snapped_grid=None, dep=None):
    """Hand the boot-resolved RAPTOR engine + walk-graph snap tables (+ the R5 deps for the
    USE_WALK_GRAPH=0 fallback) to this module.

    This is a boot/test-only, quiescent transition — it is not a live graph reload primitive.
    Refuse to replace module state while any request-owned build is in flight; clearing those
    registries would strand waiters on an old graph.  Once quiescence is established, release
    every graph-derived cache before publishing the new globals.
    """
    g = globals()
    with _WALKPATH_BUILD_LOCK:
        assert not _WALKPATH_INFLIGHT, "init requires no in-flight workplace walk builds"
    with _CELL_WALKPATH_BUILD_LOCK:
        assert not _CELL_WALKPATH_INFLIGHT, "init requires no in-flight cell walk builds"
    with _RAPTOR_TREE_BUILD_LOCK:
        assert not _RAPTOR_TREE_INFLIGHT, "init requires no in-flight RAPTOR tree builds"
    with _RAPTOR_MC_LOCK:
        assert not _RAPTOR_MC_INFLIGHT, "init requires no in-flight Monte-Carlo builds"
        assert not _MC_BUSY.locked(), "init requires no active Monte-Carlo kernel"

    # The caches have interlocking ownership: a tree entry contains its per-cell geometry and
    # branch caches, while the MC entry owns its alternative/typical fragments.  Evict their outer
    # entries rather than trying to mutate nested state, so any old graph-derived object loses its
    # only server-owned root together with the graph it came from.
    _RAPTOR_EGRESS_CACHE.clear()
    _WALKPATH_TREE_CACHE.clear()
    _CELL_WALKPATH_TREE_CACHE.clear()
    _TRANSFER_PATH_CACHE.clear()
    _RAPTOR_TREE_CACHE.clear()
    _RAPTOR_MC_CACHE.clear()
    with _MC_SCENARIO_LOCK:
        g["_MC_SCENARIO_ACTIVE"] = None

    # Publish only after invalidating every value tied to the outgoing engine/walk graph.
    g["_RAPTOR"] = raptor
    g["_RAPTOR_STOPS"] = raptor_stops
    g["_RAPTOR_CELL_POS"] = raptor_cell_pos
    g["_WG"] = wg
    g["_WG_STOP_NODES"] = wg_stop_nodes
    g["_WG_STOP_CONN"] = wg_stop_conn
    g["_WG_CELL_NODES"] = wg_cell_nodes
    g["_WG_CELL_CONN"] = wg_cell_conn
    g["_WG_STOP_GIDS"] = wg_stop_gids
    g["ORIGIN_LL"] = origin_ll
    g["_NEED_R5"] = need_r5
    g["_NETWORK"] = network
    g["_NET"] = net
    g["_SNAPPED_GRID"] = snapped_grid
    g["_DEP"] = dep


# ~24 workplace buckets x 3 numpy arrays (~1 MB/entry) — covers a small crowd of distinct
# workplaces without growing past a few tens of MB. No copy: the tuple + arrays are treated
# as immutable by every consumer (the engine reads, never writes them).
_EGRESS_CACHE_MAX = 24
_RAPTOR_EGRESS_CACHE = BoundedLRU(
    _EGRESS_CACHE_MAX, maxbytes=32 * 1024 * 1024,
    weight_fn=_array_tuple_weight)  # coarse_key -> (egress_g, egress_w, purewalk)
# Four predecessor trees at ~10 MB each.  Unlike the 24 lightweight array entries above, these
# retain the k-root distance and predecessor rows needed for later street-path extraction.
_WALKPATH_TREE_CACHE = BoundedLRU(
    4, maxbytes=48 * 1024 * 1024,
    weight_fn=_walkpath_tree_weight)        # coarse (lat,lon) -> reverse workplace walk.PathTree

# A route hover already builds a forward, cell-rooted tree when it draws the access walk.  The
# first detailed pin used to discard that work with its short-lived geometry provider, then build
# the identical tree again while hydrating alternatives.  Retain only the two most recent roots:
# predecessor/distance rows are substantial, so the byte ceiling is just as important as the count
# ceiling.  The graph identity is deliberately part of the key.  A test/boot re-init can reuse
# numeric cell ids, but it must never use predecessor rows belonging to the former walk graph.
_CELL_WALKPATH_TREE_CACHE = BoundedLRU(
    2, maxbytes=24 * 1024 * 1024,
    weight_fn=_walkpath_tree_weight)        # (id(walk_graph), cell index, cap sec) -> forward PathTree


def _transfer_path_weight(value):
    """Tight payload estimate for an immutable cached transfer polyline.

    Coordinates are Python floats inside 2-tuples.  The estimate intentionally includes tuple and
    float object storage, not just the numeric 16 bytes, so the byte cap remains conservative.
    """
    points, _approx = value
    return 64 + len(points) * 112


# Transfer footpaths are static: they depend only on the walk graph and directed stop pair, not the
# workplace, cell, walk-speed scalar, deadline, or route. Planned branch hydration used to build a
# fresh capped Dijkstra for every repeated pair, accounting for roughly half of a first pin. Keep
# immutable point tuples behind both a count bound (adversarial tiny paths) and a hard payload-byte
# bound. ``id(_WG)`` in the key prevents a test/re-init graph from reading paths from an older graph.
_TRANSFER_PATH_CACHE = BoundedLRU(
    1024, maxbytes=8 * 1024 * 1024, weight_fn=_transfer_path_weight)

# A cold JVM-free workplace build and a simultaneous geometry request must not each run the same
# destination-rooted Dijkstra.  The keyed flight covers BOTH the reverse PathTree and the arrays
# projected from it.  Flights are deliberately separate from the LRU locks: no cache lock is held
# across scipy work, and every failure wakes its waiters before the key is removed.
_WALKPATH_BUILD_LOCK = threading.Lock()
_WALKPATH_INFLIGHT = {}                    # coarse_key -> {event, result, error}

# Same-key pins can arrive concurrently (double taps / a retry while a mobile request is in
# flight).  A keyed flight prevents them from each running the same access-side Dijkstra.  As with
# workplace-tree flights, no LRU lock is held while scipy works.
_CELL_WALKPATH_BUILD_LOCK = threading.Lock()
_CELL_WALKPATH_INFLIGHT = {}                # graph-qualified cell key -> {event, result, error}


# ---- cache keys ------------------------------------------------------------------------
def coarse_key(lat, lon, max_rides=DEFAULT_MAX_RIDES, speed=None):
    """Meter-scale destination key for the heavy result caches.

    The old three-decimal (~110m latitude) bucket returned whichever exact workplace coordinate
    happened to populate the bucket first. Two nearby addresses could therefore receive identical
    egress paths, route families, and minutes based on request order. Five decimals matches the
    itinerary/MC seed precision, preserving exact-repeat reuse without cross-address contamination.
    Transfer cap and walk speed remain part of the key; speed-independent reference walk arrays
    intentionally omit only speed.
    """
    base = (round(float(lat), 5), round(float(lon), 5))
    if max_rides != DEFAULT_MAX_RIDES:
        base = base + (int(max_rides),)
    if speed and speed != DEFAULT_SPEED:
        base = base + (speed,)
    return base


def dest_key(lat, lon, max_rides=DEFAULT_MAX_RIDES, speed=None):
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


def mc_seed(dlat, dlon, max_rides, speed):
    """Deterministic per-workplace committed-MC seed. LOAD-BEARING: the byte-identical seed across
    the three MC entry points (/variance's _raptor_mc_build + the two /itinerary?pin=1 per-route
    typical helpers) is what makes the primary compare strip's committed p50 byte-identical to the
    cell's served `realistic` (REAL[id]) — route_typicals and montecarlo draw the same-shaped
    (nR, T) delta arrays from this seed. Keep this the ONE source so the three can never drift.
    (The expression is unchanged from the inline copies it replaced, so served numbers are
    byte-identical to before the extraction.)"""
    import hashlib as _hl
    return int(_hl.sha256(f"{round(dlat,5)},{round(dlon,5)},{int(max_rides)},{speed}"
                          .encode()).hexdigest()[:8], 16)


def _cell_walkpath_tree(ci, cell_ll):
    """Return the exact forward PathTree for a cell's access walks.

    Geometry providers are deliberately request-scoped, while a cell tap often follows the
    ordinary hover that drew this same access leg.  This tiny graph-qualified cache bridges that
    boundary without caching JSON or changing route choice.  ``cell_ll`` remains an explicit
    input because it pins the source used for a cache miss and makes the graph/cell contract
    visible at the call site.

    The cache holds only immutable predecessor/distance trees.  Same-key work is singleflight,
    and a waiter receives the owner's direct immutable result even if another completion evicts it
    before the waiter wakes.
    """
    wg = _WG
    if wg is None:
        return None
    cap_ref_sec = int(_RAPTOR.access_cap_min * 60 * 2)
    key = (id(wg), int(ci), cap_ref_sec)
    tree = _CELL_WALKPATH_TREE_CACHE.get(key)
    if tree is not None:
        return tree

    owner = False
    with _CELL_WALKPATH_BUILD_LOCK:
        tree = _CELL_WALKPATH_TREE_CACHE.get(key)
        if tree is not None:
            return tree
        flight = _CELL_WALKPATH_INFLIGHT.get(key)
        if flight is None:
            flight = {"event": threading.Event(), "result": None, "error": None}
            _CELL_WALKPATH_INFLIGHT[key] = flight
            owner = True

    if not owner:
        flight["event"].wait()
        if flight["error"] is not None:
            raise flight["error"]
        return flight["result"]

    try:
        # Use the graph captured above, not the mutable module global, so a later boot/test
        # re-init cannot turn this build into a tree on a different graph than the cache key.
        lat, lon = float(cell_ll[0]), float(cell_ll[1])
        tree = wg.path_tree((lon, lat), cap_ref_sec)
        _CELL_WALKPATH_TREE_CACHE.put(key, tree)
        flight["result"] = tree
        return tree
    except BaseException as exc:
        flight["error"] = exc
        raise
    finally:
        with _CELL_WALKPATH_BUILD_LOCK:
            _CELL_WALKPATH_INFLIGHT.pop(key, None)
            flight["event"].set()


def _walkpath_tree_and_egress(lat, lon):
    """Build/reuse one max-cap reverse PathTree and its exact RAPTOR walk inputs.

    The tree's vectorized projection is identical to ``WalkGraph.one_to_many`` but lets the
    access-cap egress and max-cap pure-walk arrays share one Dijkstra.  The same PathTree is put in
    the geometry cache, so the first route draw reuses its predecessor rows too.
    """
    import numpy as _np

    ckey = coarse_key(lat, lon)
    tree = _WALKPATH_TREE_CACHE.get(ckey)
    egress = _RAPTOR_EGRESS_CACHE.get(ckey)
    if tree is not None and egress is not None:
        return tree, egress

    owner = False
    with _WALKPATH_BUILD_LOCK:
        # Close the race between the optimistic reads above and registering a new flight.
        tree = _WALKPATH_TREE_CACHE.get(ckey)
        egress = _RAPTOR_EGRESS_CACHE.get(ckey)
        if tree is not None and egress is not None:
            return tree, egress
        flight = _WALKPATH_INFLIGHT.get(ckey)
        if flight is None:
            flight = {"event": threading.Event(), "result": None, "error": None}
            _WALKPATH_INFLIGHT[ckey] = flight
            owner = True

    if not owner:
        flight["event"].wait()
        if flight["error"] is not None:
            raise flight["error"]
        return flight["result"]

    try:
        if tree is None:
            tree = _WG.path_tree((lon, lat), config.MAX_MIN * 60, reverse=True)
        if egress is None:
            ecap = _RAPTOR.access_cap_min * 60
            eg = tree.distances_to(_WG_STOP_NODES, _WG_STOP_CONN, ecap)
            fin = _np.isfinite(eg)
            egress_g = _WG_STOP_GIDS[fin].astype(_np.int32)
            egress_w = _np.rint(eg[fin]).astype(_np.int64)
            pw = tree.distances_to(
                _WG_CELL_NODES, _WG_CELL_CONN, config.MAX_MIN * 60)
            purewalk = _np.where(_np.isfinite(pw), _np.rint(pw), -1).astype(_np.int64)
            egress = (egress_g, egress_w, purewalk)

        # Publish only after the full construction succeeded; an exception cannot leave a partial
        # newly-built entry that future callers mistake for a complete result.
        _WALKPATH_TREE_CACHE.put(ckey, tree)
        _RAPTOR_EGRESS_CACHE.put(ckey, egress)
        result = (tree, egress)
    except BaseException as exc:
        with _WALKPATH_BUILD_LOCK:
            flight["error"] = exc
            _WALKPATH_INFLIGHT.pop(ckey, None)
            flight["event"].set()
        raise
    else:
        with _WALKPATH_BUILD_LOCK:
            flight["result"] = result
            _WALKPATH_INFLIGHT.pop(ckey, None)
            flight["event"].set()
        return result


# ---- RAPTOR grid travel-times (flag-gated) --------------------------------------------
def raptor_egress_purewalk(lat, lon):
    """Per-workplace inputs for the RAPTOR engine, via one destination-rooted walk search:
      egress_g/egress_w — W->stop walk seconds (gid-keyed), capped at the access cap;
      purewalk          — W->cell walk seconds (cell order), capped at MAX_MIN.
    The only R5 use on the RAPTOR map path (one light walk tree, not the heavy per-cell pass).
    Cached per meter-scale workplace key."""
    import numpy as _np
    ckey = coarse_key(lat, lon)
    cached = _RAPTOR_EGRESS_CACHE.get(ckey)
    if cached is not None:
        return cached
    if USE_WALK_GRAPH:                              # JVM-free hill-aware walk router (no R5)
        # stop->W and cell->W both root at W on the TRANSPOSED graph.  The helper builds once at
        # the larger cap, then applies the original per-array caps and rounding/sentinel policy.
        _tree, res = _walkpath_tree_and_egress(lat, lon)
        return res
    # Legacy R5 walk-matrix branch. Parsing is SHARED with raptor_oracle.egress_purewalk via
    # core.r5_extract (min-dedup, minutes->seconds, -1 sentinel) so the goldens the engine
    # validates against can't drift from what this live path computes.
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point
    from . import r5_extract
    W = gpd.GeoDataFrame({"id": ["w"]}, geometry=[Point(lon, lat)], crs=config.WGS)
    cap = _RAPTOR.access_cap_min
    # egress: W -> stops (walk); our stop ids are "S<gid>" so the id bridge is just the strip
    e = pd.DataFrame(_NETWORK.walk_time_matrix(_NET, W, _RAPTOR_STOPS, _DEP, cap))
    egress_g, egress_w = r5_extract.egress_from_ttm(e, lambda to: int(to[1:]), dtype=_np.int64)
    # pure walk: W -> cells, aligned to the engine's cell order
    pw_ttm = pd.DataFrame(_NETWORK.walk_time_matrix(_NET, W, _SNAPPED_GRID, _DEP, config.MAX_MIN))
    purewalk = r5_extract.purewalk_from_ttm(pw_ttm, _RAPTOR_CELL_POS,
                                            len(_RAPTOR.cell_ids), dtype=_np.int64)
    res = (egress_g, egress_w, purewalk)
    _RAPTOR_EGRESS_CACHE.put(ckey, res)
    return res


# Per-workplace traced tree (Phase 2): one arrive-by tree serves the MAP (actual commute),
# the hover breakdown, AND color-by-line — so they are guaranteed consistent (hover == map).
# 8 entries: each holds a full JourneyTree whose lazy full-grid trace memo runs to tens of MB,
# A fully exercised planned workplace is ~39 MiB (base arrays + up to the bounded deadline-tree
# set), so eight hot entries could exceed the process RAM target before static graph/feed data.
# Four retains a useful multi-workplace working set while hard-bounding that dominant component.
# copy_mode='shallow': callers get dict(entry) so they can't reassign the cached entry's keys
# (tree/cells/dom values themselves are shared and treated as immutable).
_TREE_CACHE_MAX = 4
_RAPTOR_TREE_CACHE = BoundedLRU(_TREE_CACHE_MAX, copy_mode="shallow")
# (coarse_key incl. rides+speed) -> {tree, cells, dom, geom}
_RAPTOR_TREE_BUILD_LOCK = threading.Lock()       # registry only; never held during tree work
_RAPTOR_TREE_INFLIGHT = {}                      # key -> {event, error}

# ---- Hover route GEOMETRY (the drawn journey) ------------------------------------------
# Walk-leg paths come from PathTree predecessor chains over the SAME walk graph the times
# use. The expensive piece is the workplace-rooted REVERSE tree (one ~75-min-cap Dijkstra serving
# both the egress/pure-walk TIME arrays and every cell's drawn path), so it is populated by the
# cold compute build and cached per meter-scale workplace key;
# entries are ~10 MB (k=4 dist+pred over the 215k-node graph), hence the small bound. Walk
# paths are SPEED-INVARIANT (the scalar multiplies every edge uniformly — route choice and
# node chains don't change), so the key carries no speed/rides.
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
            ckey = coarse_key(self.dlat, self.dlon)
            t = _WALKPATH_TREE_CACHE.get(ckey)
            if t is None:
                t, _egress = _walkpath_tree_and_egress(self.dlat, self.dlon)
            self._wtree = t
        return self._wtree

    def _cell_tree(self, ci):
        t = self._cell_trees.get(ci)
        if t is None and _WG is not None:
            # 2x the access cap: the table's times are <= cap, but an R5-baked table's path can
            # run longer on OUR graph — generous so extraction never starves the cap.  The global
            # cache makes this exact tree survive the short-lived hover provider and serve the
            # first pin's alternate-route access geometry too.
            t = _cell_walkpath_tree(ci, self._cell_ll(ci))
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

    def prefetch_transfers(self, pairs):
        """Fill missing directed transfer paths with one capped tree per source stop.

        Planned selection knows the handful of surviving branch transfer pairs before geometry is
        hydrated. Grouping them avoids rebuilding the same 4-root graph predecessor tree for every
        target, while each cached pair is still produced by the exact ``PathTree.path_points`` +
        endpoint sealing path used by :meth:`transfer`.
        """
        if _WG is None:
            return
        pending = OrderedDict()
        for s, j in pairs or ():
            s, j = int(s), int(j)
            key = (id(_WG), s, j)
            if _TRANSFER_PATH_CACHE.get(key) is None:
                pending.setdefault(s, OrderedDict()).setdefault(j, None)
        for s, targets in pending.items():
            sl = self._stop_ll(s)
            if sl is None:
                continue
            tree = _WG.path_tree((sl[1], sl[0]), self._TRANSFER_CAP_REF)
            for j in targets:
                jl = self._stop_ll(j)
                if jl is None:
                    continue
                pts = tree.path_points((jl[1], jl[0]))
                result = self._straight(sl, jl) if pts is None else self._seal(pts, sl, jl)
                frozen = (tuple(tuple(point) for point in result[0]), bool(result[1]))
                _TRANSFER_PATH_CACHE.put((id(_WG), s, j), frozen)

    def transfer(self, s, j):
        sl = self._stop_ll(s); jl = self._stop_ll(j)
        if _WG is None or sl is None or jl is None:
            return self._straight(sl, jl)
        key = (id(_WG), int(s), int(j))
        cached = _TRANSFER_PATH_CACHE.get(key)
        if cached is not None:
            points, approx = cached
            return [list(point) for point in points], approx
        t = _WG.path_tree((sl[1], sl[0]), self._TRANSFER_CAP_REF)
        pts = t.path_points((jl[1], jl[0]))
        result = self._straight(sl, jl) if pts is None else self._seal(pts, sl, jl)
        frozen = (tuple(tuple(point) for point in result[0]), bool(result[1]))
        _TRANSFER_PATH_CACHE.put(key, frozen)
        return [list(point) for point in frozen[0]], frozen[1]


def raptor_tree(lat, lon, max_rides=DEFAULT_MAX_RIDES, speed=DEFAULT_SPEED,
                walk_scalar=None):
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
    speed, walk_scalar = _resolve_walk_speed(speed, walk_scalar)
    key = coarse_key(lat, lon, max_rides, speed)
    with _RAPTOR_TREE_BUILD_LOCK:
        # Pair the miss check with owner registration. Different keys leave this tiny critical
        # section independently and do all routing work concurrently; same-key callers wait on
        # only their owner's Event.
        hit = _RAPTOR_TREE_CACHE.get(key)
        if hit is not None:
            return hit
        flight = _RAPTOR_TREE_INFLIGHT.get(key)
        if flight is None:
            flight = {"event": threading.Event(), "result": None, "error": None}
            _RAPTOR_TREE_INFLIGHT[key] = flight
            owner = True
        else:
            owner = False

    if not owner:
        flight["event"].wait()
        if flight["error"] is not None:
            raise flight["error"]
        # Return the owner's strong result through the documented shallow top-level boundary.
        # With concurrent builds for >cache-size destinations, this key may already have been
        # evicted before the waiter is scheduled even though its owner succeeded.
        return dict(flight["result"])

    try:
        result = _raptor_tree_build(
            key, lat, lon, max_rides=max_rides, speed=speed, walk_scalar=walk_scalar)
    except BaseException as exc:
        with _RAPTOR_TREE_BUILD_LOCK:
            flight["error"] = exc
            _RAPTOR_TREE_INFLIGHT.pop(key, None)
            flight["event"].set()
        raise
    else:
        with _RAPTOR_TREE_BUILD_LOCK:
            flight["result"] = result
            _RAPTOR_TREE_INFLIGHT.pop(key, None)
            flight["event"].set()
        return result


def _raptor_tree_build(key, lat, lon, max_rides=DEFAULT_MAX_RIDES, speed=DEFAULT_SPEED,
                       walk_scalar=None):
    """Uncached owner path for :func:`raptor_tree`; callers must register a keyed flight first."""
    speed, walk_scalar = _resolve_walk_speed(speed, walk_scalar)
    egress_g, egress_w, purewalk = raptor_egress_purewalk(lat, lon)
    tree5 = None
    if RAPTOR_SEMANTIC == "departafter":
        # Served depart-after is the planned/scheduled value: choose the first vehicle boarding in
        # the morning window, derive home departure from that boarding, and do not charge the
        # controllable wait before it. One tree serves both selector slots for API compatibility
        # ([scheduled, scheduled]); the service-noise MC still supplies bad-day fragility.
        tree = _RAPTOR.journey_tree_departafter(egress_g, egress_w, purewalk, percentile=50.0,
                                                max_rounds=int(max_rides), walk_scalar=walk_scalar,
                                                planned=True)
        tree5 = tree
        commute = tree.commute()                          # scheduled painted minute
        cells = {c: ([int(commute[i]), int(commute[i])] if commute[i] >= 0 else [None, None])
                 for i, c in enumerate(_RAPTOR.cell_ids)}
        domd = None                                       # lazy; filled by raptor_attribution
    else:
        tree = _RAPTOR.journey_tree(egress_g, egress_w, purewalk, max_rounds=int(max_rides),
                                    walk_scalar=walk_scalar)
        commute, dom = tree.commute_and_dominant()        # arrive-by: one tree, ~0.1s — eager is fine
        cells = {c: ([int(commute[i]), int(commute[i])] if commute[i] >= 0 else [None, None])
                 for i, c in enumerate(_RAPTOR.cell_ids)}
        domd = {c: dom[i] for i, c in enumerate(_RAPTOR.cell_ids) if dom[i] is not None}
    # "geom": per-cell TYPICAL (p50/arrive-by) hover-route geometry responses (ci -> /itinerary dict
    # incl. "geom"), filled lazily by /itinerary under _GEOM_LOCK; shared across the LRU's shallow
    # copies so it lives exactly as long as the tree it was traced from. "geom5" is the SAME cache
    # for the depart-after BEST-CASE (p5) journey (None entry under arrive-by — p5==p50, one tree).
    # "tree5" is the p5 DepartAfterJourneyTree (None under arrive-by). The arrive-by entry shape is
    # unchanged for the keys it had ({tree, cells, dom, geom}); tree5/geom5 are additive + None there.
    # "branch_geom": per-cell PRE-VARIANCE pinned branch alternatives (ci -> alts list): a planned
    # pin must serve its deterministic branch alts even BEFORE /variance builds the MC entry, and
    # there is no mc["alt_geom"] to cache into yet — so the result is cached here, on the entry of
    # the very tree the branches are traced from (right lifetime for free: per-workplace key,
    # LRU-evicted with the tree). Mutated in place under _ALT_LOCK like mc["alt_geom"]; once
    # /variance lands, the MC entry's (ci, "branches") cache takes over and this stops being read.
    entry = {"tree": tree, "tree5": tree5, "cells": cells, "dom": domd,
             "planned": RAPTOR_SEMANTIC == "departafter",
             "geom": BoundedCellCache(128),
             "geom5": (BoundedCellCache(128) if tree5 is not None else None),
             "branch_geom": BoundedCellCache(64)}
    _RAPTOR_TREE_CACHE.put(key, entry)
    return entry


def compute_raptor(lat, lon, max_rides=DEFAULT_MAX_RIDES, speed=DEFAULT_SPEED,
                   walk_scalar=None):
    """Grid travel-times via the RAPTOR engine -> {id: [best, real]} (same shape as compute).
    BOTH semantics now serve from the cached traced tree so map==refine==hover (JVM-free):
    arrive-by uses the single-deadline tree (actual commute); depart-after uses the
    planned scheduled value in both slots ([scheduled, scheduled])."""
    return raptor_tree(lat, lon, max_rides, speed, walk_scalar)["cells"]


def nearest_raptor_cell(olat, olon):
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


def raptor_attribution(dlat, dlon, max_rides=DEFAULT_MAX_RIDES, speed=DEFAULT_SPEED,
                       walk_scalar=None):
    """{cellId: dominant line} from the cached tree (color-by-line, no R5). Arrive-by populates
    ``dom`` eagerly at build time (one tree, ~0.1s). Depart-after leaves it None and traces it
    LAZILY here on the first color-by-line request (``dominant()`` traces the window's ~20 per-T*
    trees, ~0.9s — kept OFF the /compute + hover path); the result is cached ON THE TREE (not the
    entry, which is a shallow copy), and the tree is shared by reference across cache copies, so
    repeats are free."""
    entry = raptor_tree(dlat, dlon, max_rides, speed, walk_scalar)
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
_RAPTOR_MC_CACHE = BoundedLRU(_MC_CACHE_MAX, copy_mode="shallow")
# (coarse_key incl. rides+speed) -> {realistic, variance, alt_bundle, alt_chips, alt_geom, dlat,
#  dlon}. The alt_* keys back /itinerary's drawn alternative routes: alt_bundle is the engine's
# dominance-window geometry handle ({alt_stop: per-cell {line: access_stop}}, the route traced from
# the SAME cached primary tree via itinerary_via_stop — no perturbed re-trace), alt_chips[ci] the
# served alt-line chip set per cell (same primary-exclusion + configured cap applies), alt_geom[ci]
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
# (/compute, /itinerary, /healthz included). Contention raises Busy -> 503 + Retry-After; the
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

# One private lossless MC scenario accelerates route scoring after /variance.  It is deliberately
# NOT stored in the bounded result cache: a cache entry holds only an opaque token, so shallow-copy
# reads and Flask serialization can never duplicate or expose the ~20MiB profile.  A newer retained
# scenario evicts the old one; an old token then cleanly falls back to the full exact kernel.
_MC_SCENARIO_MAX_BYTES = 32 * 1024 * 1024
_MC_SCENARIO_LOCK = threading.Lock()
_MC_SCENARIO_ACTIVE = None                 # (opaque token, coarse key, MonteCarloScenario)
_MC_SCENARIO_SEQ = 0


def _retain_mc_scenario(key, scenario):
    """Retain ``scenario`` iff it is valid and within the hard one-scenario RAM budget."""
    if scenario is None or int(getattr(scenario, "nbytes", _MC_SCENARIO_MAX_BYTES + 1)) > \
            _MC_SCENARIO_MAX_BYTES:
        return None
    global _MC_SCENARIO_ACTIVE, _MC_SCENARIO_SEQ
    with _MC_SCENARIO_LOCK:
        _MC_SCENARIO_SEQ += 1
        token = f"mc-scenario-{_MC_SCENARIO_SEQ:x}"
        _MC_SCENARIO_ACTIVE = (token, key, scenario)
        return token


def _mc_scenario_for(token, key):
    """Resolve an internal token to a strong local scenario reference, or reject/fallback."""
    if not token:
        return None
    with _MC_SCENARIO_LOCK:
        active = _MC_SCENARIO_ACTIVE
        if active is None or active[0] != token or active[1] != key:
            return None
        return active[2]


def raptor_mc(dlat, dlon, max_rides=DEFAULT_MAX_RIDES, speed=DEFAULT_SPEED,
              walk_scalar=None,
              perf=None):
    """{"realistic": {id: min}, "variance": {id: {frag, std, stuck, alt}}} for this workplace.
    realistic = MC p50 (clamped >= perfect); frag = p90-p50 bad-day delta; stuck = fraction of
    draws hitting the cap; alt = lines that become dominant under delays (EXCLUDING the cell's
    normal line). Reachability follows the perfect map (unreachable cells are omitted).

    Concurrency (mirrors /compute_exact + _itineraries_cached): the first arriver for a key
    owns the build; same-key arrivals wait on its Event then re-read the cache; a build needed
    while ANOTHER key's build holds _MC_BUSY raises Busy (-> 503 + Retry-After) instead of
    pinning a waitress thread queued on the serialized numba kernel."""
    speed, walk_scalar = _resolve_walk_speed(speed, walk_scalar)
    key = coarse_key(dlat, dlon, max_rides, speed)
    with _RAPTOR_MC_LOCK:
        hit = _RAPTOR_MC_CACHE.get(key)          # shallow copy: callers can't reassign cache keys
        if hit is not None:
            if perf is not None:
                perf["variance.cache_hit"] = 1
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
            raise Busy()         # surface the same retryable 503 the owner's request got
        return res
    try:
        if not _MC_BUSY.acquire(blocking=False):
            raise Busy()         # another workplace's MC is running -> 503, don't queue
        try:
            return _raptor_mc_build(key, dlat, dlon, max_rides, speed, walk_scalar, perf=perf)
        finally:
            _MC_BUSY.release()
    finally:
        with _RAPTOR_MC_LOCK:
            _RAPTOR_MC_INFLIGHT.pop(key, None)
        event.set()


def _raptor_mc_build(key, dlat, dlon, max_rides, speed, walk_scalar, perf=None):
    """The actual MC build + cache write for raptor_mc (callers go through the guard above).

    Semantic-agnostic build: the committed-plan MC (``RaptorEngine.montecarlo``) takes the
    workplace's CACHED traced tree (arrive-by ``JourneyTree`` OR depart-after
    ``DepartAfterJourneyTree`` p50) and re-runs the SAME committed model on it (commit the home
    departure + first board from the displayed plan, re-optimize the tail per perturbed draw). The
    SERVED shape differs by semantic (see _variance): arrive-by serves the committed ``realistic``
    as the headline typical; depart-after serves ONLY the ``frag``/``stuck``/``alt`` overlay (the
    headline typical is the bare depart-after p50 the map already paints, NOT the committed value).

    ``perfect`` floors the draws at the cell's painted p50 (``cells[c][1]``) under BOTH semantics:
    arrive-by ``[1]==[0]==`` the perfect commute (byte-identical to the old ``cells[c][0]`` floor);
    depart-after ``[1]==`` the served p50 HEADLINE. Flooring at p50 makes ``frag = p90-p50 >= 0`` by
    construction (one self-consistent distribution: frag/std/stuck off the SAME p50-floored draws),
    and under arrive-by keeps ``realistic >= commute`` (perfect<=committed). The internal ``realistic``
    dict is still computed (arrive-by serves it; depart-after ignores it). The PRIMARY line dropped
    from the alt chips is the cell's dominant line, which arrive-by populates eagerly (entry["dom"])
    and depart-after traces lazily here (the per-T* trees ``committed_first_legs`` already built)."""
    import numpy as _np
    t_phase = time.perf_counter() if perf is not None else None
    entry = raptor_tree(dlat, dlon, max_rides, speed, walk_scalar)   # perfect map + dom (cached)
    cells = entry["cells"]
    ids = _RAPTOR.cell_ids
    # Floor the committed MC at the painted p50 (cells[c][1]): arrive-by [1]==[0]==commute (the old
    # behavior, byte-unchanged); depart-after [1]==the served p50 headline. This makes frag=p90-p50>=0
    # under both, off ONE p50-floored distribution. (depart-after no longer SERVES `realistic` — only
    # the frag/stuck/alt overlay — so the floor's role here is purely to anchor the frag tail at the
    # served p50, not to publish a number above it.)
    perfect = _np.array([(cells[c][1] if cells[c][1] is not None else -1) for c in ids], _np.int32)
    egress_g, egress_w, purewalk = raptor_egress_purewalk(dlat, dlon)
    _perf_add(perf, "variance.inputs_ms", t_phase)
    # deterministic per-workplace seed -> the realistic numbers are stable across reboots/reloads
    seed = mc_seed(dlat, dlon, max_rides, speed)
    mc, scenario = _RAPTOR.montecarlo(
        egress_g, egress_w, purewalk, perfect=perfect, seed=seed,
        walk_scalar=walk_scalar, max_rounds=int(max_rides), tree=entry.get("tree"),
        capture_scenario=True, perf=perf)         # capture happens inside the existing draw sweep
    scenario_token = _retain_mc_scenario(key, scenario)
    # dominant line per cell (to drop the cell's PRIMARY line from its alt chips). Arrive-by filled
    # entry["dom"] eagerly at tree build; depart-after left it None (lazy) -> trace it now off the
    # SAME cached tree (the per-T* trees are already built by committed_first_legs above, so this is
    # cheap and stays JVM-free). raptor_attribution caches it on the tree for /attribution reuse.
    dom = entry["dom"]
    if dom is None and mc["alt"]:
        dom = raptor_attribution(dlat, dlon, max_rides, speed, walk_scalar)
    dom = dom or {}
    t_phase = time.perf_counter() if perf is not None else None
    realistic, variance = {}, {}
    alt_all = mc["alt"]
    alt_chips = {}                                       # ci -> [chip lines] (drawn-route source)
    # frag = the bad-day delta the chip ADDS to the displayed headline. The reconciliation invariant
    # is `displayed_headline + frag == committed_p90`, so frag depends on WHAT the headline is:
    #   - ARRIVE-BY: headline == realistic == committed p50 -> frag = committed_p90 - committed_p50,
    #     which is exactly the engine's `mc["frag"]` (kept byte-identical).
    #   - DEPART-AFTER: headline == the bare served p50 (`perfect[i]` == cells[c][1], the painted
    #     floor), NOT committed p50 (committed p50 drifts above the floor on ~77% of cells). So
    #     mc["frag"] (=committed_p90-committed_p50) would UNDERSTATE the bad day by committed_p50 -
    #     served_p50. Derive it off the absolute instead: frag = committed_p90 - served_p50 (>=0
    #     since the draws are floored at served_p50), so served_p50 + frag == committed_p90 exactly.
    _departafter = (RAPTOR_SEMANTIC == "departafter")
    cp90 = mc["committed_p90"]
    for i, c in enumerate(ids):
        if perfect[i] < 0:                              # follow the perfect map's reachability
            continue
        if not _departafter:                           # depart-after never SERVES realistic (the
            realistic[c] = int(mc["realistic"][i])     # headline is the bare served p50) -> skip it
        frag = (max(0, int(cp90[i]) - int(perfect[i])) if _departafter
                else int(mc["frag"][i]))
        v = {"frag": frag, "std": int(mc["std"][i]),
             "stuck": round(float(mc["stuck"][i]), 2)}
        a = alt_all[i] if alt_all else None
        if a:
            domc = dom.get(c)
            # alt is the dominance window {line: min_minutes}, already sorted closest-first; drop
            # the cell's PRIMARY line (its own chip is redundant) and keep a bounded closest set.
            # Some route-sequence alts can still trace to the exact primary journey; /itinerary
            # removes those after geometry is known, so keep a little headroom for the trace source
            # and apply the display cap only after final trace/dedupe.
            a = {k: vv for k, vv in a.items() if k != domc}
            if a:
                items = list(a.items())
                v["alt"] = dict(items[:RAPTOR_ALT_CHIP_CAP])
                alt_chips[i] = [k for k, _vv in items[:RAPTOR_ALT_TRACE_CAP]]
        variance[c] = v
    _perf_add(perf, "variance.assembly_ms", t_phase)
    if perf is not None:
        perf["variance.cells"] = len(variance)
        perf["variance.cells_with_alts"] = len(alt_chips)
    # alt_bundle (per-cell {line: access_stop}) + the matching per-cell chips back /itinerary's
    # drawn alternative routes (traced from the cached primary tree); alt_geom fills lazily on the
    # first alt hover (under _ALT_LOCK).
    out = {"realistic": realistic, "variance": variance,
           "alt_bundle": mc.get("alt_bundle"), "alt_chips": alt_chips,
           "alt_geom": BoundedCellCache(128), "typ": BoundedCellCache(128),
           "dlat": float(dlat), "dlon": float(dlon)}
    if scenario_token is not None:
        out["_scenario_token"] = scenario_token       # internal only; server.py selects public keys
    _RAPTOR_MC_CACHE.put(key, out)
    return out


def mc_peek(dlat, dlon, max_rides, speed):
    """Return the cached MC entry for this workplace+rides+speed, or None — WITHOUT ever
    triggering a build. /itinerary uses this so a hover never pays the ~1s MC cost; the alt
    routes simply stay [] until the frontend's /variance fetch lands (after which the next
    hover finds the entry). A shallow dict(entry) copy, so the returned wrapper is private but
    the shared alt_geom dict is the live cache value (mutated under _ALT_LOCK)."""
    return _RAPTOR_MC_CACHE.get(coarse_key(dlat, dlon, max_rides, speed))


def _legs_have_transit(legs):
    """True iff a traced journey's leg list contains a real TRANSIT leg. Used to drop WALK-ONLY
    "alternatives" — an alt enumerated by ``alt_lines_window`` whose access-stop journey degenerates
    to pure walking (e.g. the line's hop is a sub-2-min ride folded into walk by ``_TINY_HOP_MIN``,
    or the via-stop trace is just a longer walk). Such an alt is labeled with a line name yet has no
    ride: it offers the user nothing but a slower walking path (the reported "walk 26/27/27/30m"
    pointless alternatives), so it must never be served as an alternative route. The PRIMARY itself
    may legitimately be walk-only (a very close cell) — this guard applies ONLY to the alt list."""
    return any((l or {}).get("mode") == "transit" for l in (legs or ()))


def _route_sig(legs):
    """Displayed-journey signature used only to suppress exact duplicate compare rows.

    It intentionally ignores geometry points and keeps mode/line/min/wait, matching what the user
    sees in the route card. Do NOT use this for family grouping: route families are based on the
    directed boarding corridor, not exact displayed-leg equality.
    """
    out = []
    for l in legs or ():
        if not l:
            continue
        out.append((l.get("mode") or "",
                    str(l.get("name") or l.get("line") or ""),
                    int(l.get("min") or 0),
                    int(l.get("wait") or 0)))
    return tuple(out)


def _route_trace_sig(legs):
    out = []
    for l in legs or ():
        if not l:
            continue
        out.append((l.get("mode") or "",
                    str(l.get("name") or l.get("line") or ""),
                    int(l.get("min") or 0),
                    int(l.get("wait") or 0),
                    _leg_geom_sig(l)))
    return tuple(out)


def _route_label(legs):
    names = []
    for l in legs or ():
        if not l or l.get("mode") != "transit":
            continue
        n = l.get("name") or l.get("line")
        if n:
            names.append(str(n))
    return " > ".join(names)


def _alt_total(a):
    nested = a.get("typical") or a.get("best") or {}
    return int(nested.get("total", a.get("min", a.get("total", 10 ** 6))))


def _alt_metric_total(a, metric="r"):
    """Return the minute the compare UI displays for one selectable time mode.

    The canonical selector continues to use ``_alt_total``.  Recommendations are different: they
    must answer for the user's active metric.  Arrive-by realistic minutes land in ``real`` after
    committed replay, while depart-after publishes explicit ``typical`` and ``best`` journeys.
    """
    if metric == "b":
        nested = a.get("best") or a.get("typical") or {}
        value = nested.get("total", a.get("min", a.get("total")))
    else:
        value = a.get("real")
        if value is None:
            nested = a.get("typical") or a.get("best") or {}
            value = nested.get("total", a.get("min", a.get("total")))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 10 ** 6


def _leg_name(leg):
    return str((leg or {}).get("name") or (leg or {}).get("line") or "")


def _alt_display_legs(a):
    for key in ("typical", "best"):
        legs = ((a.get(key) or {}).get("legs") or [])
        if legs:
            return legs
    for key in ("legs", "geom"):
        legs = a.get(key) or []
        if legs:
            return legs
    b = a.get("_branch") or {}
    it = b.get("it") or {}
    return it.get("geom") or it.get("legs") or []


def _planned_branch_proxy_option(branch):
    """Build an exact transit-structure option without street-path hydration.

    Planned branch discovery already formats the winning instance of every structural ride shape.
    Family grouping, branch identity, dominance, and breadth selection need the scheduled totals,
    displayed leg minutes, service metadata, and transit stop polyline — but they do not need an
    access/transfer/egress Dijkstra.  Reconstruct those transit fields directly from the raw ride
    tuples so the selector can reduce hundreds of candidates before the expensive walk geometry is
    assembled.  Sparse symbolic fixtures return ``None`` and use the legacy hydrate-first path.
    """
    it = (branch or {}).get("it") or {}
    formatted = it.get("legs") or []
    raw_rides = [leg for leg in ((branch or {}).get("raw") or ()) if leg[0] == "ride"]
    jt = (branch or {}).get("jt")
    if not formatted or not raw_rides or jt is None or not hasattr(jt, "_ride_pts"):
        return None
    transit_count = sum(1 for leg in formatted if (leg or {}).get("mode") == "transit")
    if transit_count != len(raw_rides):
        return None

    ride_iter = iter(raw_rides)
    legs = []
    try:
        for displayed in formatted:
            leg = {"mode": displayed.get("mode"),
                   "name": displayed.get("line"),
                   "min": int(displayed.get("min") or 0),
                   "pts": []}
            if displayed.get("mode") == "transit":
                raw = next(ride_iter)
                pi, bpos, apos = int(raw[1]), int(raw[4]), int(raw[5])
                feed, route_id, _name, tmode = jt.line_table[int(jt.pat_line[pi])]
                leg.update({"feed": feed, "route_id": route_id, "tmode": tmode,
                            "pts": jt._ride_pts(pi, bpos, apos)})
                if displayed.get("wait"):
                    leg["wait"] = int(displayed["wait"])
            legs.append(leg)
    except (AttributeError, IndexError, StopIteration, TypeError, ValueError):
        return None
    if not any(leg.get("mode") == "transit" and leg.get("pts") for leg in legs):
        return None
    return {
        "line": branch.get("line") or _route_label(legs),
        "source": "branch",
        "_branch": branch,
        "_needs_hydration": True,
        "via_stop": int(branch["stop"]),
        "typical": {"total": int(branch["total"]), "legs": legs},
        "best": {"total": int(branch["total"]), "legs": legs},
    }


def _alt_transit_legs(a):
    return [l for l in _alt_display_legs(a) if (l or {}).get("mode") == "transit"]


def _leg_service_sig(leg):
    vals = [str((leg or {}).get(k) or "") for k in ("feed", "tmode")]
    vals = [v for v in vals if v]
    return "|".join(vals)


def _leg_route_id(leg):
    """Best available stable route id for a hydrated transit leg.

    Geometry legs intentionally expose only display data.  At runtime the engine's line table can
    usually recover the GTFS route id unambiguously from that display tuple; symbolic/degraded
    callers may provide one directly.  Ambiguous display tuples stay unresolved rather than
    guessing which route they represent.
    """
    leg = leg or {}
    explicit = leg.get("route_id") or leg.get("routeId")
    if explicit not in (None, ""):
        return str(explicit)
    data = getattr(_RAPTOR, "data", None) if _RAPTOR is not None else None
    table = (data or {}).get("line_table") or ()
    feed = str(leg.get("feed") or "")
    name = _leg_name(leg)
    mode = str(leg.get("tmode") or "")
    matches = {
        str(route_id) for row_feed, route_id, row_name, row_mode in table
        if str(row_feed or "") == feed and str(row_name or "") == name
        and str(row_mode or "") == mode
    }
    return next(iter(matches)) if len(matches) == 1 else ""


def _leg_service_meta(leg):
    """Readable structural service identity plus display metadata for one proven first ride.

    Service keys are internal join keys for family and branch metadata.  They deliberately use
    the feed-qualified GTFS route id when one is available: that is stable under a display-name
    change and is meaningful when an API response is inspected.  Sparse/degraded callers do not
    have a better durable identity, so their display-name fallback is explicitly labelled rather
    than hidden behind a digest.
    """
    leg = leg or {}
    feed = str(leg.get("feed") or "")
    mode = str(leg.get("tmode") or "")
    name = _leg_name(leg) or "Transit"
    route_id = _leg_route_id(leg)
    # Production geometry resolves to this feed-qualified route id.  A display fallback is only
    # for symbolic/degraded callers that expose no durable route identity at all.  Keep the
    # fallback readable so it cannot be mistaken for a stable route id.
    if route_id:
        key = f"service:feed={feed or 'unknown'};mode={mode or 'unknown'};route={route_id}"
    else:
        key = f"service:feed={feed or 'unknown'};mode={mode or 'unknown'};display={name}"
    return {"key": key, "name": name, "feed": feed, "mode": mode}


def _leg_geom_sig(leg):
    pts = (leg or {}).get("pts") or []
    if not pts:
        return ""
    idxs = [0, (len(pts) - 1) // 2, len(pts) - 1]
    out = []
    for i in dict.fromkeys(idxs):
        try:
            lat, lon = pts[i]
            out.append(f"{round(float(lat) * 10000)}_{round(float(lon) * 10000)}")
        except Exception:
            return ""
    return "_".join(out)


def _leg_dir_sig(leg):
    pts = (leg or {}).get("pts") or []
    parsed = []
    for point in pts:
        try:
            parsed.append((float(point[0]), float(point[1])))
        except Exception:
            continue
    if len(parsed) < 2:
        return ""
    lat0, lon0 = parsed[0]
    nxt = next(((lat, lon) for lat, lon in parsed[1:]
                if abs(lat - lat0) + abs(lon - lon0) > 1e-7), None)
    if nxt is None:
        return ""
    lat1, lon1 = nxt
    # A coarse station cell absorbs platform/stop-coordinate jitter. A 16-way bearing bucket
    # identifies direction without demanding identical stop spacing from services sharing a road
    # or rail trunk. Both values are generic structural tolerances, not network-specific knowledge.
    board = f"{round(lat0 * 1000)}_{round(lon0 * 1000)}"
    dy = lat1 - lat0
    dx = (lon1 - lon0) * math.cos(math.radians((lat0 + lat1) / 2.0))
    angle = math.atan2(dx, dy) % (2.0 * math.pi)
    heading = int(round(angle / (2.0 * math.pi / 16.0))) % 16
    return f"{board}_h{heading}"


def _leg_boarding_profile(leg):
    pts = []
    for point in (leg or {}).get("pts") or ():
        try:
            parsed = (float(point[0]), float(point[1]))
        except Exception:
            continue
        if not pts or parsed != pts[-1]:
            pts.append(parsed)
    if len(pts) < 2:
        return None
    lat0, lon0 = pts[0]
    nxt = next(((lat, lon) for lat, lon in pts[1:]
                if abs(lat - lat0) + abs(lon - lon0) > 1e-7), None)
    if nxt is None:
        return None
    lat1, lon1 = nxt
    north = (lat1 - lat0) * 111_320.0
    east = ((lon1 - lon0) * 111_320.0
            * math.cos(math.radians((lat0 + lat1) / 2.0)))
    norm = math.hypot(east, north)
    if norm <= 0:
        return None
    return {"service": _leg_service_sig(leg), "board": (lat0, lon0),
            "heading": (east / norm, north / norm), "prefix": tuple(pts[:4])}


def _leg_boarding_direction_guard(leg):
    """Coarse initial travel direction, independent of boarding-coordinate bins.

    Route-choice dedupe needs to distinguish the two directions of the same named service, but a
    full corridor signature is too strict: harmless platform jitter and bends later in the ridden
    path must still dedupe.  An eight-way bucket of only the first non-zero segment provides that
    conservative guard; missing geometry keeps the legacy fallback.
    """
    profile = _leg_boarding_profile(leg)
    if profile is None:
        return ""
    east, north = profile["heading"]
    angle = math.atan2(east, north) % (2.0 * math.pi)
    return f"h{int(round(angle / (2.0 * math.pi / 8.0))) % 8}"


def _point_distance_m(a, b):
    lat = math.radians((a[0] + b[0]) / 2.0)
    north = (a[0] - b[0]) * 111_320.0
    east = (a[1] - b[1]) * 111_320.0 * math.cos(lat)
    return math.hypot(east, north)


def _point_segment_distance_m(point, start, end):
    lat = math.radians((point[0] + start[0] + end[0]) / 3.0)
    scale_x = 111_320.0 * math.cos(lat)
    px, py = point[1] * scale_x, point[0] * 111_320.0
    ax, ay = start[1] * scale_x, start[0] * 111_320.0
    bx, by = end[1] * scale_x, end[0] * 111_320.0
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _prefixes_overlap(a, b, radius_m=120.0, min_overlap_m=200.0):
    def nearest(point, path):
        if len(path) < 2:
            return min((_point_distance_m(point, p) for p in path), default=10 ** 9)
        return min(_point_segment_distance_m(point, path[i], path[i + 1])
                   for i in range(len(path) - 1))

    def covered_length(path, other):
        covered = 0.0
        for i in range(len(path) - 1):
            start, end = path[i], path[i + 1]
            midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
            if nearest(midpoint, other) <= radius_m:
                covered += _point_distance_m(start, end)
        return covered

    # Stop spacing may differ, so measure how much of each directed prefix lies near the other
    # polyline. One coincident stop or tiny common launch segment is not a shared corridor.
    return (covered_length(a, b) >= min_overlap_m
            and covered_length(b, a) >= min_overlap_m)


def _same_boarding_profiles(pa, pb, a_name="", b_name=""):
    if pa is None or pb is None or pa["service"] != pb["service"]:
        return False
    board_distance = _point_distance_m(pa["board"], pb["board"])
    dot = pa["heading"][0] * pb["heading"][0] + pa["heading"][1] * pb["heading"][1]
    # The same dynamically discovered service can be boarded on either side of a nearby route bend:
    # its first displayed segment then points along different street axes even though it is one
    # commuter corridor. Admit nearby, non-opposite bends without weakening the generic rule for
    # differently named services. True opposite directions (the important split) remain separate.
    same_line = bool(a_name and a_name == b_name)
    if same_line and board_distance <= 300.0 and dot >= math.cos(math.radians(100.0)):
        return True
    if board_distance > 150.0:
        return False
    if dot < math.cos(math.radians(40.0)):
        return False
    return _prefixes_overlap(pa["prefix"], pb["prefix"])


def _same_boarding_corridor(a, b):
    return _same_boarding_profiles(
        _leg_boarding_profile(a), _leg_boarding_profile(b), _leg_name(a), _leg_name(b))


_SAME_BOARDING_CORRIDOR_IMPL = _same_boarding_corridor


def _leg_corridor_sig(leg):
    """Stable directed BOARDING-corridor identity for a transit leg.

    Only the start and initial direction are used. Sampling the full ridden segment made the same
    service split into different families merely because two alternatives alighted at different
    stops. Feed/mode keeps physically coincident but operationally different services separate.
    """
    service = _leg_service_sig(leg)
    direction = _leg_dir_sig(leg)
    if not direction:
        return ""
    return f"{service or 'service'}|board:{direction}"


def _alt_corridor_sig(a):
    legs = _alt_transit_legs(a)
    return _leg_corridor_sig(legs[0]) if legs else ""


def _discover_family_keys(options):
    """Tolerant, deterministic COMPLETE-LINK clustering of boarding corridors.

    Complete-link compatibility is load-bearing: single-link union can chain an eastbound option
    through a perpendicular bend into a westbound option even though the endpoints are opposites.
    """
    options = list(options or ())
    first_legs = [next(iter(_alt_transit_legs(option)), None) for option in options]
    profiles = [_leg_boarding_profile(leg) for leg in first_legs]
    services = [_leg_service_sig(leg) if leg is not None else None for leg in first_legs]
    names = [_leg_name(leg) if leg is not None else "" for leg in first_legs]
    descriptors = []
    for i, leg in enumerate(first_legs):
        pts = []
        for point in (leg or {}).get("pts") or ():
            try:
                pts.append(f"{float(point[0]):.6f},{float(point[1]):.6f}")
            except Exception:
                continue
        structural = "|".join(pts)
        descriptor = (f"{services[i]}|path:{structural}" if structural
                      else f"line:{_leg_name(leg) if leg else 'other'}")
        descriptors.append(descriptor)
    components = []
    components_by_service = {}

    def same_corridor(i, j):
        # Preserve the long-standing test/extension seam: a caller may deliberately replace the
        # public predicate. Production uses the pre-parsed profiles and avoids thousands of repeat
        # point/service parses; an override still receives the original leg objects.
        if _same_boarding_corridor is not _SAME_BOARDING_CORRIDOR_IMPL:
            return _same_boarding_corridor(first_legs[i], first_legs[j])
        return _same_boarding_profiles(profiles[i], profiles[j], names[i], names[j])

    for i in sorted(range(len(options)), key=lambda idx: (descriptors[idx], idx)):
        service = services[i]
        service_components = components_by_service.setdefault(service, [])
        compatible = [component for component in service_components if all(
            profiles[i] is not None and profiles[j] is not None and same_corridor(i, j)
            for j in component
        )]
        if compatible:
            min(compatible, key=lambda component: min(descriptors[j] for j in component)).append(i)
        else:
            component = [i]
            components.append(component)
            service_components.append(component)
    records = []
    for indices in components:
        # Keep the complete member multiset in the sort key.  A minimum-only identity can collide
        # when two incompatible components happen to share one coarsely quantized descriptor.
        members = "\x1f".join(sorted(descriptors[i] for i in indices))

        def route_structure(option):
            # This order breaker must not include user-facing line labels: a feed can rename a
            # display label without changing the boarding corridor it represents.  It is reached
            # only when first-leg component descriptors tie, where downstream structural detail
            # keeps a response-local ordinal deterministic.
            return tuple(
                (str(leg.get("mode") or ""), _leg_service_sig(leg), _leg_route_id(leg),
                 int(leg.get("min") or 0), int(leg.get("wait") or 0), _leg_geom_sig(leg))
                for leg in _alt_display_legs(option) if leg)

        option_sigs = tuple(sorted(
            (_alt_total(options[i]), route_structure(options[i]))
            for i in indices
        ))
        records.append({"indices": indices,
                        "order": (members, option_sigs, min(indices))})

    # Family ids are concise, readable ordinals over complete discovered corridor components.
    # The ordering includes all observable structure, so route enumeration order and display-name
    # changes cannot alter ids when durable geometry/service structure is unchanged.  When two
    # sparse options have no distinguishing structure at all, their individual ownership is
    # intentionally interchangeable while the emitted key set remains deterministic.
    ordered_records = sorted(records, key=lambda record: record["order"])
    keys = {}
    for ordinal, record in enumerate(ordered_records, 1):
        # Degenerate/missing geometry can produce incompatible singleton components with the same
        # descriptor.  Ordinals keep them conservatively separate without an opaque digest.
        key = f"corridor:{ordinal}"
        for i in record["indices"]:
            keys[id(options[i])] = key
    return keys


def _alt_family_key(a, context=None):
    """Opaque family identity from the first boarded corridor and direction.

    Geometry is authoritative when present, allowing differently named services on the same
    directed corridor to form one family. A dynamic service-name fallback keeps sparse fixtures
    and degraded geometry useful without embedding knowledge of any concrete route.
    """
    if context and id(a) in context:
        return context[id(a)]
    legs = _alt_transit_legs(a)
    first = _leg_name(legs[0]) if legs else ""
    sig = _alt_corridor_sig(a)
    if sig:
        return f"corridor:{sig}"
    return first or "other"


def _leg_arrival_sig(leg):
    pts = []
    for point in (leg or {}).get("pts") or ():
        try:
            parsed = (float(point[0]), float(point[1]))
        except Exception:
            continue
        if not pts or parsed != pts[-1]:
            pts.append(parsed)
    if len(pts) >= 2:
        end_lat, end_lon = pts[-1]
        prev_lat, prev_lon = pts[-2]
        north = (end_lat - prev_lat) * 111_320.0
        east = ((end_lon - prev_lon) * 111_320.0
                * math.cos(math.radians((end_lat + prev_lat) / 2.0)))
        angle = math.atan2(east, north) % (2.0 * math.pi)
        heading = int(round(angle / (2.0 * math.pi / 16.0))) % 16
        endpoint = f"{round(end_lat * 10000)}_{round(end_lon * 10000)}"
        return f"{_leg_service_sig(leg) or 'service'}|arrive:{endpoint}_h{heading}"
    return f"{_leg_service_sig(leg) or 'service'}|line:{_leg_name(leg) or 'transit'}"


def _branch_key_for_transit_legs(legs):
    if not legs:
        return "walk:only"
    if len(legs) == 1:
        return f"walk:{_leg_arrival_sig(legs[0])}"
    return f"tail:{_leg_arrival_sig(legs[-1])}"


def _alt_branch_key(a, fam=None):
    """Destination-facing branch identity inside a boarding-corridor family.

    Intermediate feeders are deliberately omitted: a direct tail and a detour that converges onto
    the same final service/alighting area are one branch and can be dominance-compared. A one-seat
    option is the structural walk-finish branch.
    """
    legs = _alt_transit_legs(a)
    return _branch_key_for_transit_legs(legs)


def _alt_slot_legs(a, slot):
    legs = ((a.get(slot) or {}).get("legs") or [])
    return legs or _alt_display_legs(a)


def _journey_choice_key(legs):
    transit = [leg for leg in legs or () if (leg or {}).get("mode") == "transit"]
    # Prefer the feed-qualified GTFS route id exposed by _leg_service_meta. Display names are a
    # compatibility fallback only when a sparse/degraded leg has no durable route id. This keeps a
    # public selection stable across label changes and distinguishes two same-named routes.
    sequence = tuple(_leg_service_meta(leg)["key"] for leg in transit)
    direction = _leg_boarding_direction_guard(transit[0]) if transit else ""
    return (direction, sequence), _branch_key_for_transit_legs(transit)


def _journey_choice_bucket(legs):
    """Necessary (not sufficient) bucket for the continuous-heading equivalence test."""
    transit = [leg for leg in legs or () if (leg or {}).get("mode") == "transit"]
    sequence = tuple(_leg_service_meta(leg)["key"] for leg in transit)
    return sequence, _branch_key_for_transit_legs(transit)


def _alt_choice_key(a):
    """Service-qualified structural identity for BOTH metric slots."""
    return (_journey_choice_key(_alt_slot_legs(a, "best")),
            _journey_choice_key(_alt_slot_legs(a, "typical")))


def _public_choice_key(a):
    """Readable, JSON-safe public identity for one exact rendered route choice.

    Family and branch describe useful *groups*, but several non-dominated choices may share both.
    The client therefore needs the existing complete structural choice identity, rather than a
    family/branch composite or a response-position ordinal, to keep selection stable.  This is
    deliberately an inspectable serialization, not an opaque digest.
    """
    return "choice:" + json.dumps(_alt_choice_key(a), separators=(",", ":"),
                                 ensure_ascii=True)


def _alt_choice_bucket(a):
    return (_journey_choice_bucket(_alt_slot_legs(a, "best")),
            _journey_choice_bucket(_alt_slot_legs(a, "typical")))


def _journey_choice_equivalent(left_legs, right_legs):
    """Whether two rendered slots describe the same practical transit choice.

    A quantized heading is useful as a stable key, but it has unavoidable bin edges: two trips on
    the same service, toward the same tail, can start 18 degrees apart and land in adjacent coarse
    buckets. Compare the actual unit headings for final dedupe instead. The generous non-opposite
    tolerance mirrors the same-service bend rule used by corridor discovery; a true reverse ride
    remains distinct. Complete route sequence + terminal branch still have to match exactly.
    """
    left = [leg for leg in left_legs or () if (leg or {}).get("mode") == "transit"]
    right = [leg for leg in right_legs or () if (leg or {}).get("mode") == "transit"]
    left_sequence = tuple(_leg_service_meta(leg)["key"] for leg in left)
    right_sequence = tuple(_leg_service_meta(leg)["key"] for leg in right)
    if left_sequence != right_sequence:
        return False
    if _branch_key_for_transit_legs(left) != _branch_key_for_transit_legs(right):
        return False
    if not left:
        return True
    lp = _leg_boarding_profile(left[0]); rp = _leg_boarding_profile(right[0])
    if lp is None or rp is None:
        return _leg_boarding_direction_guard(left[0]) == _leg_boarding_direction_guard(right[0])
    dot = lp["heading"][0] * rp["heading"][0] + lp["heading"][1] * rp["heading"][1]
    return dot >= math.cos(math.radians(100.0))


def _alt_choice_equivalent(left, right):
    """Pairwise choice equivalence across both best/typical display slots."""
    return all(_journey_choice_equivalent(_alt_slot_legs(left, slot),
                                          _alt_slot_legs(right, slot))
               for slot in ("best", "typical"))


def _alt_dedupe_key(a, family_keys=None):
    """Pre-selection identity that preserves structurally distinct destination tails."""
    return _alt_choice_key(a)


def _is_subsequence(shorter, longer):
    if not shorter:
        return True
    i = 0
    for item in longer:
        if item == shorter[i]:
            i += 1
            if i == len(shorter):
                return True
    return False


def _alt_dominates(simple, detour, family_keys=None):
    """Pareto dominance within one practical destination-facing branch.

    Structural simplification is valuable presentation information, but it is not dominance when
    it costs time or physical walking.  Exact semantic equivalents are collapsed separately;
    this predicate preserves every remaining corridor/branch tradeoff unless one route is no
    worse in every user-visible quality dimension and strictly better in at least one.
    """
    sfam = _alt_family_key(simple, family_keys); dfam = _alt_family_key(detour, family_keys)
    sbranch = _alt_branch_key(simple, sfam); dbranch = _alt_branch_key(detour, dfam)
    if sfam != dfam or sbranch != dbranch:
        return False
    sq = (_alt_total(simple), _alt_exact_seconds(simple), _alt_access_walk_min(simple),
          _alt_transfers(simple), _alt_final_walk_min(simple),
          _alt_physical_walk_min(simple), _alt_fragility(simple))
    dq = (_alt_total(detour), _alt_exact_seconds(detour), _alt_access_walk_min(detour),
          _alt_transfers(detour), _alt_final_walk_min(detour),
          _alt_physical_walk_min(detour), _alt_fragility(detour))
    return all(left <= right for left, right in zip(sq, dq)) and any(
        left < right for left, right in zip(sq, dq))


def _prune_dominated_alts(alts, context=(), family_keys=None):
    context = list(context or ())
    alts = list(alts)
    comparisons = context + alts
    family_keys = family_keys or _discover_family_keys(comparisons)

    # Resolve exact route-choice duplicates before the broader dominance pass.  Pairwise `<`
    # ranking is intentionally asymmetric and can let an equal-total alt beat the primary solely
    # because its sampled access-walk polyline sorts first.  Grouping makes the product rule
    # explicit: an equal/faster primary represents the choice; otherwise keep exactly one fastest,
    # deterministic alt.  A genuinely faster alt therefore survives.
    def equivalent_rank(option):
        return _alt_quality_rank(option)

    # Pairwise continuous-heading comparison avoids both false reverse-route collapse and false
    # duplicates at an arbitrary heading-bin boundary. Use complete-link grouping so a chain of
    # progressively bending paths cannot bridge genuinely incompatible endpoints.
    by_choice = []
    groups_by_bucket = {}
    for alt in sorted(alts, key=equivalent_rank):
        bucket = _alt_choice_bucket(alt)
        bucket_groups = groups_by_bucket.setdefault(bucket, [])
        compatible = [group for group in bucket_groups
                      if all(_alt_choice_equivalent(alt, member) for member in group)]
        if compatible:
            compatible[0].append(alt)
        else:
            group = [alt]
            by_choice.append(group)
            bucket_groups.append(group)

    deduped = []
    for options in by_choice:
        winner = min(options, key=equivalent_rank)
        winner_bucket = _alt_choice_bucket(winner)
        primary_matches = [option for option in context
                           if _alt_choice_bucket(option) == winner_bucket
                           if _alt_choice_equivalent(option, winner)]
        if (primary_matches
                and min(_alt_total(option) for option in primary_matches) <= _alt_total(winner)):
            continue
        deduped.append(winner)

    comparisons = context + deduped
    by_dominance = {}
    by_exact_choice = {}
    for option in comparisons:
        transit = _alt_transit_legs(option)
        family = _alt_family_key(option, family_keys)
        branch = _alt_branch_key(option, family)
        by_dominance.setdefault((family, branch), []).append(option)
        by_exact_choice.setdefault(_alt_choice_bucket(option), []).append(option)

    kept = []
    for option in deduped:
        transit = _alt_transit_legs(option)
        family = _alt_family_key(option, family_keys)
        branch = _alt_branch_key(option, family)
        candidates = []
        candidate_ids = set()
        for pool in (by_dominance.get((family, branch), ()),
                     by_exact_choice.get(_alt_choice_bucket(option), ())):
            for candidate in pool:
                if id(candidate) not in candidate_ids:
                    candidate_ids.add(id(candidate))
                    candidates.append(candidate)
        if not any(candidate is not option
                   and _alt_dominates(candidate, option, family_keys)
                   for candidate in candidates):
            kept.append(option)
    return kept


def _family_representative(fam, opts):
    """Fastest truthful anchor; simplicity never outranks a faster branch member."""
    return min(opts, key=_alt_quality_rank) if opts else None


def _alt_raw_legs(a):
    b = a.get("_branch") or {}
    return b.get("raw") or []


def _alt_access_walk_min(a):
    # Newer itinerary payloads split a folded first-platform allowance out of the physical
    # walk.  Prefer that explicit truth whenever it is present; older cached payloads retain
    # their historical ``min`` field as the compatibility fallback.
    legs = _alt_display_legs(a)
    physical = [leg for leg in legs if (leg or {}).get("mode") == "walk"
                and (leg or {}).get("physical_min") is not None]
    if physical:
        total = 0
        for leg in legs:
            if (leg or {}).get("mode") == "transit":
                break
            if (leg or {}).get("mode") == "walk":
                total += _leg_physical_walk_min(leg)
        return total
    raw = _alt_raw_legs(a)
    if raw:
        sec = 0
        for leg in raw:
            if leg[0] == "ride":
                break
            if leg[0] in ("access", "walk", "walk_t", "egress"):
                sec += int(leg[1])
        return sec / 60.0
    total = 0
    for leg in _alt_display_legs(a):
        if leg.get("mode") == "transit":
            break
        if leg.get("mode") == "walk":
            total += int(leg.get("min") or 0)
    return total


def _alt_final_walk_min(a):
    legs = _alt_display_legs(a)
    for leg in reversed(legs):
        if (leg or {}).get("mode") == "walk":
            return _leg_physical_walk_min(leg)
        if (leg or {}).get("mode") == "transit":
            break
    raw = _alt_raw_legs(a)
    if raw:
        last = raw[-1]
        if last[0] in ("egress", "walk"):
            return int(last[1]) / 60.0
        return 0.0
    for leg in reversed(_alt_display_legs(a)):
        if leg.get("mode") == "walk":
            return _leg_physical_walk_min(leg)
        if leg.get("mode") == "transit":
            break
    return 0.0


def _leg_physical_walk_min(leg):
    """Physical walking minutes for ranking, never folded schedule allowance."""
    leg = leg or {}
    value = leg.get("physical_min", leg.get("min", 0))
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _alt_physical_walk_min(a):
    """Total physical walk across an itinerary, with legacy/raw fallbacks."""
    legs = _alt_display_legs(a)
    if any((leg or {}).get("physical_min") is not None for leg in legs):
        return sum(_leg_physical_walk_min(leg) for leg in legs
                   if (leg or {}).get("mode") == "walk")
    raw = _alt_raw_legs(a)
    if raw:
        return sum(int(leg[1]) for leg in raw
                   if leg and leg[0] in ("access", "walk", "walk_t", "egress")) / 60.0
    return sum(_leg_physical_walk_min(leg) for leg in legs
               if (leg or {}).get("mode") == "walk")


def _alt_exact_seconds(a):
    """Unrounded door-to-door seconds when a planned raw trace is still available."""
    raw = _alt_raw_legs(a)
    branch = a.get("_branch") or {}
    try:
        metric_sec = branch.get("metric_sec")
        if metric_sec is not None:
            return max(0, int(metric_sec))
    except (TypeError, ValueError):
        pass
    if raw and branch.get("home") is not None:
        try:
            home = int(branch["home"])
            clock = home
            for leg in raw:
                if leg[0] in ("access", "walk", "walk_t", "egress"):
                    clock += int(leg[1])
                elif leg[0] == "ride":
                    clock = int(leg[3])
            return max(0, clock - home)
        except (IndexError, TypeError, ValueError):
            pass
    # Older/hydrated payloads expose rounded display legs only.  Their displayed total remains
    # the stable approximation rather than inventing precision from formatted components.
    return _alt_total(a) * 60


def _alt_transfers(a):
    return max(0, len(_alt_transit_legs(a)) - 1)


def _alt_fragility(a):
    try:
        return max(0, int(a.get("frag", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _alt_latest_board_anchor(a):
    for leg in _alt_raw_legs(a):
        if leg and leg[0] == "ride" and len(leg) >= 3:
            try:
                return int(leg[2])
            except (TypeError, ValueError):
                break
    return -1


def _is_token_transit_long_walk(a):
    """A negligible ride followed by a long physical finish is presentation-poor, not invalid."""
    transit = _alt_transit_legs(a)
    ride_min = sum(max(0, int((leg or {}).get("min") or 0)) for leg in transit)
    return bool(transit and ride_min <= 2 and _alt_final_walk_min(a) >= 8)


def _alt_quality_rank(a):
    """Shared deterministic rank for exact-choice collapse and branch representatives.

    Time remains authoritative.  All walking fields deliberately use physical walk only; a
    controllable schedule allowance may reconcile the duration but must not make a route look
    walkier or less walkable.  Token-transit is only a final presentation demotion, so it cannot
    erase a route that genuinely improves time or walking.
    """
    return (
        _alt_total(a),
        _alt_exact_seconds(a),
        _alt_access_walk_min(a),
        _alt_transfers(a),
        _alt_final_walk_min(a),
        _alt_fragility(a),
        -_alt_latest_board_anchor(a),
        1 if _is_token_transit_long_walk(a) else 0,
        _alt_selection_tie_key(a),
    )


def _alt_recommendation_rank(a, metric="r"):
    """Server-owned practical recommendation rank for a pinned route choice.

    This deliberately differs from ``_alt_quality_rank``.  The latter is a selector/internal
    representative rank, while this is the user-facing answer to "which of these should I take?"
    Routes first compete on the *displayed active minute*.  Once that is tied, less actual walking
    is more useful than sub-minute schedule precision.  The physical-walk helper excludes a folded
    first-platform/schedule allowance, so changing that presentation allowance cannot make an
    otherwise identical route win or lose.  The final structural key keeps a fully tied answer
    deterministic without encoding any route-name knowledge.
    """
    return (
        _alt_metric_total(a, metric),
        _alt_physical_walk_min(a),
        _alt_transfers(a),
        _alt_fragility(a),
        _alt_exact_seconds(a),
        -_alt_latest_board_anchor(a),
        _alt_selection_tie_key(a),
    )


def _recommend_route_choice(primary, alternatives, metric="r"):
    """Return the best practical route from an already-pruned pinned universe.

    ``primary`` is the canonical map itinerary and remains in this comparison: the map route is
    not presumed to be the best practical recommendation once the pin has discovered all
    non-dominated structural alternatives.
    """
    options = ([primary] if primary is not None else []) + list(alternatives or ())
    return min(options, key=lambda option: _alt_recommendation_rank(option, metric)) if options else None


def _recommend_route_choices(primary, alternatives):
    """Return complete-universe recommendations for both selectable UI metrics."""
    return {metric: _recommend_route_choice(primary, alternatives, metric)
            for metric in ("r", "b")}


# Compatibility alias for internal callers/tests added during the ranking migration.
_alt_representative_rank = _alt_quality_rank


def _planned_branch_structural_tie_key(a):
    """Readable, geometry-free final order for a planned branch proxy.

    Street geometry is a presentation detail.  It must not decide which of two otherwise
    equivalent scheduled candidates gets hydrated: doing so forces access/transfer/egress
    Dijkstras for every tie.  The branch payload already has a complete semantic description of
    that choice, including its access station, pattern topology, scheduled ride times and
    destination-facing tail.  Use that description as the final selector key instead.

    This deliberately is a tuple of labelled primitive fields rather than a digest.  It remains
    inspectable in a debugger and is stable across arbitrary street-polyline changes.  The key is
    reached only after total, access walk, final walk and source all tie; it cannot change the
    ordering of routes with a meaningful quality difference.
    """
    branch = a.get("_branch") or {}
    raw = branch.get("raw") or ()
    route_shapes = iter(branch.get("route_key") or ())
    sequence = []
    for leg in raw:
        if not leg:
            continue
        kind = str(leg[0])
        if kind == "ride" and len(leg) >= 7:
            # ``route_key`` carries the feed-qualified route/pattern topology and ridden stop
            # positions.  The raw tuple supplies the specific scheduled trip times and alight
            # stop.  Keeping both distinguishes genuine scheduled alternatives without looking
            # at a walk polyline.
            try:
                pattern = next(route_shapes)
            except StopIteration:
                pattern = ("pattern-index", int(leg[1]))
            sequence.append(("ride", pattern, int(leg[2]), int(leg[3]),
                             int(leg[4]), int(leg[5]), int(leg[6])))
        elif kind == "access" and len(leg) >= 2:
            sequence.append(("access", int(leg[1])))
        elif kind == "walk_t" and len(leg) >= 4:
            sequence.append(("transfer", int(leg[1]), int(leg[2]), int(leg[3])))
        elif kind == "egress" and len(leg) >= 3:
            sequence.append(("egress", int(leg[1]), int(leg[2])))
        else:
            # Planned raw shapes are deliberately finite.  Preserve an unexpected extension as
            # readable scalar fields instead of silently dropping it from the deterministic key.
            sequence.append((kind, *tuple(str(value) for value in leg[1:])))
    return (
        "planned-branch",
        ("access-stop", int(branch.get("stop", -1))),
        ("home", int(branch.get("home", -1))),
        ("route-and-schedule", tuple(sequence)),
        ("destination-branch", _alt_branch_key(a)),
    )


def _alt_selection_tie_key(a):
    """Last-resort deterministic selector key without needless proxy hydration."""
    if a.get("_needs_hydration"):
        return _planned_branch_structural_tie_key(a)
    # Already-hydrated/window options retain the exact rendered-route ordering used before this
    # optimization.  Only a proxy's formerly incidental street-polyline tie-break changes.
    return ("rendered-route", repr(_route_trace_sig(_alt_display_legs(a))))


def _assert_primary_minimum(primary, alternatives):
    """Reject a branch universe that contradicts the map-authored primary.

    The pinned inspector must not hide a genuinely faster route or silently replace the map's
    route.  Such a result is an upstream mismatch between canonical planned selection and branch
    expansion, so make it a loud invariant failure at their server boundary.
    """
    floor = _alt_total(primary)
    faster = [option for option in alternatives if _alt_total(option) < floor]
    if faster:
        raise AssertionError(
            "planned branch candidate faster than canonical primary: "
            f"primary={floor}, candidates={sorted(_alt_total(option) for option in faster)}")


def _build_family_service_catalog(routes, family_keys):
    """Catalog proven first-board services before dominance/cap selection.

    Every input route is already a complete hydrated O-D itinerary.  Consequently this bounded v1
    never advertises a merely nearby or same-track service: membership is proof that the service
    completed this exact family/branch journey in the traced route universe.
    """
    catalog = OrderedDict()
    for route in routes or ():
        transit = _alt_transit_legs(route)
        if not transit:
            continue
        family = _alt_family_key(route, family_keys)
        branch = _alt_branch_key(route, family)
        service = _leg_service_meta(transit[0])
        branches = catalog.setdefault(family, OrderedDict())
        branch_meta = branches.setdefault(branch, {"services": OrderedDict(), "tails": []})
        services = branch_meta["services"]
        services.setdefault(service["key"], service)
        # This is an ordered ride sequence, not a set. Re-boarding the same named service is a
        # real transfer and must remain visible (A > A > B is not equivalent to A > B).
        tail = tuple(name for leg in transit[1:] if (name := _leg_name(leg)))
        if tail and tail not in branch_meta["tails"]:
            branch_meta["tails"].append(tail)
    return catalog


def _select_diverse_alts(alts, cap, primary=None, complete_selected_families=False,
                         force_include=None):
    """Filter after tracing while preserving generic corridor and tail diversity.

    Distinct boarding corridors get first claim on the bounded card. Remaining slots are assigned
    round-robin to still-unrepresented destination-facing branches, then to the fastest leftovers.

    ``complete_selected_families`` is for the pinned expert disclosure.  The normal ``cap`` still
    decides corridor breadth exactly as before; after that choice, each already-visible family may
    contribute one representative for every other non-dominated destination-facing branch inside
    both the existing per-family near-tie and generic reservation ceilings.  The result may
    therefore exceed ``cap`` without admitting another boarding corridor.  Hover chips and legacy
    callers keep the strict cap.  ``force_include`` accepts one route or a collection selected by
    the separate recommendation ranks from the complete deduped/dominance-pruned pin universe.
    Missing recommendations are appended: neither time mode may advertise a route the card hid
    merely because its display cap spent the available slots on corridor coverage.
    """
    alts = list(alts)
    universe = ([primary] if primary else []) + alts
    family_keys = _discover_family_keys(universe)
    family_catalog = _build_family_service_catalog(universe, family_keys)
    primary_family = _alt_family_key(primary, family_keys) if primary else None
    if primary is not None:
        primary["_family_seed"] = primary_family
        primary["_family_catalog"] = family_catalog
    alts = _prune_dominated_alts(alts, [primary] if primary else (), family_keys)
    ordered = sorted(alts, key=_alt_quality_rank)
    selected = []
    selected_ids = set()

    def add(a):
        ident = id(a)
        if ident in selected_ids or len(selected) >= cap:
            return False
        selected.append(a); selected_ids.add(ident)
        return True

    families = OrderedDict()
    for a in ordered:
        families.setdefault(_alt_family_key(a, family_keys), []).append(a)

    primary_branch = _alt_branch_key(primary, primary_family) if primary else None

    # Reserve at most ONE slot for useful tail optionality before family breadth exhausts a roomy
    # card. The fastest branch-rich family wins; the primary counts as an already-visible branch.
    # Tiny caps stay breadth-only (in particular cap=1 must show a missing corridor first).
    family_order = [item for item in families.items() if item[0] != primary_family]
    if primary_family in families:
        family_order.append((primary_family, families[primary_family]))
    reserved = None
    if cap >= 3:
        missing_items = [(fam, opts) for fam, opts in family_order if fam != primary_family]
        missing_order = [fam for fam, _opts in missing_items]
        eligible_families = set(missing_order[:max(0, cap - 1)])
        if primary_family is not None:
            eligible_families.add(primary_family)
        displaced = (_family_representative(*missing_items[cap - 1])
                     if len(missing_items) >= cap else None)
        reservation_ceiling = (
            _alt_total(displaced) + ROUTE_FAMILY_NEAR_TIE_MIN
            if displaced is not None else None)
        reserve_candidates = []
        for fam, opts in family_order:
            if fam not in eligible_families:
                continue
            if fam == primary_family:
                visible_branches = {primary_branch} if primary_branch else set()
                family_head = _alt_total(primary)
            else:
                anchor = _family_representative(fam, opts)
                if anchor is None:
                    continue
                visible_branches = {_alt_branch_key(anchor, fam)}
                family_head = _alt_total(anchor)
            by_branch = OrderedDict()
            for option in opts:
                branch = _alt_branch_key(option, fam)
                if branch not in visible_branches:
                    by_branch.setdefault(branch, []).append(option)
            sibling_reps = [_family_representative(fam, branch_opts)
                            for branch_opts in by_branch.values()]
            sibling_reps = [option for option in sibling_reps if option is not None]
            if sibling_reps:
                sibling = min(sibling_reps, key=_alt_quality_rank)
                if (reservation_ceiling is not None
                        and _alt_total(sibling) > reservation_ceiling):
                    continue
                # Prefer optionality wholly visible inside the alternative list. A sibling of the
                # primary is still the fallback when no non-primary family is branch-rich.
                reserve_candidates.append((0 if fam != primary_family else 1,
                                           family_head, _alt_total(sibling), fam, sibling))
        if reserve_candidates:
            reserved = min(reserve_candidates, key=lambda item: item[:4])[4]

    breadth_cap = max(0, cap - (1 if reserved is not None else 0))
    for fam, opts in family_order:
        if fam == primary_family:
            continue                              # the primary already represents this corridor
        rep = _family_representative(fam, opts)
        if rep is not None:
            add(rep)
        if len(selected) >= breadth_cap:
            break
    if reserved is not None:
        add(reserved)

    # One sibling per family per pass prevents a single rich corridor from monopolizing the tail.
    while len(selected) < cap:
        progressed = False
        for fam, opts in family_order:
            represented = {
                _alt_branch_key(a, fam) for a in selected
                if _alt_family_key(a, family_keys) == fam
            }
            if fam == primary_family and primary_branch:
                represented.add(primary_branch)
            by_branch = OrderedDict()
            for a in opts:
                branch = _alt_branch_key(a, fam)
                if branch not in represented:
                    by_branch.setdefault(branch, []).append(a)
            if not by_branch:
                continue
            branch_opts = next(iter(by_branch.values()))
            rep = _family_representative(fam, branch_opts)
            if rep is not None and add(rep):
                progressed = True
            if len(selected) >= cap:
                break
        if not progressed:
            break

    for a in ordered:
        add(a)
        if len(selected) >= cap:
            break

    if complete_selected_families and cap > 0:
        # The cap above is a BREADTH budget, not a reason for an expert view to advertise a family
        # while silently dropping one of its near-window finishes.  Complete only families which
        # already earned visibility (plus the primary's family); never use this phase to admit a
        # new corridor. ``ordered`` is already equivalence-deduped and dominance-pruned.
        selected_families = {
            _alt_family_key(option, family_keys) for option in selected
        }
        if primary_family is not None:
            selected_families.add(primary_family)
        represented = {
            (_alt_family_key(option, family_keys),
             _alt_branch_key(option, _alt_family_key(option, family_keys)))
            for option in selected
        }
        if primary is not None and primary_family is not None and primary_branch is not None:
            represented.add((primary_family, primary_branch))

        def add_completion(option, fam, branch):
            if (fam, branch) in represented:
                return
            # When breadth pressure established a quality ceiling, branch completion obeys that
            # SAME ceiling. If every candidate family already fit, the caller's upstream branch
            # window remains the bound and no extra arbitrary threshold is introduced here.
            if (reservation_ceiling is not None
                    and _alt_total(option) > reservation_ceiling):
                return
            selected.append(option)
            selected_ids.add(id(option))
            represented.add((fam, branch))

        for fam, opts in family_order:
            if fam not in selected_families:
                continue
            family_ceiling = (min(_alt_total(option) for option in opts)
                              + ROUTE_FAMILY_NEAR_TIE_MIN)
            by_branch = OrderedDict()
            for option in opts:
                branch = _alt_branch_key(option, fam)
                if (fam, branch) not in represented:
                    by_branch.setdefault(branch, []).append(option)
            for branch, branch_opts in by_branch.items():
                representative = _family_representative(fam, branch_opts)
                if (representative is not None
                        and _alt_total(representative) <= family_ceiling):
                    add_completion(representative, fam, branch)

    result = sorted(selected, key=_alt_quality_rank)
    if not complete_selected_families:
        result = result[:cap]
    forced = (list(force_include)
              if isinstance(force_include, (list, tuple, set)) else [force_include])
    for forced_option in forced:
        if (forced_option is not None
                and any(forced_option is option for option in alts)
                and not any(forced_option is option for option in result)):
            # Do not evict a useful corridor/branch merely to make room for a recommendation. The
            # pin card has a small explicit cap escape hatch (at most one route per time mode).
            result.append(forced_option)
    result.sort(key=_alt_quality_rank)
    for option in result:
        option["_family_seed"] = _alt_family_key(option, family_keys)
        option["_family_catalog"] = family_catalog
        if primary_family is not None:
            option["_primary_family_seed"] = primary_family
    return result


def _uniq_strings(values):
    out = []
    for value in values:
        value = str(value or "")
        if value and value not in out:
            out.append(value)
    return out


def _family_display_name(opts):
    lines = _uniq_strings(
        _leg_name(legs[0]) for a in opts if (legs := _alt_transit_legs(a))
    )
    if not lines:
        return "Walk option", []
    shown = lines[:3]
    name = " / ".join(shown)
    if len(lines) > len(shown):
        name += f" +{len(lines) - len(shown)}"
    return name, lines


def _family_service_display_name(services):
    lines = _uniq_strings(service.get("name") for service in services or ())
    if not lines:
        return "Walk option", []
    shown = lines[:3]
    name = " / ".join(shown)
    if len(lines) > len(shown):
        name += f" +{len(lines) - len(shown)}"
    return name, lines


def _branch_display_meta(fam_name, branch, opts, tail_sequences=None):
    if str(branch).startswith("walk:"):
        if str(branch) == "walk:only":
            return {"name": "walk only", "kind": "walk", "lines": []}
        rep = _family_representative("", opts) or opts[0]
        names = [_leg_name(leg) for leg in _alt_transit_legs(rep)]
        after = fam_name if fam_name != "Walk option" else (names[0] if names else "transit")
        return {"name": f"walk after {after}", "kind": "walk", "lines": []}

    # A branch can contain several fully traced tails which converge on the same terminal approach.
    # Describe their union; choosing one representative here falsely hid the other usable finishes.
    tails = [tuple(tail) for tail in (tail_sequences or ()) if tail]
    if not tails:
        for option in opts:
            tail = tuple(name for leg in _alt_transit_legs(option)[1:]
                         if (name := _leg_name(leg)))
            if tail and tail not in tails:
                tails.append(tail)
    lines = _uniq_strings(name for tail in tails for name in tail)
    if tails and all(len(tail) == 1 for tail in tails):
        name = f"transfer to {' / '.join(lines)}"
    elif len(tails) == 1:
        tail = tails[0]
        name = (f"via {' > '.join(tail[:-1])} to {tail[-1]}"
                if len(tail) > 1 else f"transfer to {tail[0]}")
    elif tails:
        # Prefer direct finishes before detours and join them as natural alternatives. A slash
        # list such as "L / 44 > K / K" reads like one malformed route rather than three choices.
        labels = sorted((" > ".join(tail) for tail in tails),
                        key=lambda label: (label.count(" > "), label))
        shown = labels[:3]
        if len(shown) == 2:
            choices = " or ".join(shown)
        elif len(shown) == 3:
            choices = f"{shown[0]}, {shown[1]}, or {shown[2]}"
        else:
            choices = shown[0]
        suffix = f" +{len(labels) - len(shown)} more" if len(labels) > len(shown) else ""
        name = f"via {choices}{suffix}"
    else:
        name = "transit finish"
    return {"name": name, "kind": "transit", "lines": lines}


def _merge_family_catalogs(routes):
    merged = OrderedDict()
    for route in routes or ():
        for family, branches in (route.get("_family_catalog") or {}).items():
            family_out = merged.setdefault(family, OrderedDict())
            for branch, branch_meta in branches.items():
                branch_out = family_out.setdefault(
                    branch, {"services": OrderedDict(), "tails": []})
                for key, service in (branch_meta.get("services") or {}).items():
                    branch_out["services"].setdefault(key, dict(service))
                for tail in branch_meta.get("tails") or ():
                    tail = tuple(tail)
                    if tail and tail not in branch_out["tails"]:
                        branch_out["tails"].append(tail)
    return merged


def _visible_service_map(routes):
    visible = OrderedDict()
    for route in routes or ():
        transit = _alt_transit_legs(route)
        if transit:
            service = _leg_service_meta(transit[0])
            visible.setdefault(service["key"], service)
    return visible


def _family_service_rows(family, members, by_branch, catalog):
    """Family union + exact branch subsets, with visible-member fallback."""
    family_catalog = catalog.get(family) or OrderedDict()
    if not family_catalog:
        for branch, branch_opts in by_branch.items():
            family_catalog[branch] = {
                "services": _visible_service_map(branch_opts), "tails": []}

    visible_family = _visible_service_map(members)
    union = OrderedDict()
    branch_rows = {}
    # Only rendered branches contribute to the family union.  A service proven solely on a branch
    # which selection suppressed must not appear without any branch-specific compatibility row.
    for branch, branch_opts in by_branch.items():
        branch_meta = family_catalog.get(branch) or {
            "services": _visible_service_map(branch_opts), "tails": []}
        visible_branch = _visible_service_map(branch_opts)
        rows = []
        for key, service in (branch_meta.get("services") or {}).items():
            base = dict(service)
            base["shown"] = key in visible_branch
            base["branchKeys"] = [branch]
            rows.append(base)
            item = union.setdefault(key, {**service, "shown": False, "branchKeys": []})
            item["shown"] = item["shown"] or key in visible_family
            if branch not in item["branchKeys"]:
                item["branchKeys"].append(branch)
        branch_rows[branch] = rows
    return list(union.values()), branch_rows


def _copy_service_rows(rows):
    return [{**row, "branchKeys": list(row.get("branchKeys") or ())} for row in rows or ()]


def _annotate_route_families(primary, alts):
    """Attach the server-authoritative family/branch rendering model.

    Keys are intentionally opaque. The browser groups on them and renders these derived labels; it
    does not independently rediscover corridors or encode local service knowledge. ``primary`` is
    an alt-shaped private facade, while ``alts`` are the actual response objects.
    """
    routes = ([primary] if primary else []) + list(alts or ())
    if not routes:
        return
    # A family/branch pair is intentionally broad: it cannot identify two equally valid
    # scheduled choices that board the same corridor and finish the same way. Publish the
    # pre-existing exact structural identity alongside the grouping metadata for every option.
    # Exact semantic duplicates are collapsed before this boundary, so this remains unique in a
    # served option set without inventing a positional suffix.
    for route in routes:
        route["choice_key"] = _public_choice_key(route)
    catalog = _merge_family_catalogs(routes)
    family_keys = _discover_family_keys(routes)
    primary_seed = ((primary or {}).get("_family_seed") if primary else None) or next((
        route.get("_primary_family_seed") for route in alts or ()
        if route.get("_primary_family_seed")), None)
    if primary is not None and primary_seed:
        family_keys[id(primary)] = primary_seed
    for route in alts or ():
        if route.get("_family_seed"):
            family_keys[id(route)] = route["_family_seed"]
    families = {}
    for route in routes:
        fam = _alt_family_key(route, family_keys)
        branch = _alt_branch_key(route, fam)
        families.setdefault(fam, []).append((route, branch))
    ordered_families = sorted(families.items(), key=lambda item: (
        min(_alt_total(route) for route, _branch in item[1]), item[0]
    ))
    primary_family = _alt_family_key(primary, family_keys) if primary else None
    alternate_rank = 0
    for _rank, (fam, members) in enumerate(ordered_families):
        opts = [route for route, _branch in members]
        by_branch = OrderedDict()
        for route, branch in members:
            by_branch.setdefault(branch, []).append(route)
        family_services, branch_services = _family_service_rows(
            fam, opts, by_branch, catalog)
        fam_name, fam_lines = _family_service_display_name(family_services)
        branch_keys = list(by_branch)
        tags = []
        if len(family_services) > 1:
            tags.append("shared corridor")
        if len(branch_keys) > 1:
            tags.append("multiple finishes")
        if fam == primary_family:
            sub = "primary boarding corridor"
        else:
            alternate_rank += 1
            sub = "alternate corridor" if alternate_rank == 1 else "backup corridor"
            if alternate_rank > 1:
                tags.append("backup")
        fam_meta = {"key": fam, "name": fam_name, "sub": sub,
                    "lines": fam_lines, "services": _copy_service_rows(family_services),
                    "tags": tags}
        branch_meta = {}
        for branch, branch_opts in by_branch.items():
            preserved_branch = (catalog.get(fam) or {}).get(branch) or {}
            meta = _branch_display_meta(
                fam_name, branch, branch_opts,
                tail_sequences=preserved_branch.get("tails") or ())
            services = _copy_service_rows(branch_services.get(branch) or ())
            branch_meta[branch] = {
                "key": branch, **meta, "services": services,
                "serviceKeys": [service["key"] for service in services],
            }
        for route, branch in members:
            route["family"] = {**fam_meta, "lines": list(fam_meta["lines"]),
                               "services": _copy_service_rows(fam_meta["services"]),
                               "tags": list(fam_meta["tags"])}
            route["branch"] = {**branch_meta[branch],
                               "lines": list(branch_meta[branch]["lines"]),
                               "services": _copy_service_rows(branch_meta[branch]["services"]),
                               "serviceKeys": list(branch_meta[branch]["serviceKeys"])}
    for route in routes:
        for key in list(route):
            if str(key).startswith("_"):
                route.pop(key, None)


def _alt_route_preamble(ci, dlat, dlon, max_rides, speed, cache_key=None):
    """Shared front-half of BOTH ``_itinerary_alts`` variants (arrive-by + depart-after): resolve
    the cell's MC alt chips/bundle, the per-cell geom cache, and the per-line access-stop map. The
    arrive-by/depart-after halves differ ONLY in the per-line trace + output shape (and so are NOT
    merged — that would mix the flat ``{line,min,legs}`` and nested ``{line,best,typical}`` shapes
    that two suites assert byte-for-byte); this de-dups the identical preamble they share.

    Returns one of:
      ("empty", None)               — no MC entry yet (variance not built) -> nothing to cache into;
      ("cached", out)               — the cell's alts are already assembled (return ``out``);
      ("chipless", geom_cache)      — MC entry exists but this cell has no window chips / no
                                      alt_stop map: no chip lines to trace, but the entry's
                                      geom cache IS usable (the depart-after branch expansion
                                      caches its branch-only result there);
      ("ready", (geom_cache, cell_alt_stop, chips)) — proceed to trace each chip line.

    The geom-cache lookup happens BEFORE the chips check: a pinned chipless cell's branch-only
    alts are cached under ``(ci, "branches")`` and must be FOUND on the next click — the old
    order (chips gate first) made every chipless pin re-run the branch enumeration."""
    mc = mc_peek(dlat, dlon, max_rides, speed)
    if mc is None:
        return ("empty", None)                          # variance not built yet -> no alts
    geom_cache = mc["alt_geom"]
    key = ci if cache_key is None else cache_key
    cached = geom_cache.get(key)
    if cached is not None:
        return ("cached", cached)
    chips = mc.get("alt_chips", {}).get(ci)
    bundle = mc.get("alt_bundle")
    if not chips or not bundle:
        return ("chipless", geom_cache)
    alt_stop = bundle.get("alt_stop")
    cell_alt_stop = (alt_stop[ci] if alt_stop and ci < len(alt_stop) else None) or {}
    if not cell_alt_stop:
        return ("chipless", geom_cache)
    return ("ready", (geom_cache, cell_alt_stop, chips))


def _itinerary_alts(ci, dlat, dlon, max_rides, speed, provider=None, pin=False):
    """The drawn alternative routes for cell ``ci``: for each alt chip line the MC overlay surfaced
    for this cell (the dominance-window alts, after the SAME primary-exclusion + cap /variance
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
    speed, walk_scalar = _resolve_walk_speed(speed)
    # A pinned response must enumerate the full server-known alternate universe before its
    # recommendation rank runs.  Keep that richer result under a distinct cache key so a cheap
    # prior hover (which intentionally traces only the bounded chip list) cannot hide options
    # from the pin or poison its recommendation marker.
    status, payload = _alt_route_preamble(
        ci, dlat, dlon, max_rides, speed,
        cache_key=(ci, "pinned") if pin else None)
    if status in ("empty", "chipless"):                 # no MC yet / no chip lines for this cell
        return []
    if status == "cached":
        return payload
    geom_cache, cell_alt_stop, chips = payload          # each chip line -> its access stop
    if pin:
        # ``alt_stop`` is the complete MC-proven per-cell alternate map; ``alt_chips`` is only a
        # bounded hover-display slice.  Insertion order comes from the engine's deterministic
        # near-window ordering, while final order remains server-owned below.
        chips = list(cell_alt_stop)
    entry = raptor_tree(dlat, dlon, max_rides, speed, walk_scalar)         # cache hit (same tree)
    tree = entry["tree"]
    primary = entry.get("geom", {}).get(ci) or {}
    seen = set()
    psig = _route_trace_sig(primary.get("geom"))
    if psig:
        seen.add(psig)
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
        if not _legs_have_transit(it["geom"]):
            continue                                     # walk-only alt: pointless, drop it
        sig = _route_trace_sig(it["geom"])
        if sig and sig in seen:
            continue                                     # exact primary/alt duplicate
        if sig:
            seen.add(sig)
        out.append({"line": line, "min": it["total"], "legs": it["geom"]})
    primary_option = {"line": _route_label(primary.get("geom")) or "primary",
                      "min": primary.get("total", 10 ** 6),
                      "legs": primary.get("geom") or []}
    recommendation_candidates = {}
    if pin:
        family_keys = _discover_family_keys([primary_option] + out)
        recommendation_universe = _prune_dominated_alts(
            out, [primary_option], family_keys)
        # Normal production trees can score every structural alternative against the retained MC
        # scenario before presentation caps.  The hasattr guard preserves sparse unit fixtures
        # which exercise geometry/caching without a complete route-typicals implementation.
        entry_for_recommendation = raptor_tree(dlat, dlon, max_rides, speed, walk_scalar)
        full_typicals = (
            _itinerary_alt_typicals(ci, dlat, dlon, max_rides, speed,
                                    recommendation_universe)
            if hasattr(entry_for_recommendation.get("tree"), "_select") else None)
        if full_typicals is not None:
            primary_typical = full_typicals.get("prim")
            if primary_typical is not None:
                primary_option["real"], primary_option["frag"] = primary_typical
            for option, typical in zip(recommendation_universe,
                                       full_typicals.get("alts", ())):
                if typical is not None:
                    option["real"], option["frag"] = typical
        recommendation_candidates = _recommend_route_choices(
            primary_option, recommendation_universe)
    forced_recommendations = [
        candidate for candidate in recommendation_candidates.values()
        if candidate is not None and candidate is not primary_option
    ]
    out = _select_diverse_alts(
        out, RAPTOR_ALT_CHIP_CAP, primary=primary_option,
        force_include=forced_recommendations)
    for option in out:
        metrics = [metric for metric, candidate in recommendation_candidates.items()
                   if option is candidate]
        if metrics:
            option["_recommendation_metrics"] = metrics
    with _ALT_LOCK:                                      # cache the assembled alts for this cell
        geom_cache[(ci, "pinned") if pin else ci] = out
    return out


def _alts_typ_sig(alts):
    """Value signature of the alts list a per-route typicals/frag result was computed for.
    The typ cache stores results POSITIONALLY aligned to the request's alts list, but that list
    can differ between requests against the SAME cell (e.g. a pin raced the MC build and saw
    ``alts=[]``, or the MC entry was evicted and rebuilt mid-request — ``mc_peek`` is read once
    for the alts and once for the typicals). Keying the cached entry by this signature makes a
    mismatched read recompute instead of serving frags zipped onto the wrong strips forever."""
    sig = []
    for a in alts or ():
        legs = a.get("legs") or _alt_display_legs(a)    # arrive-by flat shape / depart-after nested
        sig.append((str(a.get("line") or ""), int(a.get("via_stop", -1)),
                    _route_trace_sig(legs)))
    return tuple(sig)


def _alt_typicals_preamble(ci, dlat, dlon, max_rides, speed, alts):
    """Shared front-half of BOTH ``_itinerary_alt_typicals`` variants (arrive-by + depart-after):
    the per-pinned-cell ``typ`` cache guard. Returns one of:
      ("empty", None)             — variance not built yet (-> None to the caller);
      ("cached", out)             — this cell's typicals were computed FOR THIS SAME alts list
                                    (signature match; return ``out``);
      ("ready", (typ_cache, mc, sig)) — proceed; cache ``{"sig": sig, "out": result}`` into
                                    ``typ_cache[ci]`` (under _TYP_LOCK).
    The cached entry carries the signature of the alts list it was aligned to; a signature
    mismatch (different alts served this request) falls through to a recompute that overwrites
    the stale entry. The halves still diverge after this (different floors / route_typicals
    flags / output shape), so they are NOT merged — only the identical guard is shared."""
    mc = mc_peek(dlat, dlon, max_rides, speed)
    if mc is None:
        return ("empty", None)                          # variance not built yet -> no typicals
    typ_cache = mc["typ"]
    sig = _alts_typ_sig(alts)
    cached = typ_cache.get(ci)
    if cached is not None and cached.get("sig") == sig:
        return ("cached", cached["out"])
    return ("ready", (typ_cache, mc, sig))


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
    speed, walk_scalar = _resolve_walk_speed(speed)
    status, payload = _alt_typicals_preamble(ci, dlat, dlon, max_rides, speed, alts)
    if status == "empty":
        return None
    if status == "cached":
        return payload
    typ_cache, mc, alts_sig = payload
    entry = raptor_tree(dlat, dlon, max_rides, speed, walk_scalar)   # cache hit (same tree)
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
    # PRIMARY floor = the HEADLINE metric (cells[ci][1]) — the SAME floor _raptor_mc_build passes to
    # montecarlo for this cell (arrive-by [1]==[0]==commute; depart-after [1]==painted p50). Sharing
    # the floor + the seed makes the primary's committed p50 here byte-identical to the served
    # realistic (REAL[id]) under BOTH semantics, so the primary strip matches the headline exactly.
    prim_best = cells.get(_RAPTOR.cell_ids[ci], [None, None])[1]
    stops = [int(s_star) if (not is_walk and s_star >= 0) else -1]
    floors = [prim_best]
    for a in alts:
        stops.append(int(cell_alt_stop.get(a["line"], -1)))
        floors.append(a.get("min"))
    egress_g, egress_w, _purewalk = raptor_egress_purewalk(dlat, dlon)
    # SAME per-workplace seed as _raptor_mc_build's /variance MC: route_typicals draws the same-shaped
    # (nR, T) delta arrays, so the PRIMARY route's committed p50 here is byte-identical to that cell's
    # served `realistic` (REAL[id]) -> the primary compare strip matches the headline exactly.
    seed = mc_seed(dlat, dlon, max_rides, speed)
    scenario = _mc_scenario_for(mc.get("_scenario_token"),
                                coarse_key(dlat, dlon, max_rides, speed))
    pairs = _RAPTOR.route_typicals(tree, ci, stops, egress_g, egress_w,
                                   perfect_route_mins=floors, seed=seed,
                                   walk_scalar=walk_scalar, max_rounds=int(max_rides),
                                   scenario=scenario)
    out = {"prim": pairs[0] if pairs else None, "alts": pairs[1:] if len(pairs) > 1 else []}
    with _TYP_LOCK:                                      # cache the typicals for this pinned cell,
        typ_cache[ci] = {"sig": alts_sig, "out": out}    # keyed by the alts list they align to
    return out


# ---- Depart-after /itinerary: BOTH the best-case (p5) and typical (p50) journeys -------
# Under depart-after the BEST-CASE and TYPICAL are DIFFERENT percentiles of the [DEP, DEP+WINDOW]
# departure window, so each is a DIFFERENT drawn journey (can be a different route). The frontend's
# metric toggle must switch between them LOCALLY (no re-fetch), so /itinerary ships BOTH journeys in
# one response. Each journey's total is the cell's painted percentile EXACTLY (hover==map for BOTH):
# the p50 journey traced from the cached p50 DepartAfterJourneyTree (== cells[c][1]), the p5 journey
# from the cached p5 tree (== cells[c][0]). See the JSON contract docstring in _itinerary_departafter.
def _itinerary_alts_departafter(ci, entry, dlat, dlon, max_rides, speed, prov50, prov5,
                                expand_branches=False, perf=None):
    """The drawn alternative routes for cell ``ci`` under DEPART-AFTER, each carrying BOTH its
    best-case (p5) and typical (p50) journey (total + geom legs) so the compare card can switch on
    the metric toggle without re-fetching. For each alt chip line the MC overlay surfaced for this
    cell (the dominance-window alts, after the SAME primary-exclusion + cap /variance applies),
    trace that line's access stop (the bundle's per-cell ``alt_stop`` map) on the p50 tree at the
    alt's OWN per-stop p50 + p5 percentiles (``itinerary_via_stop(..., percentile=)``) — the per-stop
    percentile guarantees the alt's best-case <= its typical. Returns
    ``[{line, best:{total,legs}, typical:{total,legs}}]`` (closest-first). On a plain HOVER this is
    [] until the MC build has run (we never trigger it here — the chips come from the MC); a PINNED
    planned cell (``expand_branches``) serves its deterministic branch alternatives even before the
    MC exists (cached on the tree entry's ``branch_geom``). ``prov50``/``prov5`` are the primary
    route's geom providers reused so the per-cell access walk-tree (one Dijkstra each) is shared
    (p50 geom on prov50, p5 geom on prov5)."""
    branch_cache = bool(expand_branches and entry.get("planned"))
    status, payload = _alt_route_preamble(
        ci, dlat, dlon, max_rides, speed,
        cache_key=(ci, "branches") if branch_cache else None)
    if status == "empty":
        # No MC entry yet (/variance hasn't landed). A plain hover has nothing to draw (the alt
        # CHIPS come from the MC), but a PINNED cell must still serve its deterministic branch
        # alternatives — they are traced from the cached tree, not the MC (the product spec:
        # pin-without-variance shows structurally discovered sibling branches). There is no
        # mc["alt_geom"] to
        # cache into yet, so the pre-variance result is cached on the TREE entry instead
        # ("branch_geom", stored at the tail below); once the MC lands, the preamble's
        # (ci, "branches") path takes over.
        if not branch_cache:
            return []
        pre_cache = entry.get("branch_geom")
        if pre_cache is not None:
            cached = pre_cache.get(ci)
            if cached is not None:
                return cached
        geom_cache, cell_alt_stop, chips = None, {}, []
    elif status == "cached":
        return payload
    elif status == "ready":
        geom_cache, cell_alt_stop, chips = payload
    else:                                   # "chipless": MC built, no window chips for this cell;
        geom_cache, cell_alt_stop, chips = payload, {}, []   # geom cache still usable (branches)
    if not chips and not branch_cache:
        return []                           # plain hover on a chipless cell: nothing to draw
    tree50 = entry["tree"]                               # owns arrivalW + the per-T* tree cache
    primary50 = entry.get("geom", {}).get(ci) or {}
    primary5 = (entry.get("geom5") or {}).get(ci) or primary50
    seen = set()
    psig = (_route_trace_sig((primary5 or {}).get("geom")),
            _route_trace_sig((primary50 or {}).get("geom")))
    if psig[0] or psig[1]:
        seen.add(psig)
    if prov50 is None:
        prov50 = _JourneyGeomProvider(dlat, dlon)
    if prov5 is None:
        prov5 = _JourneyGeomProvider(dlat, dlon)
    out = []

    def _add(line, it50, it5, source="window", via_stop=None):
        if it50 is None and it5 is None:
            return
        # Walk-only alt: BOTH percentiles' journeys degenerate to pure walking (no ride) -> pointless,
        # drop it (a line-labeled slower walk path; see _legs_have_transit). Keep it only if at least
        # one percentile traces a real transit leg (an alt where p5 rides but p50 walks still informs).
        has5 = it5 is not None and _legs_have_transit(it5["geom"])
        has50 = it50 is not None and _legs_have_transit(it50["geom"])
        if not (has5 or has50):
            return
        sig5 = _route_trace_sig((it5 or it50)["geom"])
        sig50 = _route_trace_sig((it50 or it5)["geom"])
        sig = (sig5, sig50)
        if sig in seen:
            return                                      # exact primary/alt duplicate
        seen.add(sig)
        traced_label = (_route_label((it50 or it5)["geom"]) or
                        _route_label((it5 or it50)["geom"]) or line)
        d = {"line": traced_label, "source": source}
        if traced_label != line:
            d["chip_line"] = line
        if via_stop is not None:
            d["via_stop"] = int(via_stop)
        if it50 is not None:
            d["typical"] = {"total": it50["total"], "legs": it50["geom"]}
        if it5 is not None:
            d["best"] = {"total": it5["total"], "legs": it5["geom"]}
        out.append(d)

    for line in chips:                                   # preserve the chip (closest-first) order
        s = cell_alt_stop.get(line)
        if s is None:
            continue
        if entry.get("planned"):
            it50 = tree50.itinerary_via_stop(ci, int(s), geom_provider=prov50,
                                             percentile="planned")
            it5 = it50
        else:
            # Each alt journey is anchored on the alt's OWN per-stop percentile (p5 best-case, p50
            # typical) so alt p5 <= alt p50 holds PER alt.
            it50 = tree50.itinerary_via_stop(ci, int(s), geom_provider=prov50, percentile=50)
            it5 = tree50.itinerary_via_stop(ci, int(s), geom_provider=prov5, percentile=5)
        _add(line, it50, it5, via_stop=s)

    if branch_cache:
        t_branch = time.perf_counter() if perf is not None else None
        branch_candidates = 0
        base = int(primary50.get("total", -1))
        # Filter only on a coverage-safe necessary predicate over EVERY accessible stop. Public
        # chip labels/caps are intentionally not seeds: two corridors can share a label, and an
        # in-window family can rank below the UI chip cap while still belonging in the pinned card.
        branch_stops = None
        try:
            branch_stops = tree50.planned_branch_access_stops(
                ci, base, RAPTOR_BRANCH_WINDOW_MIN)
        except (AttributeError, IndexError, TypeError, ValueError):
            pass
        # Enumeration runs GEOMETRY-FREE (geom_provider=None): it traces every morning-window
        # anchor at every access stop and dedupes to structural ride shapes.  The selector needs
        # exact transit stop polylines for corridor/direction identity, but not street paths, so
        # production branches use a cheap structural proxy here.  Only the few selected survivors
        # are hydrated below.  Sparse fixtures which cannot construct a proxy retain the legacy
        # hydrate-first fallback.
        branch_kwargs = {"geom_provider": None}
        if branch_stops is not None:
            branch_kwargs["access_stops"] = branch_stops
        for b in tree50.planned_branch_itineraries(
                ci, base, RAPTOR_BRANCH_WINDOW_MIN, **branch_kwargs):
            branch_candidates += 1
            d = _planned_branch_proxy_option(b)
            if d is None:
                it = tree50._format_planned_raw(
                    ci, b["stop"], b["raw"], b["home"], b["jt"],
                    geom_provider=prov50, planned_total=b["total"],
                    planned_target_sec=b.get("target_sec", b.get("metric_sec")))
                if it is None or not _legs_have_transit(it.get("geom")):
                    continue
                d = {"line": b["line"], "source": "branch", "_branch": b,
                     # Branch alts ALWAYS know their access stop (every enumeration candidate
                     # carries "stop"), so set via_stop unconditionally — the old by-label frag
                     # fallback could never match a relabeled multi-leg branch alt.
                     "via_stop": int(b["stop"]),
                     "typical": {"total": it["total"], "legs": it["geom"]},
                    "best": {"total": it["total"], "legs": it["geom"]}}
            out.append(d)
        _perf_add(perf, "pin.branch_enumeration_ms", t_branch)
        if perf is not None:
            perf["pin.branch_candidates"] = branch_candidates

    def hydrate_proxy(a):
        """Fill exact walk geometry for one proxy, returning False for an invalidated route."""
        if not a.pop("_needs_hydration", False):
            return True
        b = a.get("_branch") or {}
        it = tree50._format_planned_raw(
            ci, b.get("stop"), b.get("raw"), b.get("home"), b.get("jt"),
            geom_provider=prov50, planned_total=b.get("total"),
            planned_target_sec=b.get("target_sec", b.get("metric_sec")))
        if it is None or not _legs_have_transit(it.get("geom")):
            return False
        a["typical"] = {"total": it["total"], "legs": it["geom"]}
        a["best"] = {"total": it["total"], "legs": it["geom"]}
        return True

    # Proxy legs carry exact transit polylines but intentionally empty walk geometry.  Their
    # selection-only tie-break is the branch's readable scheduled structure, not its incidental
    # street polyline.  Consequently no proxy is hydrated here: hydration begins only after
    # family/dominance/card selection has reduced the candidate set to user-visible routes.

    t_select = time.perf_counter() if perf is not None else None
    best_by_label = {}
    for a in out:
        # The alt's signature pair mirrors psig's slot order (best/p5 first, typical/p50
        # second). An alt carrying only ONE journey falls back to that journey in BOTH slots,
        # so it is dropped iff its single route matches the primary's best AND typical.
        asig = (_route_trace_sig((a.get("best") or a.get("typical") or {}).get("legs")),
                _route_trace_sig((a.get("typical") or a.get("best") or {}).get("legs")))
        if (psig[0] or psig[1]) and asig == psig:
            continue
        dedupe_key = _alt_dedupe_key(a)
        cur = best_by_label.get(dedupe_key)
        if cur is None or _alt_quality_rank(a) < _alt_quality_rank(cur):
            best_by_label[dedupe_key] = a
    primary_option = {
        "line": _route_label((primary50 or {}).get("geom")) or "primary",
        "typical": {"total": primary50.get("total", 10 ** 6),
                    "legs": primary50.get("geom") or []},
        "best": {"total": primary5.get("total", primary50.get("total", 10 ** 6)),
                 "legs": primary5.get("geom") or primary50.get("geom") or []},
    }
    if branch_cache:
        _assert_primary_minimum(primary_option, best_by_label.values())
    # The pinned recommendation is decided from the entire structural candidate universe, before
    # presentation breadth/caps.  This is deliberately separate from the card selector: the
    # canonical map itinerary can lose to a same-minute route with less physical walking, and a
    # useful recommendation must be retained even if its corridor was not otherwise selected.
    #
    # Planned branch proxies retain their raw walk durations, so this pruning/ranking phase does
    # not need to hydrate a street polyline for every candidate.  When MC data exists, score
    # fragility across that same complete universe before comparison; without it the absent
    # bad-day field is an honest neutral tie rather than a fabricated estimate.
    recommendation_universe = list(best_by_label.values())
    recommendation_candidates = {}
    if branch_cache:
        recommendation_families = _discover_family_keys(
            [primary_option] + recommendation_universe)
        recommendation_universe = _prune_dominated_alts(
            recommendation_universe, [primary_option], recommendation_families)
        # Sparse/offline journey fixtures expose the structural branch API but intentionally omit
        # the MC tree's per-stop selector. They still receive the deterministic no-frag fallback;
        # production trees always provide it, so the normal pin includes bad-day impact.
        full_fragility = (
            _itinerary_alt_typicals_departafter(
                ci, entry, dlat, dlon, max_rides, speed, recommendation_universe, perf=perf)
            if hasattr(tree50, "_select") else None)
        if full_fragility is not None:
            if full_fragility.get("prim_frag") is not None:
                primary_option["frag"] = full_fragility["prim_frag"]
            for option, frag in zip(recommendation_universe,
                                    full_fragility.get("alt_frags", ())):
                if frag is not None:
                    option["frag"] = frag
        recommendation_candidates = _recommend_route_choices(
            primary_option, recommendation_universe)
    forced_recommendations = [
        candidate for candidate in recommendation_candidates.values()
        if candidate is not None and candidate is not primary_option
    ]
    out = _select_diverse_alts(
        list(best_by_label.values()), RAPTOR_ALT_CHIP_CAP, primary=primary_option,
        # The recommendation-first UI keeps only a few practical rows visible. Its expanded expert
        # disclosure, unlike hover chips, should be complete for every family the breadth selector
        # admitted. Branch enumeration/dominance already bounds the candidate universe.
        complete_selected_families=branch_cache,
        force_include=forced_recommendations)
    _perf_add(perf, "pin.selection_ms", t_select)
    if perf is not None:
        perf["pin.selection_candidates"] = len(best_by_label)
        perf["pin.selection_survivors"] = len(out)
    t_hydrate = time.perf_counter() if perf is not None else None
    final = []
    final_seen = set()
    transfer_pairs = []
    for a in out:
        if not a.get("_needs_hydration"):
            continue
        for leg in ((a.get("_branch") or {}).get("raw") or ()):
            if leg[0] == "walk_t" and len(leg) >= 4:
                transfer_pairs.append((int(leg[2]), int(leg[3])))
    if transfer_pairs:
        prov50.prefetch_transfers(transfer_pairs)
    for a in out:
        # Hydrate only the selected proxy branches.  Structural selection has already reduced the
        # production candidate set from hundreds to the small expert-card result, so transfer walk
        # path extraction is paid only for routes the user can actually inspect.
        if not hydrate_proxy(a):
            continue
        # _branch is the selection-time raw-leg handle (raw tuples + a JourneyTree ref), not
        # JSON-serializable, so it never reaches the cache/response.
        a.pop("_branch", None)
        sig = (_route_trace_sig((a.get("best") or a.get("typical") or {}).get("legs")),
               _route_trace_sig((a.get("typical") or a.get("best") or {}).get("legs")))
        if sig in final_seen or ((psig[0] or psig[1]) and sig == psig):
            continue
        final_seen.add(sig)
        recommendation_metrics = [
            metric for metric, candidate in recommendation_candidates.items()
            if a is candidate
        ]
        if recommendation_metrics:
            # Internal transit from structural selection to the response assembler below.  The
            # metadata annotator strips underscore fields before JSON serialization.
            a["_recommendation_metrics"] = recommendation_metrics
        final.append(a)
    out = final
    _perf_add(perf, "pin.hydration_geometry_ms", t_hydrate)
    if perf is not None:
        perf["pin.hydrated_routes"] = len(final)
    if geom_cache is not None:
        with _ALT_LOCK:                                  # cache the assembled alts for this cell
            geom_cache[(ci, "branches") if branch_cache else ci] = out
    elif branch_cache:
        # Pre-variance pin (mc None): cache the branch-only alts on the tree entry so a repeat
        # pin doesn't re-enumerate. Same locked-mutation rule (shared value of a cached entry).
        pre_cache = entry.get("branch_geom")
        if pre_cache is not None:
            with _ALT_LOCK:
                pre_cache[ci] = out
    return out


def _itinerary_alt_typicals_departafter(ci, entry, dlat, dlon, max_rides, speed, alts,
                                        perf=None):
    """Per-ROUTE committed-plan FRAGILITY (p90-p50) for a PINNED depart-after cell ``ci``, one per
    route: the PRIMARY (the cell's p50 journey) then each alt in ``alts`` (closest-first). Every
    route is scored by the SAME committed Monte-Carlo (``RaptorEngine.route_typicals``) on the p50
    DepartAfterJourneyTree, floored at that route's p50 best-case, so its p90 tail (-> frag) is
    measured against the SAME baseline the headline shows. We surface ONLY the fragility (p90-p50)
    per route — the TYPICAL displayed per route is the bare p50 journey total (``best``/``typical``
    are already on each route from ``_itinerary_alts_departafter``), NOT the MC committed value
    (the depart-after typical headline is the bare p50, not the committed number).

    Returns {"prim_frag": int|None, "alt_frags": [int|None, ...]} aligned to ``alts``. Lazy +
    cached per pinned cell in the MC entry (``typ``) under _TYP_LOCK; ONLY on /itinerary?pin=1."""
    speed, walk_scalar = _resolve_walk_speed(speed)
    t_phase = time.perf_counter() if perf is not None else None
    status, payload = _alt_typicals_preamble(ci, dlat, dlon, max_rides, speed, alts)
    if status == "empty":
        return None
    if status == "cached":
        return payload
    typ_cache, _mc, alts_sig = payload
    tree = entry["tree"]                                 # the p50 tree (committed MC base)
    s_star, _aw, _lh, is_walk = tree._select(ci)
    cells = entry["cells"]
    # PRIMARY floor = the painted p50 (cells[ci][1]) — the SAME floor _raptor_mc_build passes to
    # montecarlo for this cell. Each alt floors at its OWN p50 journey best-case (the alt's "typical"
    # total). Sharing the floor + the per-workplace seed makes the per-route p90-p50 here the SAME
    # tail the served /variance frag is measured from.
    prim_p50 = cells.get(_RAPTOR.cell_ids[ci], [None, None])[1]
    stops = [int(s_star) if (not is_walk and s_star >= 0) else -1]
    floors = [prim_p50]
    for a in alts:
        # Every depart-after alt is constructed WITH via_stop (window alts from the chip's
        # access stop, branch alts unconditionally from the enumeration candidate). -1 is a
        # defensive default only (an unexpected shape yields a kind-0 row -> None frag); the
        # old by-``line`` bundle fallback could never match a relabeled branch alt anyway.
        stops.append(int(a.get("via_stop", -1)))
        floors.append((a.get("typical") or {}).get("total"))
    egress_g, egress_w, _purewalk = raptor_egress_purewalk(dlat, dlon)
    seed = mc_seed(dlat, dlon, max_rides, speed)
    scenario = _mc_scenario_for(_mc.get("_scenario_token"),
                                coarse_key(dlat, dlon, max_rides, speed))
    pairs = _RAPTOR.route_typicals(tree, ci, stops, egress_g, egress_w,
                                   perfect_route_mins=floors, seed=seed,
                                   walk_scalar=walk_scalar, max_rounds=int(max_rides),
                                   return_committed_p90=True, scenario=scenario)
    _perf_add(perf, "pin.frag_replay_ms", t_phase)
    if perf is not None:
        perf["pin.frag_routes"] = len(stops)
    # Per-route frag = committed_p90 - that route's displayed served p50 (its `floors` entry), so each
    # strip reconciles `displayed_typical + frag == committed_p90` — the SAME contract the served
    # per-cell /variance frag uses. We must NOT use pairs[k][1] (= committed_p90 - committed_p50): the
    # displayed per-route typical is the bare served p50, not committed p50, so committed-p50-based
    # frag would understate the bad day exactly where committed_p50 > served_p50 (the bug this fixes).
    # pairs[k] = (committed_p50, _frag_unused, committed_p90); None for an unreachable-via-its-stop
    # route. floors[k] is None for a walk-only primary -> no frag (the strip shows no chip).
    def _df(k):
        p = pairs[k] if (pairs and k < len(pairs)) else None
        if p is None or floors[k] is None:
            return None
        return max(0, int(p[2]) - int(floors[k]))
    prim_frag = _df(0)
    alt_frags = [_df(k) for k in range(1, len(alts) + 1)]
    out = {"prim_frag": prim_frag, "alt_frags": alt_frags}
    with _TYP_LOCK:                                      # cache the fragility for this pinned cell,
        typ_cache[ci] = {"sig": alts_sig, "out": out}    # keyed by the alts list it aligns to
    return out


def _publish_choice_recommendation(res, primary_route, pin, recommended_route=None,
                                   recommended_routes=None):
    """Publish distinct canonical-map and pinned-practical selection identities.

    ``choice_key`` remains the backwards-compatible primary/map choice.  ``map_choice_key``
    names that role explicitly for new clients.  On a pinned response, the structural branch
    selector may have marked one force-retained alternative as the complete-universe practical
    recommendation; otherwise the canonical map route won that comparison.  Never make the
    browser infer either identity from response order.
    """
    map_choice_key = primary_route.get("choice_key") or res.get("choice_key")
    if map_choice_key:
        res["map_choice_key"] = map_choice_key
    if not pin:
        return
    recommendations = dict(recommended_routes or {})
    if recommended_route is not None:
        recommendations.setdefault("r", recommended_route)
    for metric in ("r", "b"):
        recommendations.setdefault(
            metric, _recommend_route_choice(primary_route, res.get("alts") or (), metric))
    recommended_choice_keys = {}
    for metric, recommendation in recommendations.items():
        key = (recommendation or {}).get("choice_key") or map_choice_key
        if key:
            recommended_choice_keys[metric] = key
    if recommended_choice_keys:
        res["recommended_choice_keys"] = recommended_choice_keys
    # Backwards-compatible default: realistic/scheduled remains the initial time mode.
    recommended_choice_key = recommended_choice_keys.get("r") or map_choice_key
    if recommended_choice_key:
        res["recommended_choice_key"] = recommended_choice_key


def itinerary_departafter(cid, olat, olon, dlat, dlon, max_rides, speed, walk_scalar, pin=False,
                          perf=None):
    """Assemble the DEPART-AFTER /itinerary response.

    In the served planned mode, ``best`` and ``typical`` both mirror the same scheduled journey so
    older clients keep their shape while the metric itself is [scheduled, scheduled]. JSON shape:

      {
        # the TYPICAL (p50) journey on the response root (back-compat: root total == cells[c][1]):
        "total": int,  "xfers": int,  "legs": [...],  "geom": [...],
        # the BEST-CASE (p5) journey (total == cells[c][0]); never null on a reachable cell:
        "best":    {"total": int, "xfers": int, "legs": [...], "geom": [...]},
        # the TYPICAL journey ALSO mirrored under "typical" for symmetry with "best":
        "typical": {"total": int, "xfers": int, "legs": [...], "geom": [...]},
        # alternative routes, each carrying BOTH percentiles' journeys (closest-first):
        "alts": [{"line": str,
                  "best":    {"total": int, "legs": [...]},
                  "typical": {"total": int, "legs": [...]}
                  [, "frag": int]   # only on ?pin=1
                 }, ...],
        # ONLY on ?pin=1: the PRIMARY route's per-route fragility (p90-p50), MC-committed:
        "frag": int
      }

    hover==map holds for BOTH percentiles by construction: root/typical.total == cells[c][1] and
    best.total == cells[c][0], because both journeys are traced from the SAME arrivalW the served
    /compute map paints with (the p50 + p5 DepartAfterJourneyTrees). The MC supplies ONLY the
    per-cell fragility (on pin) + the alt-line set; the displayed TYPICAL is the bare p50, never the
    committed MC value. ``error`` key on an unreachable cell."""
    t_assembly = time.perf_counter() if perf is not None else None
    ci = _RAPTOR.cell_index.get(cid) if cid is not None else None
    if ci is None:
        ci = nearest_raptor_cell(olat, olon)
    if ci is None:
        return {"error": "no route"}
    entry = raptor_tree(dlat, dlon, max_rides, speed, walk_scalar)
    # TYPICAL (p50) journey — cached per cell in entry["geom"].
    t_primary = time.perf_counter() if perf is not None else None
    typ_j = entry["geom"].get(ci)
    prov50 = None
    if typ_j is None:
        prov50 = _JourneyGeomProvider(dlat, dlon)
        typ_j = entry["tree"].itinerary(ci, geom_provider=prov50)
        if typ_j is not None:
            with _GEOM_LOCK:
                entry["geom"][ci] = typ_j
    if typ_j is None:
        return {"error": "no route"}
    # BEST-CASE slot: under planned depart-after it is the same scheduled journey; legacy
    # percentile mode keeps a separate p5 tree.
    prov5 = None
    if entry.get("planned"):
        best_j = typ_j
    else:
        best_j = entry["geom5"].get(ci)
        if best_j is None:
            prov5 = _JourneyGeomProvider(dlat, dlon)
            best_j = entry["tree5"].itinerary(ci, geom_provider=prov5)
            if best_j is not None:
                with _GEOM_LOCK:
                    entry["geom5"][ci] = best_j
    _perf_add(perf, "pin.primary_geometry_ms", t_primary)
    # Root = the TYPICAL journey (back-compat default), plus explicit best/typical objects.
    res = dict(typ_j)
    res["typical"] = {k: typ_j[k] for k in ("total", "xfers", "legs", "geom") if k in typ_j}
    if best_j is not None:
        res["best"] = {k: best_j[k] for k in ("total", "xfers", "legs", "geom") if k in best_j}
    # Drawn alternative routes, each with BOTH percentiles' journeys. Empty until /variance built
    # the MC for this workplace (we never trigger it here, so the hover stays cheap).
    # COPIES of the cached alt dicts: the pin path annotates per-route ``frag`` below, and
    # mutating the shared cached objects would (a) race concurrent requests iterating them
    # (dict-changed-during-iteration 500s) and (b) leak pin-only fields onto plain hovers.
    res["alts"] = [dict(a) for a in _itinerary_alts_departafter(
        ci, entry, dlat, dlon, max_rides, speed, prov50, prov5, expand_branches=pin,
        perf=perf)]
    primary_route = {
        "line": _route_label(typ_j.get("geom")) or "primary",
        "typical": {"total": typ_j["total"], "legs": typ_j.get("geom") or []},
        "best": {"total": (best_j or typ_j)["total"],
                 "legs": (best_j or typ_j).get("geom") or []},
    }
    # Keep this object reference while annotation removes private selector fields.  A marked alt
    # was chosen from the full pin universe and force-retained through the display cap.
    recommended_alts = {
        metric: alt
        for alt in res["alts"]
        for metric in alt.get("_recommendation_metrics", ())
    }
    t_annotation = time.perf_counter() if perf is not None else None
    _annotate_route_families(primary_route, res["alts"])
    res["family"] = primary_route["family"]
    res["branch"] = primary_route["branch"]
    res["choice_key"] = primary_route["choice_key"]
    _perf_add(perf, "pin.annotation_ms", t_annotation)
    # PINNED cells (?pin=1): per-route fragility (p90-p50), MC-committed, on the primary + each alt.
    if pin and res.get("alts") is not None:
        fr = _itinerary_alt_typicals_departafter(ci, entry, dlat, dlon, max_rides, speed,
                                                 res["alts"], perf=perf)
        if fr is not None:
            if fr.get("prim_frag") is not None:
                res["frag"] = fr["prim_frag"]
                primary_route["frag"] = fr["prim_frag"]
            for a, f in zip(res["alts"], fr.get("alt_frags", [])):
                if f is not None:
                    a["frag"] = f
    _publish_choice_recommendation(res, primary_route, pin,
                                   recommended_routes=recommended_alts)
    _perf_add(perf, "pin.assembly_ms", t_assembly)
    if perf is not None:
        perf["pin.options"] = 1 + len(res["alts"])
    return res


def itinerary_arriveby(cid, olat, olon, dlat, dlon, max_rides, speed, walk_scalar, pin=False,
                       perf=None):
    """Assemble the ARRIVE-BY /itinerary response for cell ``cid`` (or the nearest grid cell).
    Geometry from the SAME traced journey the breakdown shows (the hover==map invariant extends
    to the drawn route — never recomputed via another path). Returns the breakdown dict with
    ``geom``, ``alts`` (drawn alternatives, empty until /variance built the MC), and — on
    ?pin=1 — per-route committed-plan typical + fragility (``real``/``frag``)."""
    t_assembly = time.perf_counter() if perf is not None else None
    ci = _RAPTOR.cell_index.get(cid) if cid is not None else None
    if ci is None:
        ci = nearest_raptor_cell(olat, olon)
    res = None
    provider = None
    if ci is not None:
        entry = raptor_tree(dlat, dlon, max_rides, speed, walk_scalar)
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
    # COPIES of the cached alt dicts (the pin path annotates real/frag below; mutating the
    # shared cached objects raced concurrent requests and leaked pin-only fields onto hovers).
    res["alts"] = ([dict(a) for a in
                    _itinerary_alts(ci, dlat, dlon, max_rides, speed, provider=provider,
                                    pin=pin)]
                   if ci is not None and "error" not in res else [])
    if "error" not in res:
        t_annotation = time.perf_counter() if perf is not None else None
        primary_route = {"line": _route_label(res.get("geom")) or "primary",
                         "min": res.get("total", 10 ** 6), "legs": res.get("geom") or []}
        recommended_alts = {
            metric: alt
            for alt in res["alts"]
            for metric in alt.get("_recommendation_metrics", ())
        }
        _annotate_route_families(primary_route, res["alts"])
        res["family"] = primary_route["family"]
        res["branch"] = primary_route["branch"]
        res["choice_key"] = primary_route["choice_key"]
        _perf_add(perf, "pin.annotation_ms", t_annotation)
    # PINNED cells (?pin=1) also get a per-ROUTE committed-plan typical + fragility so the
    # compare card can show every strip on the SAME metric as the selector (the primary's typical
    # was already best-case-vs-typical; the alts now carry their own). Lazy + cached per pinned
    # cell, scored by the same committed MC; NEVER computed on a plain hover (the gate below).
    if (pin and ci is not None and "error" not in res
            and res.get("alts") is not None):
        t_frag = time.perf_counter() if perf is not None else None
        typ = _itinerary_alt_typicals(ci, dlat, dlon, max_rides, speed, res["alts"])
        _perf_add(perf, "pin.frag_replay_ms", t_frag)
        if typ is not None:
            prim = typ.get("prim")
            if prim is not None:
                res["real"], res["frag"] = prim[0], prim[1]
                primary_route["frag"] = prim[1]
            for a, p in zip(res["alts"], typ.get("alts", [])):
                if p is not None:
                    a["real"], a["frag"] = p[0], p[1]
    if "error" not in res:
        _publish_choice_recommendation(res, primary_route, pin,
                                       recommended_routes=recommended_alts)
    _perf_add(perf, "pin.assembly_ms", t_assembly)
    if perf is not None:
        perf["pin.options"] = 1 + len(res.get("alts") or ())
    return res
