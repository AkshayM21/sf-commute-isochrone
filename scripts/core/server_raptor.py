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
import math
import threading
from collections import OrderedDict

from . import config

# ---- Shared flags (parsed EARLY: they decide whether the in-process JVM starts at all) ----
# THE DEFAULT IS THE JVM-FREE STACK: USE_RAPTOR + USE_WALK_GRAPH + DEPART-AFTER p5/p50 + the
# service-noise overlay. R5/the JVM is no longer loaded by default. The default SEMANTIC flipped
# arrive-by -> depart-after on 2026-06-17: depart-after is R5-validated (MAE 0.75) and TRUE-ZERO
# walk-speed monotone (no single-departure "latest run" jiggle), so the served map is best-case
# (p5) / typical (p50) over the [08:35, 09:05] window. Opt INTO arrive-by-09:00 (the best-case
# perfect-timing read) with RAPTOR_SEMANTIC=arriveby. Opt back into the legacy R5 path with
# USE_RAPTOR=0 (and/or USE_WALK_GRAPH=0 for the R5 walk matrix).
USE_RAPTOR = os.environ.get("USE_RAPTOR", "1").lower() in ("1", "true", "yes", "on")
RAPTOR_SEMANTIC = os.environ.get("RAPTOR_SEMANTIC", "departafter").lower()
RAPTOR_MC = os.environ.get("RAPTOR_MC", "1").lower() in ("1", "true", "yes", "on")
USE_WALK_GRAPH = os.environ.get("USE_WALK_GRAPH", "1").lower() in ("1", "true", "yes", "on")

DEFAULT_MAX_RIDES = 8                             # R5's ride cap (rides = transfers + 1)

# Walk-speed toggle (RAPTOR only): scalar = 4.8 / pace. The engine multiplies every WALK
# reference-second (access/egress/pure-walk, baked @4.8) by it. Default medium (the user is a
# fast walker but most aren't); the access table / egress stay reference seconds + cached once.
# The presets live in core.config (beside WALK_KMH, their reference speed).
WALK_SPEEDS = config.WALK_SPEEDS
DEFAULT_SPEED = config.DEFAULT_SPEED


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
    USE_WALK_GRAPH=0 fallback) to this module. Called ONCE by server.py after it builds the
    grid / loads the engine. After this the builders read these module globals."""
    g = globals()
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
_RAPTOR_EGRESS_CACHE = BoundedLRU(_EGRESS_CACHE_MAX)  # coarse_key -> (egress_g, egress_w, purewalk)


# ---- cache keys ------------------------------------------------------------------------
def coarse_key(lat, lon, max_rides=DEFAULT_MAX_RIDES, speed=None):
    """~110m bucket for the heavy-result caches (_EXACT/_ATTR/_RAPTOR_TREE/_RAPTOR_MC). Like
    dest_key, the transfer cap AND the walk speed are part of the key so a capped/slow result
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


# ---- RAPTOR grid travel-times (flag-gated) --------------------------------------------
def raptor_egress_purewalk(lat, lon):
    """Per-workplace inputs for the RAPTOR engine, via ONE-origin R5 WALK matrices:
      egress_g/egress_w — W->stop walk seconds (gid-keyed), capped at the access cap;
      purewalk          — W->cell walk seconds (cell order), capped at MAX_MIN.
    The only R5 use on the RAPTOR map path (one light walk tree, not the heavy per-cell pass).
    Cached per ~110m workplace bucket."""
    import numpy as _np
    ckey = coarse_key(lat, lon)
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
# so this (like the MC cache below) stays an order of magnitude smaller than the result LRUs.
# copy_mode='shallow': callers get dict(entry) so they can't reassign the cached entry's keys
# (tree/cells/dom values themselves are shared and treated as immutable).
_TREE_CACHE_MAX = 8
_RAPTOR_TREE_CACHE = BoundedLRU(_TREE_CACHE_MAX, copy_mode="shallow")
# (coarse_key incl. rides+speed) -> {tree, cells, dom, geom}

# ---- Hover route GEOMETRY (the drawn journey) ------------------------------------------
# Walk-leg paths come from PathTree predecessor chains over the SAME walk graph the times
# use. The expensive piece is the workplace-rooted REVERSE tree (one ~75-min-cap Dijkstra
# serving every cell's egress + pure-walk path), so it's cached per ~110m workplace bucket;
# entries are ~10 MB (k=4 dist+pred over the 215k-node graph), hence the small bound. Walk
# paths are SPEED-INVARIANT (the scalar multiplies every edge uniformly — route choice and
# node chains don't change), so the key carries no speed/rides.
_WALKPATH_TREE_CACHE = BoundedLRU(4)         # coarse (lat,lon) -> walk.PathTree (reverse, W-rooted)
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


def raptor_tree(lat, lon, max_rides=DEFAULT_MAX_RIDES, speed=DEFAULT_SPEED, walk_scalar=1.0):
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
    key = coarse_key(lat, lon, max_rides, speed)
    hit = _RAPTOR_TREE_CACHE.get(key)
    if hit is not None:
        return hit
    egress_g, egress_w, purewalk = raptor_egress_purewalk(lat, lon)
    tree5 = None
    if RAPTOR_SEMANTIC == "departafter":
        # Under depart-after best-case (p5) and typical (p50) are DIFFERENT percentiles of the
        # departure window -> DIFFERENT drawn journeys (can be different routes), so each must show
        # its OWN journey. We build BOTH a p50 tree (the typical headline + the map color) and a p5
        # tree (the best-case), each anchored on the SAME arrivalW the served map paints with, so
        # tree.itinerary(ci).total == that percentile EXACTLY (hover==map for BOTH percentiles, by
        # construction). Two cheap trees (~50ms each; the per-T* reverse-traced trees inside are
        # lazy, only built on first hover/color-by-line); the p5 painted minute comes straight off
        # tree5.commute() (the SAME kernel engine.departafter p5 uses), so we no longer pay a
        # separate departafter(percentiles=(5,)) pass. The dominant line is NOT computed here:
        # dominant() would trace the window's ~20 per-T* trees (~0.9s) and only color-by-line needs
        # it — built lazily on the first /attribution (cached inside the p50 tree), which holds the
        # /compute + hover build to ~80ms.
        tree = _RAPTOR.journey_tree_departafter(egress_g, egress_w, purewalk, percentile=50.0,
                                                max_rounds=int(max_rides), walk_scalar=walk_scalar)
        tree5 = _RAPTOR.journey_tree_departafter(egress_g, egress_w, purewalk, percentile=5.0,
                                                 max_rounds=int(max_rides), walk_scalar=walk_scalar)
        commute = tree.commute()                          # p50 painted minute, == p50 itinerary total
        p5 = tree5.commute()                              # p5 painted minute, == p5 itinerary total
        cells = {c: ([int(p5[i]) if p5[i] >= 0 else int(commute[i]),
                      int(commute[i])] if commute[i] >= 0 else [None, None])
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
    entry = {"tree": tree, "tree5": tree5, "cells": cells, "dom": domd,
             "geom": {}, "geom5": ({} if tree5 is not None else None)}
    _RAPTOR_TREE_CACHE.put(key, entry)
    return entry


def compute_raptor(lat, lon, max_rides=DEFAULT_MAX_RIDES, speed=DEFAULT_SPEED, walk_scalar=1.0):
    """Grid travel-times via the RAPTOR engine -> {id: [best, real]} (same shape as compute).
    BOTH semantics now serve from the cached traced tree so map==refine==hover (JVM-free):
    arrive-by uses the single-deadline tree (actual commute); depart-after uses the
    DepartAfterJourneyTree's painted [p5, p50] (the p50 map color == the breakdown total)."""
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
                       walk_scalar=1.0):
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


def raptor_mc(dlat, dlon, max_rides=DEFAULT_MAX_RIDES, speed=DEFAULT_SPEED, walk_scalar=1.0):
    """{"realistic": {id: min}, "variance": {id: {frag, std, stuck, alt}}} for this workplace.
    realistic = MC p50 (clamped >= perfect); frag = p90-p50 bad-day delta; stuck = fraction of
    draws hitting the cap; alt = lines that become dominant under delays (EXCLUDING the cell's
    normal line). Reachability follows the perfect map (unreachable cells are omitted).

    Concurrency (mirrors /compute_exact + _itineraries_cached): the first arriver for a key
    owns the build; same-key arrivals wait on its Event then re-read the cache; a build needed
    while ANOTHER key's build holds _MC_BUSY raises Busy (-> 503 + Retry-After) instead of
    pinning a waitress thread queued on the serialized numba kernel."""
    key = coarse_key(dlat, dlon, max_rides, speed)
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
            raise Busy()         # surface the same retryable 503 the owner's request got
        return res
    try:
        if not _MC_BUSY.acquire(blocking=False):
            raise Busy()         # another workplace's MC is running -> 503, don't queue
        try:
            return _raptor_mc_build(key, dlat, dlon, max_rides, speed, walk_scalar)
        finally:
            _MC_BUSY.release()
    finally:
        with _RAPTOR_MC_LOCK:
            _RAPTOR_MC_INFLIGHT.pop(key, None)
        event.set()


def _raptor_mc_build(key, dlat, dlon, max_rides, speed, walk_scalar):
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
    # deterministic per-workplace seed -> the realistic numbers are stable across reboots/reloads
    seed = mc_seed(dlat, dlon, max_rides, speed)
    mc = _RAPTOR.montecarlo(egress_g, egress_w, purewalk, perfect=perfect, seed=seed,
                            walk_scalar=walk_scalar, max_rounds=int(max_rides),
                            tree=entry.get("tree"))   # reuse the cached trace (no re-trace)
    # dominant line per cell (to drop the cell's PRIMARY line from its alt chips). Arrive-by filled
    # entry["dom"] eagerly at tree build; depart-after left it None (lazy) -> trace it now off the
    # SAME cached tree (the per-T* trees are already built by committed_first_legs above, so this is
    # cheap and stays JVM-free). raptor_attribution caches it on the tree for /attribution reuse.
    dom = entry["dom"]
    if dom is None and mc["alt"]:
        dom = raptor_attribution(dlat, dlon, max_rides, speed, walk_scalar)
    dom = dom or {}
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


def mc_peek(dlat, dlon, max_rides, speed):
    """Return the cached MC entry for this workplace+rides+speed, or None — WITHOUT ever
    triggering a build. /itinerary uses this so a hover never pays the ~1s MC cost; the alt
    routes simply stay [] until the frontend's /variance fetch lands (after which the next
    hover finds the entry). A shallow dict(entry) copy, so the returned wrapper is private but
    the shared alt_geom dict is the live cache value (mutated under _ALT_LOCK)."""
    return _RAPTOR_MC_CACHE.get(coarse_key(dlat, dlon, max_rides, speed))


def _alt_route_preamble(ci, dlat, dlon, max_rides, speed):
    """Shared front-half of BOTH ``_itinerary_alts`` variants (arrive-by + depart-after): resolve
    the cell's MC alt chips/bundle, the per-cell geom cache, and the per-line access-stop map. The
    arrive-by/depart-after halves differ ONLY in the per-line trace + output shape (and so are NOT
    merged — that would mix the flat ``{line,min,legs}`` and nested ``{line,best,typical}`` shapes
    that two suites assert byte-for-byte); this de-dups the identical preamble they share.

    Returns one of:
      ("empty", None)               — no alts (variance not built / no chips / no alt_stop);
      ("cached", out)               — the cell's alts are already assembled (return ``out``);
      ("ready", (geom_cache, cell_alt_stop, chips)) — proceed to trace each chip line.
    Byte-identical to the inline preambles it replaced (same checks, same order, same returns)."""
    mc = mc_peek(dlat, dlon, max_rides, speed)
    if mc is None:
        return ("empty", None)                          # variance not built yet -> no alts
    chips = mc.get("alt_chips", {}).get(ci)
    bundle = mc.get("alt_bundle")
    if not chips or not bundle:
        return ("empty", None)
    geom_cache = mc["alt_geom"]
    cached = geom_cache.get(ci)
    if cached is not None:
        return ("cached", cached)
    alt_stop = bundle.get("alt_stop")
    cell_alt_stop = (alt_stop[ci] if alt_stop and ci < len(alt_stop) else None) or {}
    if not cell_alt_stop:
        return ("empty", None)
    return ("ready", (geom_cache, cell_alt_stop, chips))


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
    status, payload = _alt_route_preamble(ci, dlat, dlon, max_rides, speed)
    if status == "empty":
        return []
    if status == "cached":
        return payload
    geom_cache, cell_alt_stop, chips = payload          # each chip line -> its access stop
    walk_scalar = config.WALK_KMH / WALK_SPEEDS.get(speed, config.WALK_KMH)
    tree = raptor_tree(dlat, dlon, max_rides, speed, walk_scalar)["tree"]  # cache hit (same tree)
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


def _alt_typicals_preamble(ci, dlat, dlon, max_rides, speed):
    """Shared front-half of BOTH ``_itinerary_alt_typicals`` variants (arrive-by + depart-after):
    the per-pinned-cell ``typ`` cache guard. Returns one of:
      ("empty", None)             — variance not built yet (-> None to the caller);
      ("cached", out)             — this cell's typicals are already computed (return ``out``);
      ("ready", (typ_cache, mc))  — proceed; cache the assembled result into ``typ_cache[ci]``.
    The halves still diverge after this (different floors / route_typicals flags / output shape),
    so they are NOT merged — only the identical guard is shared. Byte-identical to the inline guard."""
    mc = mc_peek(dlat, dlon, max_rides, speed)
    if mc is None:
        return ("empty", None)                          # variance not built yet -> no typicals
    typ_cache = mc["typ"]
    cached = typ_cache.get(ci)
    if cached is not None:
        return ("cached", cached)
    return ("ready", (typ_cache, mc))


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
    status, payload = _alt_typicals_preamble(ci, dlat, dlon, max_rides, speed)
    if status == "empty":
        return None
    if status == "cached":
        return payload
    typ_cache, mc = payload
    walk_scalar = config.WALK_KMH / WALK_SPEEDS.get(speed, config.WALK_KMH)
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
    pairs = _RAPTOR.route_typicals(tree, ci, stops, egress_g, egress_w,
                                   perfect_route_mins=floors, seed=seed,
                                   walk_scalar=walk_scalar, max_rounds=int(max_rides))
    out = {"prim": pairs[0] if pairs else None, "alts": pairs[1:] if len(pairs) > 1 else []}
    with _TYP_LOCK:                                      # cache the typicals for this pinned cell
        typ_cache[ci] = out
    return out


# ---- Depart-after /itinerary: BOTH the best-case (p5) and typical (p50) journeys -------
# Under depart-after the BEST-CASE and TYPICAL are DIFFERENT percentiles of the [DEP, DEP+WINDOW]
# departure window, so each is a DIFFERENT drawn journey (can be a different route). The frontend's
# metric toggle must switch between them LOCALLY (no re-fetch), so /itinerary ships BOTH journeys in
# one response. Each journey's total is the cell's painted percentile EXACTLY (hover==map for BOTH):
# the p50 journey traced from the cached p50 DepartAfterJourneyTree (== cells[c][1]), the p5 journey
# from the cached p5 tree (== cells[c][0]). See the JSON contract docstring in _itinerary_departafter.
def _itinerary_alts_departafter(ci, entry, dlat, dlon, max_rides, speed, prov50, prov5):
    """The drawn alternative routes for cell ``ci`` under DEPART-AFTER, each carrying BOTH its
    best-case (p5) and typical (p50) journey (total + geom legs) so the compare card can switch on
    the metric toggle without re-fetching. For each alt chip line the MC overlay surfaced for this
    cell (the dominance-window alts, after the SAME primary-exclusion + 4-cap /variance applies),
    trace that line's access stop (the bundle's per-cell ``alt_stop`` map) on the p50 tree at the
    alt's OWN per-stop p50 + p5 percentiles (``itinerary_via_stop(..., percentile=)``) — the per-stop
    percentile guarantees the alt's best-case <= its typical. Returns
    ``[{line, best:{total,legs}, typical:{total,legs}}]`` (closest-first), or [] until the MC build
    has run (we never trigger it here) / no alt chips. ``prov50``/``prov5`` are the primary route's
    geom providers reused so the per-cell access walk-tree (one Dijkstra each) is shared (p50 geom on
    prov50, p5 geom on prov5)."""
    status, payload = _alt_route_preamble(ci, dlat, dlon, max_rides, speed)
    if status == "empty":
        return []
    if status == "cached":
        return payload
    geom_cache, cell_alt_stop, chips = payload
    tree50 = entry["tree"]                               # owns arrivalW + the per-T* tree cache
    if prov50 is None:
        prov50 = _JourneyGeomProvider(dlat, dlon)
    if prov5 is None:
        prov5 = _JourneyGeomProvider(dlat, dlon)
    out = []
    for line in chips:                                   # preserve the chip (closest-first) order
        s = cell_alt_stop.get(line)
        if s is None:
            continue
        # Each alt journey is anchored on the alt's OWN per-stop percentile (p5 best-case, p50
        # typical) so alt p5 <= alt p50 holds PER alt (the latest-departure best-case would mix the
        # cell's p5/p50 deadline trees and could read the alt's best-case SLOWER than its typical).
        # The p50 tree owns the per-T* reverse-traced trees + arrivalW for both (same workplace).
        it50 = tree50.itinerary_via_stop(ci, int(s), geom_provider=prov50, percentile=50)
        it5 = tree50.itinerary_via_stop(ci, int(s), geom_provider=prov5, percentile=5)
        if it50 is None and it5 is None:
            continue
        d = {"line": line}
        if it50 is not None:
            d["typical"] = {"total": it50["total"], "legs": it50["geom"]}
        if it5 is not None:
            d["best"] = {"total": it5["total"], "legs": it5["geom"]}
        out.append(d)
    with _ALT_LOCK:                                      # cache the assembled alts for this cell
        geom_cache[ci] = out
    return out


def _itinerary_alt_typicals_departafter(ci, entry, dlat, dlon, max_rides, speed, alts):
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
    status, payload = _alt_typicals_preamble(ci, dlat, dlon, max_rides, speed)
    if status == "empty":
        return None
    if status == "cached":
        return payload
    typ_cache, mc = payload
    walk_scalar = config.WALK_KMH / WALK_SPEEDS.get(speed, config.WALK_KMH)
    tree = entry["tree"]                                 # the p50 tree (committed MC base)
    s_star, _aw, _lh, is_walk = tree._select(ci)
    bundle = mc.get("alt_bundle") or {}
    alt_stop = bundle.get("alt_stop")
    cell_alt_stop = (alt_stop[ci] if alt_stop and ci < len(alt_stop) else None) or {}
    cells = entry["cells"]
    # PRIMARY floor = the painted p50 (cells[ci][1]) — the SAME floor _raptor_mc_build passes to
    # montecarlo for this cell. Each alt floors at its OWN p50 journey best-case (the alt's "typical"
    # total). Sharing the floor + the per-workplace seed makes the per-route p90-p50 here the SAME
    # tail the served /variance frag is measured from.
    prim_p50 = cells.get(_RAPTOR.cell_ids[ci], [None, None])[1]
    stops = [int(s_star) if (not is_walk and s_star >= 0) else -1]
    floors = [prim_p50]
    for a in alts:
        stops.append(int(cell_alt_stop.get(a["line"], -1)))
        floors.append((a.get("typical") or {}).get("total"))
    egress_g, egress_w, _purewalk = raptor_egress_purewalk(dlat, dlon)
    seed = mc_seed(dlat, dlon, max_rides, speed)
    pairs = _RAPTOR.route_typicals(tree, ci, stops, egress_g, egress_w,
                                   perfect_route_mins=floors, seed=seed,
                                   walk_scalar=walk_scalar, max_rounds=int(max_rides),
                                   return_committed_p90=True)
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
    with _TYP_LOCK:                                      # cache the fragility for this pinned cell
        typ_cache[ci] = out
    return out


def itinerary_departafter(cid, olat, olon, dlat, dlon, max_rides, speed, walk_scalar, pin=False):
    """Assemble the DEPART-AFTER /itinerary response carrying BOTH the best-case (p5) and typical
    (p50) journeys for cell ``cid`` (or the nearest grid cell to olat/olon). JSON shape:

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
    ci = _RAPTOR.cell_index.get(cid) if cid is not None else None
    if ci is None:
        ci = nearest_raptor_cell(olat, olon)
    if ci is None:
        return {"error": "no route"}
    entry = raptor_tree(dlat, dlon, max_rides, speed, walk_scalar)
    # TYPICAL (p50) journey — cached per cell in entry["geom"].
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
    # BEST-CASE (p5) journey — cached per cell in entry["geom5"].
    best_j = entry["geom5"].get(ci)
    prov5 = None
    if best_j is None:
        prov5 = _JourneyGeomProvider(dlat, dlon)
        best_j = entry["tree5"].itinerary(ci, geom_provider=prov5)
        if best_j is not None:
            with _GEOM_LOCK:
                entry["geom5"][ci] = best_j
    # Root = the TYPICAL journey (back-compat default), plus explicit best/typical objects.
    res = dict(typ_j)
    res["typical"] = {k: typ_j[k] for k in ("total", "xfers", "legs", "geom") if k in typ_j}
    if best_j is not None:
        res["best"] = {k: best_j[k] for k in ("total", "xfers", "legs", "geom") if k in best_j}
    # Drawn alternative routes, each with BOTH percentiles' journeys. Empty until /variance built
    # the MC for this workplace (we never trigger it here, so the hover stays cheap).
    res["alts"] = _itinerary_alts_departafter(ci, entry, dlat, dlon, max_rides, speed,
                                              prov50, prov5)
    # PINNED cells (?pin=1): per-route fragility (p90-p50), MC-committed, on the primary + each alt.
    if pin and res.get("alts") is not None:
        fr = _itinerary_alt_typicals_departafter(ci, entry, dlat, dlon, max_rides, speed,
                                                 res["alts"])
        if fr is not None:
            if fr.get("prim_frag") is not None:
                res["frag"] = fr["prim_frag"]
            for a, f in zip(res["alts"], fr.get("alt_frags", [])):
                if f is not None:
                    a["frag"] = f
    return res


def itinerary_arriveby(cid, olat, olon, dlat, dlon, max_rides, speed, walk_scalar, pin=False):
    """Assemble the ARRIVE-BY /itinerary response for cell ``cid`` (or the nearest grid cell).
    Geometry from the SAME traced journey the breakdown shows (the hover==map invariant extends
    to the drawn route — never recomputed via another path). Returns the breakdown dict with
    ``geom``, ``alts`` (drawn alternatives, empty until /variance built the MC), and — on
    ?pin=1 — per-route committed-plan typical + fragility (``real``/``frag``)."""
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
    res["alts"] = (_itinerary_alts(ci, dlat, dlon, max_rides, speed, provider=provider)
                   if ci is not None and "error" not in res else [])
    # PINNED cells (?pin=1) also get a per-ROUTE committed-plan typical + fragility so the
    # compare card can show every strip on the SAME metric as the selector (the primary's typical
    # was already best-case-vs-typical; the alts now carry their own). Lazy + cached per pinned
    # cell, scored by the same committed MC; NEVER computed on a plain hover (the gate below).
    if (pin and ci is not None and "error" not in res
            and res.get("alts") is not None):
        typ = _itinerary_alt_typicals(ci, dlat, dlon, max_rides, speed, res["alts"])
        if typ is not None:
            prim = typ.get("prim")
            if prim is not None:
                res["real"], res["frag"] = prim[0], prim[1]
            for a, p in zip(res["alts"], typ.get("alts", [])):
                if p is not None:
                    a["real"], a["frag"] = p[0], p[1]
    return res
