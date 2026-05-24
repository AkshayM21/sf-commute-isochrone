#!/usr/bin/env python
"""
Live commute-isochrone server: set ANY workplace address (or click/drag the pin)
and the map recomputes door-to-door times from a grid of SF origins.

The R5 network is loaded ONCE at startup and kept warm, so each recompute is just
the travel-time-matrix step (no network rebuild). Run:

    .venv/bin/python scripts/server.py        # then open http://127.0.0.1:8000

Env: GRID_M (default 250) trades detail for speed.
"""
import os, io, json, copy, time, threading, datetime as dt, urllib.request, urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, geopandas as gpd, shapely
from shapely.geometry import Point, box
from flask import Flask, request, jsonify

from r5py import TransportNetwork, TravelTimeMatrix, TransportMode
from r5py.r5.regional_task import RegionalTask
import com.conveyal.r5            # JVM already started by the r5py imports above
from destination import DEST_LAT, DEST_LON, DEST_LABEL  # configurable; see .env
from itineraries import load_routes
from route_map import route_shapes

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UTM = "EPSG:32610"
GRID_M = int(os.environ.get("GRID_M", "200"))   # fine grid is fine — reverse routing is sub-second
DEP = dt.datetime(2026, 5, 20, 8, 35)
WINDOW = dt.timedelta(minutes=int(os.environ.get("WINDOW_MIN", "30")))
DEFAULT = (DEST_LAT, DEST_LON, DEST_LABEL)
GTFS = [DATA / "muni_current.zip", DATA / "bart_gtfs.zip"]
ROUTES = load_routes(GTFS)
LINES = route_shapes(GTFS)          # GTFS line geometries for the overlay


def route_name(route_id, feed):
    """Feed-aware route name (Muni '8' vs BART '8'=Red-N). Matches fastest_itin()/
    load_routes(): a leg's `route_id` is resolved within its own `feed`'s id->name map."""
    try:
        k = str(int(float(route_id)))
    except (ValueError, TypeError):
        k = str(route_id)
    return ROUTES.get(str(feed), {}).get(k, k)

print(f"[boot] building grid @ {GRID_M}m + loading R5 network (once)...")
NEIGH = gpd.read_file(DATA / "sf_neighborhoods.geojson").to_crs("EPSG:4326")
_poly = NEIGH.to_crs(UTM).union_all()
_minx, _miny, _maxx, _maxy = _poly.bounds
_pts = [Point(x, y) for x in np.arange(_minx+GRID_M/2, _maxx, GRID_M)
        for y in np.arange(_miny+GRID_M/2, _maxy, GRID_M)]
_g = gpd.GeoDataFrame(geometry=_pts, crs=UTM)
_g = _g[_g.within(_poly)].reset_index(drop=True)
_g["id"] = _g.index.astype(str)
GRID = _g.to_crs("EPSG:4326")[["id", "geometry"]]
ORIGIN_LL = {r.id: (r.geometry.y, r.geometry.x) for r in GRID.itertuples()}

# cell squares (sent to browser once) + neighborhood label per cell
_half = GRID_M / 2
_sq = gpd.GeoSeries([box(p.x-_half, p.y-_half, p.x+_half, p.y+_half) for p in _g.geometry],
                    crs=UTM).to_crs("EPSG:4326")
_cells = gpd.GeoDataFrame({"id": _g["id"].values}, geometry=_sq.values, crs="EPSG:4326")
_cells = gpd.sjoin(_cells, NEIGH[["name", "geometry"]], how="left", predicate="intersects") \
            .drop_duplicates("id").drop(columns="index_right")
CELLS_GEOJSON = json.loads(_cells.to_json())
for f in CELLS_GEOJSON["features"]:
    f["properties"] = {"id": f["properties"]["id"], "n": f["properties"].get("name")}

NET = TransportNetwork(str(DATA / "osm_sf.pbf"),
                       [str(DATA / "muni_current.zip"), str(DATA / "bart_gtfs.zip")])

# Pre-snap the grid to the street network ONCE at startup. This lets every
# /compute call pass snap_to_network=False (no re-snapping 1000+ points per
# request) and, more importantly, lets us use the grid as the destination set
# of a single R5 routing tree (see compute() below).
SNAPPED_GRID = GRID.copy()
SNAPPED_GRID["geometry"] = NET.snap_to_network(GRID.geometry)
SNAPPED_GRID = SNAPPED_GRID[SNAPPED_GRID.geometry != shapely.Point()].reset_index(drop=True)
print(f"[boot] ready: {len(GRID)} origins ({len(SNAPPED_GRID)} on-network). "
      f"Open http://127.0.0.1:8000")

# Pre-extract origin ids/geometries once (used by the threaded exact router below).
_EXACT_IDS = list(SNAPPED_GRID.id)
_EXACT_GEOMS = list(SNAPPED_GRID.geometry)
MAX_INT32 = (2 ** 31) - 1
# The TransitLayer is needed to materialise a recorded path's transit legs (route names,
# board/alight stops). It's read-only and shared across threads like the rest of NET.
_TRANSIT_LAYER = NET._transport_network.transitLayer

# Thread pool for the EXACT recompute. R5 routing is read-only against the shared
# warm TransportNetwork and each task clones its own Java RegionalTask, so per-origin
# routing parallelises safely across threads in this one JVM (same pattern r5py's own
# DetailedItineraries uses via joblib). ~8 threads is the sweet spot on this box:
# the Java RAPTOR releases the GIL, but Python-side result extraction does not, so the
# real-world speedup tops out ~4.8x around the physical-core count (benchmarked).
_N_PHYS = os.cpu_count() or 8
EXACT_THREADS = int(os.environ.get("EXACT_THREADS", str(min(8, _N_PHYS))))
_EXACT_POOL = ThreadPoolExecutor(max_workers=EXACT_THREADS,
                                 thread_name_prefix="r5-exact")

# Serialise heavy routing jobs (/compute_exact, /attribution) against each other so a
# single heavy request can use all cores; concurrent heavy requests would just thrash
# the same cores and also blow up transient memory. Light /itinerary (one OD pair) and
# the fast /compute do NOT take this lock, so hover stays responsive while a heavy
# refine/attribution job runs. (See JOB 2.)
_HEAVY_LOCK = threading.Lock()

# The user-facing exact refine (/compute_exact) takes PRIORITY over the background
# attribution prewarm: the frontend calls /compute (which kicks off the prewarm) and then
# /compute_exact, so we must not let the ~30s prewarm grab _HEAVY_LOCK ahead of the exact
# pass. _EXACT_PENDING counts exact jobs that are waiting-for or holding the lock; the
# prewarm spins (briefly) until it's zero before acquiring, so exact always wins the lock.
_EXACT_PENDING = 0
_EXACT_PENDING_LOCK = threading.Lock()

# ---- Shared per-cell itinerary + attribution prewarm cache ----------------------------
# ONE background pass (~30s, keyed by workplace) computes the FULL exact itinerary for
# EVERY grid cell using R5's recorded paths (the SAME forward cell->workplace journeys
# behind compute_exact, so each itinerary's total == the cell's exact map-color value).
# The moment a workplace is set (first /compute hit) we kick this off in a BACKGROUND
# thread and cache it by destination:
#   _ITIN_CACHE[dest_key] = {cellId: {total, xfers, legs:[...]}}
# /itinerary then serves a cached cell instantly (~ms, no R5 call, no heavy lock), and
# /attribution derives the "color by line" map from the SAME cache (the dominant line of
# each cached itinerary). Both endpoints share this one prewarm -- no double work.
_ITIN_CACHE = {}                 # dest_key -> {cellId: itinerary dict}
_ITIN_INFLIGHT = {}             # dest_key -> threading.Event (compute in progress)
_ITIN_CACHE_LOCK = threading.Lock()
_PREWARM_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prewarm")


def _dest_key(lat, lon):
    return (round(float(lat), 5), round(float(lon), 5))


def _itineraries_cached(dlat, dlon, background=False):
    """Return the full per-cell itinerary map for (dlat,dlon) -> {cellId: itin dict},
    using/populating the shared prewarm cache. If another thread is already computing this
    destination, wait for it instead of recomputing. When ``background`` (the prewarm),
    yield to any pending /compute_exact before grabbing the heavy lock so the user-facing
    time refine always goes first."""
    key = _dest_key(dlat, dlon)
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
    if not owner:                # someone else is computing it; wait for their result
        event.wait()
        return _ITIN_CACHE.get(key, {})
    try:
        if background:
            # The frontend sends /compute -> (prewarm fires) -> /compute_exact almost
            # immediately. Give that exact request a moment to register as pending, then
            # always yield to any pending/active exact before grabbing the heavy lock, so
            # the user-facing time refine finishes first and the prewarm fills in after.
            time.sleep(2.0)
            while True:
                with _EXACT_PENDING_LOCK:
                    clear = _EXACT_PENDING == 0
                if clear:
                    break
                time.sleep(0.1)
        with _HEAVY_LOCK:        # serialise vs /compute_exact so cores aren't thrashed
            itins = prewarm_itineraries(dlat, dlon)
        with _ITIN_CACHE_LOCK:
            _ITIN_CACHE[key] = itins
        return itins
    finally:
        with _ITIN_CACHE_LOCK:
            _ITIN_INFLIGHT.pop(key, None)
        event.set()


def _dominant_line(itin):
    """The "color by line" attribution for a cached itinerary: the transit leg carrying
    the most ride time, or 'walk only' when the trip has no transit legs."""
    rides = [l for l in itin["legs"] if l["mode"] != "walk"]
    if not rides:
        return "walk only"
    return max(rides, key=lambda l: l["min"])["line"]


def _attribution_from_cache(dlat, dlon, background=False):
    """Derive {cellId: dominantLine} from the shared per-cell itinerary cache (building it
    if needed). Same answer the old standalone attribution() gave, from one shared pass."""
    return {cid: _dominant_line(it) for cid, it in
            _itineraries_cached(dlat, dlon, background).items()}


def _prewarm_itineraries(dlat, dlon):
    """Fire-and-forget: compute+cache the full per-cell itineraries for a just-set
    workplace, in the background, unless already cached or in flight."""
    key = _dest_key(dlat, dlon)
    with _ITIN_CACHE_LOCK:
        if key in _ITIN_CACHE or key in _ITIN_INFLIGHT:
            return
    _PREWARM_POOL.submit(_itineraries_cached, dlat, dlon, True)


app = Flask(__name__)


def compute(lat, lon):
    """Door-to-door times from every grid cell TO (lat, lon), as {id: [best, real]}.

    FAST APPROXIMATION (reverse one-to-many): R5 is dramatically faster computing
    ONE routing tree from a single point to many destinations than many separate
    one-origin trees. So instead of routing grid-cell -> workplace for all ~1.3k
    cells (a separate tree each, ~60s), we route a single tree FROM the workplace
    TO every (pre-snapped) grid cell (~0.2s) and treat that as the cell's commute
    time. The morning commute (cell -> work) and the reverse (work -> cell) are not
    perfectly symmetric (one-way streets, asymmetric schedules), so this introduces
    a small error: measured MAE ~2 min vs the exact many->one method, corr >0.95,
    with near-identical reachable-cell sets. The static scripts/isochrone.py remains
    the exact (forward) reference. departure_time_window stays at 30 min and
    percentiles stay [5=best, 50=realistic] -- the reverse call is so cheap that
    shrinking them buys nothing.
    """
    origin = gpd.GeoDataFrame({"id": ["w"]}, geometry=[Point(lon, lat)], crs="EPSG:4326")
    ttm = TravelTimeMatrix(
        NET, origins=origin, destinations=SNAPPED_GRID, snap_to_network=True,
        departure=DEP, departure_time_window=WINDOW, max_time=dt.timedelta(minutes=75),
        transport_modes=[TransportMode.TRANSIT, TransportMode.WALK],
        percentiles=[5, 50], speed_walking=4.8)
    ttm = pd.DataFrame(ttm)
    cells = {}
    for _, r in ttm.iterrows():
        b, rl = r["travel_time_p5"], r["travel_time_p50"]
        cells[str(r["to_id"])] = [None if pd.isna(b) else int(b),
                                  None if pd.isna(rl) else int(rl)]
    return cells


def compute_exact(lat, lon):
    """EXACT (forward) door-to-door: route every grid cell -> workplace. This is the
    slow, accurate direction (one R5 routing tree per origin).

    SPEEDUP (JOB 1): r5py's own TravelTimeMatrix runs the per-origin trees SERIALLY in a
    Python list comprehension (~66s for ~1362 origins @ GRID_M=300), which is the real
    bottleneck -- R5's RAPTOR itself is fast per tree, but Python<->Java marshaling is
    paid 1362 times in sequence and nothing is parallelised. Here we instead build the
    routing request ONCE, then drive R5's TravelTimeComputer directly from a thread pool:
    each thread clones the Java RegionalTask (cheap, Cloneable) and routes one origin to
    the single workplace. The Java RAPTOR call releases the GIL, so threads inside this
    one warm JVM run concurrently. This is bit-exact to the serial matrix (benchmarked:
    MAE 0.000 min, identical reachable set) -- we keep the 30-min departure window and
    [5=best, 50=realistic] percentiles untouched, since shrinking the window saves only
    ~3s but costs ~2 min MAE and drops reachable cells. Measured: ~66s -> ~14s (~4.7x).

    Returns the same shape as before: {id: [best, real]}."""
    dest = gpd.GeoDataFrame({"id": ["d"]}, geometry=[Point(lon, lat)], crs="EPSG:4326")

    # Build the routing template once (sets destination point set + linkage cache, in
    # this calling thread, so worker threads never race on cache population).
    template = RegionalTask(
        NET, origin=None, destinations=dest,
        departure=DEP, departure_time_window=WINDOW, max_time=dt.timedelta(minutes=75),
        transport_modes=[TransportMode.TRANSIT, TransportMode.WALK],
        percentiles=[5, 50], speed_walking=4.8)
    template.destinations = dest

    def _one(i):
        req = copy.copy(template)        # clones the underlying Java RegionalTask
        req.origin = _EXACT_GEOMS[i]
        computer = com.conveyal.r5.analyst.TravelTimeComputer(req, NET)
        # travelTimes.getValues() -> [percentile_index][destination_index]; 1 destination
        vals = computer.computeTravelTimes().travelTimes.getValues()
        b = int(vals[0][0]); rl = int(vals[1][0])
        return _EXACT_IDS[i], [None if b == MAX_INT32 else b,
                               None if rl == MAX_INT32 else rl]

    cells = {}
    for cid, pair in _EXACT_POOL.map(_one, range(len(_EXACT_IDS)), chunksize=8):
        cells[cid] = pair
    return cells


@app.route("/compute_exact")
def _compute_exact():
    global _EXACT_PENDING
    lat = float(request.args["lat"]); lon = float(request.args["lon"])
    t0 = dt.datetime.now()
    with _EXACT_PENDING_LOCK:             # mark exact as pending so the prewarm yields
        _EXACT_PENDING += 1
    try:
        with _HEAVY_LOCK:                  # one heavy job at a time; hover stays free
            cells = compute_exact(lat, lon)
    finally:
        with _EXACT_PENDING_LOCK:
            _EXACT_PENDING -= 1
    ms = (dt.datetime.now() - t0).total_seconds() * 1000
    print(f"[exact] ({lat:.4f},{lon:.4f}) {ms:.0f}ms")
    return jsonify({"dest": [lat, lon], "cells": cells, "ms": round(ms)})


def _path_route_name(raw):
    """R5 records a path leg's route as a string "NAME (route_id)" where NAME is the
    GTFS route_short_name/long_name resolved within the leg's own feed -- i.e. already
    feed-aware (it yields Muni '8' but BART 'Red-N (8)'). That NAME is byte-identical to
    load_routes()/route_name() (verified), so we just take the part before " (".
    Falls back to the raw string if the format is unexpected."""
    s = str(raw)
    i = s.rfind(" (")
    return s[:i] if i > 0 else s


_TINY_HOP_MIN = 2.0   # suppress sub-2-min transit hops (fold into adjacent walk)


def _build_itin(p50, itin_map, route_name_fn):
    """Turn an R5 recorded-path result for ONE cell into the /itinerary JSON breakdown.

    ``p50`` is the cell's authoritative realistic travel time (minutes) -- the SAME value
    compute_exact colors the map with -- and ``itin_map`` is that cell's path-result
    multimap (RouteSequence -> Iterations), as produced with includePathResults. We:
      1. pick the recorded ITERATION whose total best matches p50 (so the breakdown's total
         == the cell's exact map color, fixing the old "breakdown != color" mismatch);
      2. read the EXACT per-leg components from R5 -- ride time per leg comes from the
         StopSequence's rideTimesSeconds (the TransitLeg.inVehicleTime field is unreliable
         in this build), wait per leg from the iteration's waitTimes, plus access/egress
         walk and the aggregate transfer-walk time;
      3. lay them out as walk[access] -> (wait+ride) per leg -> walk[transfer] -> ... ->
         walk[egress], so the legs ALWAYS sum exactly to round(p50);
      4. FOLD any sub-2-min transit hop into the surrounding walk (its ride+wait time
         becomes walking), so we never show nonsense like "ride the 14 for 1 minute";
      5. round and reconcile any residual into a walk leg so the displayed legs still sum
         to round(p50).
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
    # iteration whose total best matches the authoritative p50 (tie -> faster trip)
    _, rseq, it = min(cands, key=lambda c: (abs(round(c[0] / 60.0) - p50), c[0]))
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


def prewarm_itineraries(dlat, dlon):
    """Full exact itinerary for EVERY grid cell -> workplace, via R5 recorded paths.

    This is the per-cell breakdown analogue of compute_exact / attribution: route every
    cell -> workplace ONCE with includePathResults, and for each cell build the full
    door-to-door breakdown ({total, xfers, legs}) with _build_itin. Because these are the
    same forward (cell->workplace) journeys behind compute_exact, every itinerary's total
    equals that cell's exact map-color value. Runs ~30s on the shared thread pool (in the
    background prewarm); /itinerary then serves a cell from this cache instantly, and
    /attribution derives the dominant line per cell from it. Returns {cellId: itin dict}."""
    dest = gpd.GeoDataFrame({"id": ["d"]}, geometry=[Point(dlon, dlat)], crs="EPSG:4326")
    template = RegionalTask(
        NET, origin=None, destinations=dest,
        departure=DEP, departure_time_window=WINDOW, max_time=dt.timedelta(minutes=75),
        transport_modes=[TransportMode.TRANSIT, TransportMode.WALK],
        percentiles=[5, 50], speed_walking=4.8)
    template.destinations = dest
    template._regional_task.includePathResults = True   # record paths, not just times
    template._regional_task.nPathsPerTarget = 8         # a few options to find the p50 one

    def _one(i):
        req = copy.copy(template)
        req.origin = _EXACT_GEOMS[i]
        result = com.conveyal.r5.analyst.TravelTimeComputer(req, NET).computeTravelTimes()
        p50 = result.travelTimes.getValues()[1][0]      # realistic (matches compute_exact)
        if p50 == MAX_INT32:
            return _EXACT_IDS[i], None                  # unreachable -> omit
        paths = result.paths
        itin_map = paths.iterationsForPathTemplates[0].asMap() if paths is not None else None
        if itin_map is None:
            return _EXACT_IDS[i], {"total": int(p50), "xfers": 0,
                                   "legs": [{"mode": "walk", "line": None, "min": int(p50)}]}
        return _EXACT_IDS[i], _build_itin(p50, itin_map, _path_route_name)

    itins = {}
    for cid, itin in _EXACT_POOL.map(_one, range(len(_EXACT_IDS)), chunksize=8):
        if itin is not None:
            itins[cid] = itin
    return itins


def fastest_itin(olat, olon, dlat, dlon):
    """On-demand single-OD breakdown via R5 recorded paths -- the FALLBACK for /itinerary
    cache misses (off-grid points, or before the prewarm has finished). Mirrors exactly
    what the prewarm caches: route this one origin -> workplace with path recording, take
    the p50 (realistic) time, and build the same breakdown with _build_itin, so a cold
    lookup reads identically to a cached one and its total matches the map color."""
    o = NET.snap_to_network(gpd.GeoSeries([Point(olon, olat)], crs="EPSG:4326")).iloc[0]
    if o.is_empty:
        return None
    dest = gpd.GeoDataFrame({"id": ["d"]}, geometry=[Point(dlon, dlat)], crs="EPSG:4326")
    template = RegionalTask(
        NET, origin=None, destinations=dest,
        departure=DEP, departure_time_window=WINDOW, max_time=dt.timedelta(minutes=75),
        transport_modes=[TransportMode.TRANSIT, TransportMode.WALK],
        percentiles=[5, 50], speed_walking=4.8)
    template.destinations = dest
    template._regional_task.includePathResults = True
    template._regional_task.nPathsPerTarget = 8
    req = copy.copy(template)
    req.origin = o
    result = com.conveyal.r5.analyst.TravelTimeComputer(req, NET).computeTravelTimes()
    p50 = result.travelTimes.getValues()[1][0]
    if p50 == MAX_INT32:
        return None
    paths = result.paths
    itin_map = paths.iterationsForPathTemplates[0].asMap() if paths is not None else None
    if itin_map is None:
        return {"total": int(p50), "xfers": 0,
                "legs": [{"mode": "walk", "line": None, "min": int(p50)}]}
    return _build_itin(p50, itin_map, _path_route_name)


@app.route("/itinerary")
def _itinerary():
    cid = request.args.get("id")
    if cid is not None and cid in ORIGIN_LL:
        olat, olon = ORIGIN_LL[cid]
    else:
        olat, olon = float(request.args["olat"]), float(request.args["olon"])
    dlat, dlon = float(request.args["dlat"]), float(request.args["dlon"])
    res = None
    # FAST PATH: serve the prewarmed per-cell cache (a plain dict lookup; no R5 call, no
    # heavy lock) so hover/click is ~instant once the workplace's prewarm has run.
    if cid is not None:
        with _ITIN_CACHE_LOCK:
            cached = _ITIN_CACHE.get(_dest_key(dlat, dlon))
        if cached is not None and cid in cached:
            res = cached[cid]
    # FALLBACK: cache miss (off-grid point, or prewarm not finished) -> compute on demand.
    if res is None:
        res = fastest_itin(olat, olon, dlat, dlon) or {"error": "no route"}
    res = dict(res)                       # don't mutate the cached object with olat/olon
    res["olat"], res["olon"] = round(olat, 5), round(olon, 5)
    return jsonify(res)


@app.route("/attribution")
def _attribution():
    """The "color by line" map: dominant transit line per cell, DERIVED from the shared
    per-cell itinerary prewarm cache (one pass feeds both /itinerary and /attribution)."""
    dlat = float(request.args["dlat"]); dlon = float(request.args["dlon"])
    t0 = dt.datetime.now()
    with _ITIN_CACHE_LOCK:
        cached = _dest_key(dlat, dlon) in _ITIN_CACHE
    attr = _attribution_from_cache(dlat, dlon)   # serve prewarm cache, or compute+cache now
    ms = (dt.datetime.now() - t0).total_seconds() * 1000
    print(f"[attr] ({dlat:.4f},{dlon:.4f}) {ms:.0f}ms -> {len(attr)} cells"
          f"{' (cached)' if cached else ''}")
    return jsonify(attr)


@app.route("/compute")
def _compute():
    lat = float(request.args["lat"]); lon = float(request.args["lon"])
    t0 = dt.datetime.now()
    cells = compute(lat, lon)
    ms = (dt.datetime.now() - t0).total_seconds() * 1000
    print(f"[compute] ({lat:.4f},{lon:.4f}) {ms:.0f}ms")
    # The address was just set -> start the (~30s) per-cell itinerary prewarm in the
    # background now, so /itinerary (hover/click breakdown) is an instant cache hit and the
    # "Primary line" attribution is ready too -- both from this one pass. No FE change.
    _prewarm_itineraries(lat, lon)
    return jsonify({"dest": [lat, lon], "cells": cells, "ms": round(ms)})


@app.route("/geocode")
def _geocode():
    q = request.args.get("q", "")
    if not q.strip():
        return jsonify({"error": "empty"}), 400
    if "san francisco" not in q.lower() and "sf" not in q.lower():
        q = q + ", San Francisco, CA"
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": q, "format": "json", "limit": 1})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "sf-commute-isochrone/1.0"})
        d = json.load(urllib.request.urlopen(req, timeout=12))
        if d:
            return jsonify({"lat": float(d[0]["lat"]), "lon": float(d[0]["lon"]),
                            "label": d[0].get("display_name", q)})
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    return jsonify({"error": "not found"}), 404


@app.route("/")
def _index():
    return PAGE.replace("__CELLS__", json.dumps(CELLS_GEOJSON)) \
               .replace("__LINES__", json.dumps(LINES)) \
               .replace("__DEFAULT__", json.dumps(DEFAULT))


PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SF Commute Explorer — set your own workplace</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  :root{--panel:#171a21;--ink:#e8eaed;--mut:#9aa3af;--line:#272b34;--accent:#5ab0ff}
  *{box-sizing:border-box} html,body{margin:0;height:100%;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  #map{position:absolute;inset:0;background:#0b0d10}.leaflet-container{background:#0b0d10}
  #panel{position:absolute;top:0;right:0;width:330px;height:100%;background:var(--panel);color:var(--ink);
    z-index:1000;display:flex;flex-direction:column;border-left:1px solid var(--line)}
  h1{font-size:15px;margin:0;padding:14px 16px 3px}.sub{color:var(--mut);font-size:12px;padding:0 16px 10px;line-height:1.45}
  .ctl{padding:9px 16px;border-top:1px solid var(--line)}
  .lab{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin-bottom:7px}
  .addr{display:flex;gap:6px}.addr input{flex:1;background:#0e1116;border:1px solid var(--line);color:var(--ink);
    padding:8px;border-radius:8px;font-size:13px}.addr button{background:var(--accent);color:#06121f;border:0;
    padding:0 12px;border-radius:8px;font-weight:600;cursor:pointer}
  .seg{display:flex;gap:6px}.seg button{flex:1;background:#11141a;color:var(--ink);border:1px solid var(--line);
    padding:7px;border-radius:8px;font-size:12.5px;cursor:pointer}.seg button.on{background:var(--accent);color:#06121f;font-weight:600;border-color:var(--accent)}
  input[type=range]{width:100%;accent-color:var(--accent)} .val{color:var(--accent);font-weight:600}
  #list{flex:1;overflow:auto;border-top:1px solid var(--line)}
  .nb{display:flex;justify-content:space-between;padding:7px 16px;font-size:13px;cursor:pointer;border-bottom:1px solid #1c2027}
  .nb:hover{background:#1f2530}.nb .t{font-variant-numeric:tabular-nums;font-weight:600}.nb small{color:var(--mut)}
  #legend{position:absolute;left:14px;bottom:16px;z-index:1000;background:rgba(20,23,29,.92);color:var(--ink);
    padding:11px 13px;border:1px solid var(--line);border-radius:10px;font-size:12px}
  #legend .bar{height:11px;width:240px;border-radius:3px;margin:7px 0 4px}
  #legend .sc{display:flex;justify-content:space-between;color:var(--mut);font-size:11px}
  #busy{position:absolute;top:12px;left:12px;z-index:1100;background:var(--accent);color:#06121f;font-weight:600;
    padding:7px 12px;border-radius:20px;font-size:12.5px;display:none;box-shadow:0 2px 10px rgba(0,0,0,.4)}
  .leaflet-tooltip.tt{background:#11141a;border:1px solid var(--line);color:var(--ink);white-space:normal;padding:0;box-shadow:0 6px 20px rgba(0,0,0,.45)}
  .leaflet-tooltip.tt:before{border:none!important}
  .leaflet-popup-content{margin:0;font-size:12.5px}
  .leaflet-popup-content-wrapper{background:#11141a;color:var(--ink);border:1px solid var(--line);border-radius:10px}
  .leaflet-popup-tip{background:#11141a}
  .bd{width:238px;padding:11px 13px;font:12.5px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--ink)}
  .bd .h{font-size:13px;margin-bottom:8px}
  .bd .legs{display:flex;flex-wrap:wrap;gap:5px 5px;align-items:center}
  .bd .leg{padding:1px 7px;border-radius:7px;white-space:nowrap;font-size:11.5px;font-weight:600}
  .bd .walk{background:#262b33;color:#b6bdc8;font-weight:500}
  .bd .wait{color:#8089b3;font-style:italic;font-weight:500;padding:1px 3px;background:repeating-linear-gradient(45deg,#1c2030,#1c2030 3px,#161a26 3px,#161a26 6px);border-radius:6px}
  .bd .arr{color:#4b515c;font-size:10px}
  .bd .gm{color:#5ab0ff;display:inline-block;margin-top:9px;font-weight:600;text-decoration:none}
  .bd .gm:hover{text-decoration:underline}
  .bd .foot{color:#5c6470;margin-top:5px;font-size:10.5px}
  .credit{padding:9px 16px;color:#5c6470;font-size:10.5px;border-top:1px solid var(--line)}
</style></head>
<body>
<div id="map"></div><div id="busy">computing…</div>
<div id="prompt" style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;z-index:1050;pointer-events:none">
  <div style="background:rgba(20,23,29,.94);color:#e8eaed;padding:18px 24px;border:1px solid #272b34;border-radius:12px;font-size:15px;text-align:center;box-shadow:0 10px 34px rgba(0,0,0,.55)">
    Enter your workplace address  →<br><span style="color:#9aa3af;font-size:12.5px">(top-right) to map your commute</span></div></div>
<div id="panel">
  <h1>Commute to your workplace</h1>
  <div class="sub">Enter your workplace address to begin. <b>Hover</b> any area for its time + the route
    behind it; <b>click</b> to pin it with a Google Maps link.</div>
  <div class="ctl"><div class="lab">Workplace</div>
    <div class="addr"><input id="addr" placeholder="e.g. 1 Market St"><button id="go">Set</button></div>
    <div id="dest" style="color:var(--mut);font-size:11px;margin-top:6px"></div></div>
  <div class="ctl"><div class="lab">Time estimate</div>
    <div class="seg" id="metric"><button data-v="r" class="on">Realistic</button><button data-v="b">Best-case</button></div></div>
  <div class="ctl"><div class="lab">Color by</div>
    <div class="seg" id="cmode"><button data-v="time" class="on">Time</button><button data-v="line">Primary line</button></div></div>
  <div class="ctl"><div class="lab">Sweet spot — green ≤ <span class="val" id="idval">25</span> min (recolors)</div>
    <input type="range" id="ideal" min="10" max="45" value="25"></div>
  <div class="ctl"><div class="lab">Max commute — hide above <span class="val" id="thrval">40</span> min (filters)</div>
    <input type="range" id="thr" min="10" max="70" value="40"></div>
  <div class="ctl"><div class="lab">Overlay real transit lines</div>
    <div style="display:flex;flex-wrap:wrap;gap:11px;font-size:12.5px">
      <label style="cursor:pointer"><input type="checkbox" data-m="bart"> <span style="color:#6f8cff">BART</span></label>
      <label style="cursor:pointer"><input type="checkbox" data-m="metro"> <span style="color:#ff6b6b">Metro</span></label>
      <label style="cursor:pointer"><input type="checkbox" data-m="bus"> <span style="color:#f6a04d">Bus</span></label>
      <label style="cursor:pointer"><input type="checkbox" data-m="cable"> <span style="color:#5cd65c">Cable</span></label>
    </div></div>
  <div class="ctl" style="padding-bottom:5px"><div class="lab">Neighborhoods</div></div>
  <div id="list"></div>
  <div class="credit">R5/r5py · Muni (511) + BART · network kept warm; only the matrix recomputes.
    Live map is a fast reverse-routing approximation (~±2 min); <b>scripts/isochrone.py</b> is the exact build.</div>
</div>
<div id="legend"><b id="legtitle">Realistic door-to-door (min)</b><div class="bar"></div>
  <div class="sc"></div></div>
<script>
const CELLS=__CELLS__, LINES=__LINES__, DEFAULT=__DEFAULT__;
let metric="r", ideal=25, thr=40, cmode="time", TT={}, ATTR={}, NB={}, DESTLL=null, LINECOLOR={};
const gmaps=(olat,olon)=>`https://www.google.com/maps/dir/?api=1&origin=${olat},${olon}&destination=${DESTLL[0]},${DESTLL[1]}&travelmode=transit`;
let bdToken=0;  // cancels stale hover-breakdown fetches
const map=L.map("map").setView([37.762,-122.43],12.3);
L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",{maxZoom:19}).addTo(map);
const rgb=c=>`rgb(${c[0]},${c[1]},${c[2]})`;
function ramp(){const hi=ideal+25;return {hi,S:[[0,[0,104,55]],[ideal*.45,[26,152,80]],[ideal*.72,[102,189,99]],
  [ideal*.9,[166,217,106]],[ideal,[255,255,191]],[ideal+(hi-ideal)*.35,[253,174,97]],
  [ideal+(hi-ideal)*.7,[244,109,67]],[hi,[215,48,39]]]};}
function color(v){if(v==null)return null;const {S}=ramp();if(v<=0)return rgb(S[0][1]);
  for(let i=1;i<S.length;i++){if(v<=S[i][0]){const a=S[i-1],b=S[i],t=(v-a[0])/((b[0]-a[0])||1);
    return rgb(a[1].map((c,j)=>Math.round(c+(b[1][j]-c)*t)));}}return rgb(S.at(-1)[1]);}
function legend(){const sc=document.querySelector("#legend .sc"),bar=document.querySelector("#legend .bar");
  if(cmode=="line"){bar.style.display="none";sc.style.flexWrap="wrap";
    const ord=Object.keys(LINECOLOR);
    sc.innerHTML=ord.length?ord.slice(0,16).map(l=>`<span style="white-space:nowrap;margin:0 8px 3px 0"><span style="display:inline-block;width:9px;height:9px;background:${LINECOLOR[l]};border-radius:2px"></span> ${l}</span>`).join(""):"computing line map…";
    document.getElementById("legtitle").textContent="Primary transit line per area";return;}
  const {hi,S}=ramp();bar.style.display="block";sc.style.flexWrap="nowrap";
  bar.style.background="linear-gradient(90deg,"+S.map(s=>rgb(s[1])+" "+Math.round(s[0]/hi*100)+"%").join(",")+")";
  sc.innerHTML=`<span>0</span><span>${ideal} (ideal)</span><span>${hi}+</span>`;
  document.getElementById("legtitle").textContent=(metric=="r"?"Realistic":"Best-case")+" door-to-door (min)";}
function val(id){const v=TT[id];return v?(metric=="r"?v[1]:v[0]):null;}
function style(f){const id=f.properties.id,v=val(id);
  if(v==null||v>thr)return{fillOpacity:0,opacity:0,weight:0};
  if(cmode=="line"){const ln=ATTR[id];return ln?{fillColor:LINECOLOR[ln]||"#888",fillOpacity:.76,weight:0}:{fillColor:"#888",fillOpacity:.06,weight:0};}
  return{fillColor:color(v),fillOpacity:.72,weight:0};}
function buildLineColors(){const PAL=["#e6194B","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6","#bfef45","#fabed4","#469990","#dcbeff","#9A6324","#800000","#aaffc3","#808000","#000075"];
  const cnt={};Object.values(ATTR).forEach(l=>cnt[l]=(cnt[l]||0)+1);
  LINECOLOR={};Object.keys(cnt).sort((a,b)=>cnt[b]-cnt[a]).forEach((l,i)=>LINECOLOR[l]=l=="walk only"?"#7fd1ff":PAL[i%PAL.length]);}
function lineColor(line){  // match the overlay legend colors
  if(/^(Red|Yellow|Blue|Green|Orange|Gr[ae]y)-[NSEW]$/i.test(line))return "#6f8cff";  // BART
  if(["PM","PH","CA"].includes(line))return "#5cd65c";                                 // cable
  if(/^[JKLMNT]$/.test(line)||line==="KLM")return "#ff6b6b";                           // Muni Metro
  return "#f6a04d";                                                                    // bus
}
function bdHTML(d){
  if(d.error)return `<div class="bd"><span style="color:#9aa3af">No transit route within ~75 min.</span></div>`;
  const chips=[];
  d.legs.forEach(g=>{
    if(!g.line){chips.push(`<span class="leg walk">walk ${g.min}m</span>`);return;}
    if(g.wait&&g.wait>=1)chips.push(`<span class="leg wait">wait ${g.wait}m</span>`);
    const c=lineColor(g.line);
    chips.push(`<span class="leg" style="background:${c}26;color:${c};border:1px solid ${c}55">${g.line} · ${g.min}m</span>`);
  });
  return `<div class="bd"><div class="h"><b>${d.name||""}</b> · ${d.total} min · ${d.xfers} transfer${d.xfers==1?"":"s"}</div>`+
    `<div class="legs">${chips.join('<span class="arr">▸</span>')}</div>`+
    `<a class="gm" href="${gmaps(d.olat,d.olon)}" target="_blank">Open in Google Maps ↗</a>`+
    `<div class="foot">typical fastest trip (~8:35am)</div></div>`;}
let bdTimer;
function loadBreak(f,setHTML){const v=val(f.properties.id);if(v==null){setHTML("—");return;}
  setHTML(`<div class="bd"><div class="h"><b>${f.properties.n||""}</b> · ${v} min</div><div style="color:#9aa3af">loading route…</div></div>`);
  clearTimeout(bdTimer);const my=++bdToken;
  bdTimer=setTimeout(async()=>{try{
    const r=await fetch(`/itinerary?id=${f.properties.id}&dlat=${DESTLL[0]}&dlon=${DESTLL[1]}`);
    const d=await r.json();if(my!==bdToken)return;d.name=f.properties.n;setHTML(bdHTML(d));
  }catch(e){if(my===bdToken)setHTML("error");}},150);}
const layer=L.geoJSON(CELLS,{style,onEachFeature:(f,l)=>{
  l.on("mouseover",()=>{if(!DESTLL)return;l.setStyle({weight:1.4,color:"#fff"});
    l.bindTooltip("",{className:"tt",sticky:true,opacity:1}).openTooltip();
    loadBreak(f,h=>{if(l.getTooltip())l.setTooltipContent(h);});});
  l.on("mouseout",()=>{layer.resetStyle(l);clearTimeout(bdTimer);});
  l.on("click",()=>{if(!DESTLL)return;
    l.bindPopup("…",{maxWidth:285}).openPopup();
    loadBreak(f,h=>{if(l.getPopup())l.setPopupContent(h);});});
}}).addTo(map);
// transit-line overlays (off by default; non-interactive so cells stay hoverable)
const LINESTYLE={bus:{color:"#f6a04d",weight:1.3,opacity:.5},metro:{color:"#ff6b6b",weight:2.4,opacity:.85},
  cable:{color:"#5cd65c",weight:2,opacity:.8},bart:{color:"#6f8cff",weight:3,opacity:.85}};
const overlays={};
["bart","metro","bus","cable"].forEach(m=>{overlays[m]=L.geoJSON(
  {type:"FeatureCollection",features:LINES.features.filter(f=>f.properties.mode===m)},
  {style:()=>LINESTYLE[m],interactive:false});});
document.querySelectorAll("input[data-m]").forEach(cb=>cb.onchange=()=>{
  const m=cb.dataset.m;if(cb.checked)overlays[m].addTo(map);else map.removeLayer(overlays[m]);});
let pin=null;   // workplace marker — set via the address box only (no map-click / drag)

function redraw(){layer.setStyle(style);legend();renderList();}
function renderList(){const rows=Object.entries(NB).filter(([k,v])=>v!=null).sort((a,b)=>a[1]-b[1]);
  document.getElementById("list").innerHTML=rows.map(([n,v])=>{const c=v<=thr?color(v):"#555";
    return `<div class="nb"><span><span style="color:${c}">●</span> ${n}</span><span class="t" style="color:${c}">${v}<small> min</small></span></div>`;}).join("");}
function aggregate(){const acc={};CELLS.features.forEach(f=>{const v=val(f.properties.id),n=f.properties.n;
  if(v==null||!n)return;(acc[n]=acc[n]||[]).push(v);});NB={};
  for(const n in acc){acc[n].sort((a,b)=>a-b);NB[n]=acc[n][Math.floor(acc[n].length/2)];}}
function busy(t){const b=document.getElementById("busy");if(t){b.textContent=t;b.style.display="block";}else b.style.display="none";}
async function setWorkplace(lat,lon,label){
  document.getElementById("prompt").style.display="none";
  DESTLL=[lat,lon];ATTR={};LINECOLOR={};
  if(pin)pin.setLatLng([lat,lon]);else pin=L.marker([lat,lon]).addTo(map);
  map.setView([lat,lon],12.6);
  busy("estimating…");
  try{const r=await fetch(`/compute?lat=${lat}&lon=${lon}`);const d=await r.json();
    TT=d.cells;DESTLL=d.dest;document.getElementById("dest").textContent=(label||"")+`  ·  fast ~${d.ms}ms`;
    try{localStorage.setItem("wp",JSON.stringify({lat,lon,label}))}catch(e){}   // survives refresh
    aggregate();redraw();
  }catch(e){busy(false);alert("compute failed: "+e);return;}
  busy("refining (exact)…");                       // pipeline: exact pass refines the approximation
  try{const r2=await fetch(`/compute_exact?lat=${lat}&lon=${lon}`);
    if(r2.ok){const d2=await r2.json();TT=d2.cells;
      document.getElementById("dest").textContent=(label||"")+`  ·  exact ${(d2.ms/1000).toFixed(1)}s`;
      aggregate();redraw();}
  }catch(e){/* keep the fast approximation if exact isn't available */}
  busy(false);
  if(cmode=="line")loadAttribution();
}
async function loadAttribution(){if(!DESTLL)return;busy("mapping lines…");
  try{const r=await fetch(`/attribution?dlat=${DESTLL[0]}&dlon=${DESTLL[1]}`);
    if(r.ok){ATTR=await r.json();buildLineColors();redraw();}
    else{document.getElementById("legtitle").textContent="line map unavailable yet";}
  }catch(e){}finally{busy(false);}}
function seg(id,set){document.querySelectorAll(`#${id} button`).forEach(b=>b.onclick=()=>{
  document.querySelectorAll(`#${id} button`).forEach(x=>x.classList.remove("on"));b.classList.add("on");set(b.dataset.v);aggregate();redraw();});}
seg("metric",v=>metric=v);
seg("cmode",v=>{cmode=v;if(v=="line"&&Object.keys(ATTR).length==0&&DESTLL)loadAttribution();});
document.getElementById("ideal").oninput=e=>{ideal=+e.target.value;document.getElementById("idval").textContent=ideal;redraw();};
document.getElementById("thr").oninput=e=>{thr=+e.target.value;document.getElementById("thrval").textContent=thr;redraw();};
document.getElementById("go").onclick=async()=>{const q=document.getElementById("addr").value;
  if(!q)return;busy("finding address…");
  try{const r=await fetch(`/geocode?q=${encodeURIComponent(q)}`);const d=await r.json();
    if(d.lat){busy(false);setWorkplace(d.lat,d.lon,d.label.split(",").slice(0,2).join(","));}
    else{busy(false);alert("address not found");}}catch(e){busy(false);alert(e);}};
document.getElementById("addr").addEventListener("keydown",e=>{if(e.key=="Enter")document.getElementById("go").click();});
legend();   // draw the legend; map stays blank until you set an address
try{const wp=JSON.parse(localStorage.getItem("wp")||"null");   // restore last workplace
  if(wp&&wp.lat){document.getElementById("addr").value=wp.label||"";setWorkplace(wp.lat,wp.lon,wp.label);}
}catch(e){}
</script></body></html>"""


if __name__ == "__main__":
    # threaded=True (JOB 2): each request runs in its own thread so a slow
    # /compute_exact or /attribution can't freeze /itinerary (hover) or /compute.
    # R5 routing is thread-safe (read-only shared network; per-task Java request clones)
    # and jpype auto-attaches Python threads to the JVM. Heavy jobs are additionally
    # serialised against each other by _HEAVY_LOCK so they don't thrash cores, while the
    # light single-OD /itinerary never takes that lock and stays responsive.
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8000")), threaded=True)
