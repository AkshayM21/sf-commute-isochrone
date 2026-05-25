"""Bake the RAPTOR runtime precompute + generate R5 ground-truth oracles (the ONLY JVM step).

Boots R5 once and produces, for the canonical model (pick_service_date, WALK 4.8, p5/p50,
3 feeds):

  data/raptor_cache/access_<gridm>m_<fp>.npz   — cell -> stop WALK access table (CSR, seconds),
      keyed to RAPTOR gids (bridged here), workplace-INDEPENDENT. This is the runtime
      precompute the engine consumes (grid is fixed). Built from an R5 walk-only matrix.

  tests/raptor_golden/oracle_<name>.npz        — per validation workplace: R5 EXACT per-cell
      forward door-to-door times (p5,p50, the depart-after ground truth), plus that workplace's
      egress (W->stops walk secs) and purewalk (W->cells walk secs) so the JVM-free validator
      can feed the engine exactly what the live server would compute per request.

R5 has no native arrive-by, so the headline oracle is R5's depart-after window p50 (what the
spike validated). The engine reproduces it by inverting the reverse profile; arrive-by-09:00
is the same profile read at one deadline (validated separately on a forward-sweep sample).

Run uncapped on a big box; here R5_MAX_MEMORY caps the heap so it coexists with a running
server. Usage: R5_MAX_MEMORY=4G EXACT_THREADS=6 .venv/bin/python scripts/raptor_oracle.py
"""
import os, sys, time, copy
import datetime as dt
from pathlib import Path

# Cap the JVM heap BEFORE r5py starts it (offline batch tool; capping here is for memory
# coexistence with a running server, NOT the live-server throughput rule).
_mem = os.environ.get("R5_MAX_MEMORY")
if _mem and "--max-memory" not in sys.argv:
    sys.argv += ["--max-memory", _mem]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Point
from concurrent.futures import ThreadPoolExecutor

from core import config, grid as gridmod, feeds, network, raptor_build
from r5py import TransportMode, TravelTimeMatrix
import com.conveyal.r5 as r5  # noqa  JVM started by importing network

MAX_INT32 = (2 ** 31) - 1
WALK_CAP_MIN = 30                      # store stops within this walk of a cell (assembly caps tighter)
GRID_M = int(os.environ.get("GRID_M", str(config.GRID_M)))
CACHE = config.DATA / "raptor_cache"
GOLDEN = ROOT / "tests" / "raptor_golden"

# Diverse validation workplaces (lat, lon).
WORKPLACES = {
    "downtown":   (37.7942, -122.3950),   # 1 Market St — dense FiDi, multi-feed hub
    "sunset":     (37.7558, -122.4942),   # Outer Sunset avenues — transit-sparse W
    "bayview":    (37.7299, -122.3890),   # Bayview/Hunters Point — transit-sparse SE
    "westportal": (37.7405, -122.4663),   # West Portal — Muni-metro transfer hub
    "caltrain":   (37.7766, -122.3933),   # 4th & King — near Caltrain / Mission Bay
}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- boot
gtfs = config.gtfs_paths()
svc = feeds.pick_service_date(gtfs)
DEP = config.departure(svc)
log(f"[boot] service date {svc} dep {DEP} window {config.window()} pct {config.PERCENTILES}")
log("[boot] building network (this is the JVM step)...")
t = time.time()
NET = network.build_network(gtfs)
log(f"[boot] network in {time.time()-t:.1f}s")

NEIGH = gridmod.load_neighborhoods()
GRID = gridmod.build_grid(NEIGH, GRID_M)[["id", "geometry"]]
SNAP = GRID.copy()
SNAP["geometry"] = NET.snap_to_network(GRID.geometry)
SNAP = SNAP[SNAP.geometry != shapely.Point()].reset_index(drop=True)
CELL_IDS = list(SNAP.id)
CELL_GEOM = list(SNAP.geometry)
log(f"[boot] grid {len(GRID)} -> snapped {len(SNAP)} on-network")

N_PHYS = os.cpu_count() or 8
THREADS = int(os.environ.get("EXACT_THREADS", str(max(2, min(6, N_PHYS - 2)))))
POOL = ThreadPoolExecutor(max_workers=THREADS)
log(f"[boot] threads={THREADS}")

# RAPTOR gid bridge (JVM-free build of the same structures the engine uses)
RDATA = raptor_build.load_or_build(gtfs, svc, verbose=True)
coord_to_gid = {}
for g in range(RDATA["n_stops"]):
    la, lo = RDATA["stop_lat"][g], RDATA["stop_lon"][g]
    if not np.isnan(la):
        coord_to_gid[(round(float(la), 5), round(float(lo), 5))] = g


def to_gid(lat, lon):
    return coord_to_gid.get((round(float(lat), 5), round(float(lon), 5)))


# ---------------------------------------------------------------- transit stops (R5 layer)
from com.conveyal.r5.streets import VertexStore
TL = NET._transport_network.transitLayer
n_r5 = TL.getStopCount()
slat = np.full(n_r5, np.nan); slon = np.full(n_r5, np.nan)
for i in range(n_r5):
    c = TL.getCoordinateForStopFixed(i)
    if c is not None:
        slon[i] = VertexStore.fixedDegreesToFloating(c.x)
        slat[i] = VertexStore.fixedDegreesToFloating(c.y)
valid = ~np.isnan(slat) & ~np.isnan(slon) & (slat != 0)
r5_idx = np.where(valid)[0]
# bridge each R5 stop -> gid (prefix ids "S" to avoid the id-collision 0-time bug)
stops_gdf = gpd.GeoDataFrame({"id": ["S" + str(i) for i in r5_idx]},
                             geometry=[Point(slon[i], slat[i]) for i in r5_idx], crs=config.WGS)
r5_to_gid = {int(i): to_gid(slat[i], slon[i]) for i in r5_idx}
nbridge = sum(1 for v in r5_to_gid.values() if v is not None)
log(f"[stops] R5 {n_r5} valid {len(r5_idx)} bridged-to-gid {nbridge}")


# ---------------------------------------------------------------- ACCESS table (one-time)
def build_access():
    t = time.time()
    ttm = pd.DataFrame(TravelTimeMatrix(
        NET, origins=SNAP, destinations=stops_gdf, departure=DEP,
        transport_modes=[TransportMode.WALK], speed_walking=config.WALK_KMH,
        max_time=dt.timedelta(minutes=WALK_CAP_MIN), snap_to_network=True))
    tcol = "travel_time" if "travel_time" in ttm.columns else \
        [c for c in ttm.columns if c.startswith("travel_time")][0]
    ttm["from_id"] = ttm["from_id"].astype(str); ttm["to_id"] = ttm["to_id"].astype(str)
    cpos = {c: i for i, c in enumerate(CELL_IDS)}
    spos = {"S" + str(i): int(i) for i in r5_idx}
    rows = [[] for _ in range(len(CELL_IDS))]   # per cell: list of (gid, sec)
    for fr, to, v in zip(ttm["from_id"], ttm["to_id"], ttm[tcol]):
        if pd.isna(v):
            continue
        ci = cpos.get(fr); r5i = spos.get(to)
        if ci is None or r5i is None:
            continue
        g = r5_to_gid.get(r5i)
        if g is None:
            continue
        rows[ci].append((g, int(round(float(v) * 60))))   # minutes -> seconds
    # CSR (dedup to min per gid)
    off = np.zeros(len(rows) + 1, dtype=np.int64)
    to_l, w_l = [], []
    for ci, r in enumerate(rows):
        best = {}
        for g, sec in r:
            if g not in best or sec < best[g]:
                best[g] = sec
        for g, sec in best.items():
            to_l.append(g); w_l.append(sec)
        off[ci + 1] = len(to_l)
    access_to = np.array(to_l, dtype=np.int32)
    access_w = np.array(w_l, dtype=np.int32)
    log(f"[access] {len(CELL_IDS)} cells x {len(r5_idx)} stops in {time.time()-t:.1f}s; "
        f"{len(access_to)} pairs (avg {len(access_to)/max(1,len(CELL_IDS)):.1f}/cell)")
    return off, access_to, access_w


# ---------------------------------------------------------------- per-workplace R5 oracle
template_cache = {}
def forward_oracle(lat, lon):
    """Exact R5 per-cell forward p5/p50 (depart-after window) for the full snapped grid."""
    dest = gpd.GeoDataFrame({"id": ["d"]}, geometry=[Point(lon, lat)], crs=config.WGS)
    tmpl = network.routing_template(NET, dest, DEP)
    def one(i):
        req = copy.copy(tmpl); req.origin = CELL_GEOM[i]
        vals = r5.analyst.TravelTimeComputer(req, NET).computeTravelTimes().travelTimes.getValues()
        b = int(vals[0][0]); rl = int(vals[1][0])
        return (-1 if b == MAX_INT32 else b, -1 if rl == MAX_INT32 else rl)
    res = list(POOL.map(one, range(len(CELL_GEOM))))
    return (np.array([r[0] for r in res], dtype=np.int32),
            np.array([r[1] for r in res], dtype=np.int32))


def egress_purewalk(lat, lon):
    """egress = W->stops walk secs (gid-keyed); purewalk = W->cells walk secs (cell order)."""
    W = gpd.GeoDataFrame({"id": ["W"]}, geometry=[Point(lon, lat)], crs=config.WGS)
    # W -> stops (egress)
    e = pd.DataFrame(TravelTimeMatrix(
        NET, origins=W, destinations=stops_gdf, departure=DEP,
        transport_modes=[TransportMode.WALK], speed_walking=config.WALK_KMH,
        max_time=dt.timedelta(minutes=WALK_CAP_MIN), snap_to_network=True))
    ec = "travel_time" if "travel_time" in e.columns else \
        [c for c in e.columns if c.startswith("travel_time")][0]
    spos = {"S" + str(i): int(i) for i in r5_idx}
    egr = {}
    for to, v in zip(e["to_id"].astype(str), e[ec]):
        if pd.isna(v):
            continue
        g = r5_to_gid.get(spos.get(to, -1))
        if g is not None:
            egr[g] = min(egr.get(g, 1e18), int(round(float(v) * 60)))
    eg = np.array(sorted(egr), dtype=np.int32)
    ew = np.array([egr[g] for g in eg], dtype=np.int32)
    # W -> cells (purewalk), cap at MAX_MIN
    pw_ttm = pd.DataFrame(TravelTimeMatrix(
        NET, origins=W, destinations=SNAP, departure=DEP,
        transport_modes=[TransportMode.WALK], speed_walking=config.WALK_KMH,
        max_time=dt.timedelta(minutes=config.MAX_MIN), snap_to_network=True))
    pc = "travel_time" if "travel_time" in pw_ttm.columns else \
        [c for c in pw_ttm.columns if c.startswith("travel_time")][0]
    cpos = {c: i for i, c in enumerate(CELL_IDS)}
    pw = np.full(len(CELL_IDS), -1, dtype=np.int32)
    for to, v in zip(pw_ttm["to_id"].astype(str), pw_ttm[pc]):
        if pd.isna(v):
            continue
        i = cpos.get(to)
        if i is not None:
            pw[i] = int(round(float(v) * 60))
    return eg, ew, pw


# ---------------------------------------------------------------- run
CACHE.mkdir(parents=True, exist_ok=True)
GOLDEN.mkdir(parents=True, exist_ok=True)

off, access_to, access_w = build_access()
fp = raptor_build._fingerprint(gtfs, svc.strftime("%Y%m%d"),
                               raptor_build.band_seconds(), raptor_build.FOOTPATH_M)
acc_path = CACHE / f"access_{GRID_M}m_{fp}.npz"
np.savez(acc_path, cell_ids=np.array(CELL_IDS), access_off=off, access_to=access_to,
         access_w=access_w, grid_m=GRID_M, n_stops=RDATA["n_stops"], raptor_fp=fp,
         service_date=svc.strftime("%Y%m%d"))
log(f"[access] saved {acc_path.name} ({acc_path.stat().st_size/1e6:.1f} MB)")

if os.environ.get("ONLY_ACCESS"):
    log("ONLY_ACCESS set: skipping R5 oracle regeneration (oracles are fingerprint-independent).")
    sys.exit(0)

for name, (lat, lon) in WORKPLACES.items():
    t = time.time()
    best, p50 = forward_oracle(lat, lon)
    eg, ew, pw = egress_purewalk(lat, lon)
    nreach = int((p50 >= 0).sum())
    out = GOLDEN / f"oracle_{name}.npz"
    np.savez(out, name=name, lat=lat, lon=lon, cell_ids=np.array(CELL_IDS),
             best=best, p50=p50, egress_g=eg, egress_w=ew, purewalk=pw,
             service_date=svc.strftime("%Y%m%d"))
    log(f"[oracle {name}] ({lat},{lon}) {time.time()-t:.1f}s  reachable {nreach}/{len(CELL_IDS)} "
        f"egress-stops {len(eg)} -> {out.name}")

log("\nDONE. access table + 5 workplace oracles written.")
