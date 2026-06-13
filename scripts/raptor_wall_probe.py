"""Definitive probe of the Phase-2 route wall: at RAPTOR's OWN chosen departure D*, is RAPTOR's
traced journey as FAST as R5's earliest-arrival journey, or slower?

  - same total  -> RAPTOR found an equal-time route; disagreement is just route ambiguity (artifact)
  - R5 faster   -> RAPTOR's trace is suboptimal in travel time (a real gap: the reverse
                   latest-departure tree reconstructs a valid-but-not-fastest journey)

Prints concrete side-by-side examples + the distribution of (RAPTOR_total - R5_total).

NOTE: this probe NEEDS the JVM — the R5 side of the diff uses server.NET + recorded paths,
which are None under the default JVM-free boot (USE_RAPTOR=1 USE_WALK_GRAPH=1 arriveby).
We force the R5 path below by defaulting USE_WALK_GRAPH=0 before importing server.
"""
import os, sys
import datetime as dt
from pathlib import Path
_mem = os.environ.get("R5_MAX_MEMORY")
if _mem and "--max-memory" not in sys.argv:
    sys.argv += ["--max-memory", _mem]
# Force the R5 boot (server._NEED_R5) BEFORE `import server`: under the default JVM-free env
# server.NET/server.network are None and every R5 call below would AttributeError. setdefault
# respects an explicit override; the assert after import catches a still-lean env loudly.
os.environ.setdefault("USE_WALK_GRAPH", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import server
assert server._NEED_R5, ("this probe needs the R5/JVM boot (server.NET) — "
                         "don't force USE_WALK_GRAPH=1/RAPTOR_SEMANTIC=arriveby for it")
from core import config, network, raptor_engine, raptor_golden

GOLDEN = ROOT / "tests" / "raptor_golden"
NAME = os.environ.get("WP", "downtown")
N = int(os.environ.get("N", "60"))

eng = raptor_engine.RaptorEngine(verbose=False)
z = np.load(GOLDEN / f"oracle_{NAME}.npz", allow_pickle=True)
lat, lon = float(z["lat"]), float(z["lon"])
pw = raptor_golden.purewalk_aligned(eng, z)
tree = eng.journey_tree(z["egress_g"], z["egress_w"], pw)
commute, dom = tree.commute_and_dominant()
svc = eng.service_date


import com.conveyal.r5 as r5
MAX_INT32 = (2 ** 31) - 1
def r5_itin_at(olat, olon, dep_dt):
    o = server.NET.snap_to_network(gpd.GeoSeries([Point(olon, olat)], crs=config.WGS)).iloc[0]
    if o.is_empty:
        return None
    dest = gpd.GeoDataFrame({"id": ["d"]}, geometry=[Point(lon, lat)], crs=config.WGS)
    # canonical params from the ONE template (single-departure window for the probe)
    t = network.routing_template(server.NET, dest, dep_dt, paths=True,
                                 window=dt.timedelta(minutes=1))
    t.origin = o
    result = r5.analyst.TravelTimeComputer(t, server.NET).computeTravelTimes()
    vals = result.travelTimes.getValues()
    p5 = int(vals[0][0]); p50 = int(vals[1][0])
    if p50 == MAX_INT32:
        return None
    it = server._recorded_itin(t)
    if it:
        it["_p5"] = (None if p5 == MAX_INT32 else p5)
    return it


def seq(it):
    return " ".join(l["line"] for l in it["legs"] if l["mode"] == "transit") or "walk"


sample = [i for i in range(0, len(eng.cell_ids), max(1, len(eng.cell_ids) // N))][:N]
deltas, deltas5, eq, r5faster, mefaster = [], [], 0, 0, 0
examples = []
for i in sample:
    if commute[i] < 0:
        continue
    tr = tree._trace(i)
    if tr is None:
        continue
    legs_raw, latest_home = tr
    my = tree.itinerary(i)
    olat, olon = server.ORIGIN_LL[eng.cell_ids[i]]
    dstar = dt.datetime(svc.year, svc.month, svc.day) + dt.timedelta(seconds=int(latest_home))
    r5it = r5_itin_at(olat, olon, dstar)
    if not r5it:
        continue
    d = my["total"] - r5it["total"]            # >0 => RAPTOR slower than R5 from the SAME departure
    d5 = my["total"] - r5it["_p5"] if r5it.get("_p5") is not None else None  # vs R5 best-case
    if d5 is not None:
        deltas5.append(d5)
    deltas.append(d)
    if d == 0:
        eq += 1
    elif d > 0:
        r5faster += 1
    else:
        mefaster += 1
    if len(examples) < 8 and (d != 0 or seq(my) != seq(r5it)):
        examples.append((eng.cell_ids[i], int(latest_home), my, r5it, d))

print(f"workplace={NAME}  departure = RAPTOR's own latest-feasible D*  (n={len(deltas)})")
print("\nExamples (RAPTOR vs R5 from the SAME departure):")
for cid, lh, my, r5b, d in examples:     # r5b, NOT r5: that name is the com.conveyal.r5 module
    tag = "TIE (route ambiguity)" if d == 0 else (f"R5 faster by {d}m (RAPTOR suboptimal)" if d > 0
                                                  else f"RAPTOR faster by {-d}m")
    print(f"  cell {cid}: RAPTOR {my['total']}m [{seq(my)}]  |  R5 {r5b['total']}m [{seq(r5b)}]  -> {tag}")
deltas = np.array(deltas)
print(f"\nSame-departure travel-time delta (RAPTOR - R5):")
print(f"  equal time (route ambiguity only): {eq}/{len(deltas)} = {eq/len(deltas)*100:.0f}%")
print(f"  R5 strictly faster (RAPTOR trace suboptimal): {r5faster}/{len(deltas)} = {r5faster/len(deltas)*100:.0f}%")
print(f"  RAPTOR faster: {mefaster}")
print(f"  mean delta {deltas.mean():+.2f}m  p95 {np.percentile(deltas,95):.0f}m  max {deltas.max()}m")
d5 = np.array(deltas5)
print(f"\nvs R5 BEST-CASE (p5) at the same departure (RAPTOR - R5_p5):")
print(f"  mean {d5.mean():+.2f}m  |  within 1m: {np.mean(np.abs(d5)<=1)*100:.0f}%  within 2m: {np.mean(np.abs(d5)<=2)*100:.0f}%")
print("  (near 0 => RAPTOR arrive-by-latest == R5 best-case: the gap vs p50 is the WAIT percentile,")
print("   i.e. a depart-vs-arrive/percentile semantic, NOT a missing/sub-optimal route)")
