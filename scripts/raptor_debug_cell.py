"""Clock-level diff of ONE cell: RAPTOR's traced arrive-by journey vs R5's recorded path at the
SAME departure D*. Reveals whether RAPTOR's ~6-min same-departure optimism is a fixable
feasibility/walk discrepancy or inherent best-case timing.

NOTE: this diagnostic NEEDS the JVM for the R5 side of the diff — under the default JVM-free
boot server.NET is None, so we force the R5 path before importing server. The script's own
RaptorEngine is constructed directly and reads no USE_* flags, so it is unaffected.
"""
import os, sys
import datetime as dt
from pathlib import Path
_mem = os.environ.get("R5_MAX_MEMORY")
if _mem and "--max-memory" not in sys.argv:
    sys.argv += ["--max-memory", _mem]
# Force the R5 boot (server._NEED_R5): this script NEEDS server.NET for the R5 half of the diff.
os.environ["USE_WALK_GRAPH"] = "0"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import server
from core import config, network, raptor_engine, raptor_golden

GOLDEN = ROOT / "tests" / "raptor_golden"
NAME = os.environ.get("WP", "downtown")
CELL = os.environ.get("CELL", "49")           # default: the same-line L 10-min-gap cell

eng = raptor_engine.RaptorEngine(verbose=False)
z = np.load(GOLDEN / f"oracle_{NAME}.npz", allow_pickle=True)
lat, lon = float(z["lat"]), float(z["lon"])
pw = raptor_golden.purewalk_aligned(eng, z)
tree = eng.journey_tree(z["egress_g"], z["egress_w"], pw)
ci = eng.cell_index[CELL]
tr = tree._trace(ci)
legs_raw, latest_home = tr
my = tree.itinerary(ci)


def hm(s):
    return f"{int(s)//3600:02d}:{int(s)%3600//60:02d}:{int(s)%60:02d}"


print(f"=== cell {CELL} (workplace {NAME}) ===")
print(f"RAPTOR latest_home D* = {hm(latest_home)}  commute = {my['total']}m")
print("RAPTOR raw legs (exact seconds):")
t = latest_home
for leg in legs_raw:
    if leg[0] == "ride":
        # 7-element ride tuple ("ride", pi, dep, arr, bpos, apos, alight_stop) — *_rest keeps
        # this resilient to future appends (matches raptor_journey's positional readers)
        _, pi, dep, arr, *_rest = leg
        feed, rid, name, mode = eng.data["line_table"][int(eng.data["pat_line"][pi])]
        print(f"   wait {hm(t)}->{hm(dep)} ({(dep-t)//60}m)  RIDE {name} {hm(dep)}->{hm(arr)} ({(arr-dep)//60}m)")
        t = arr
    else:
        print(f"   {leg[0]:7} {hm(t)}->{hm(t+leg[1])} ({leg[1]//60}m {leg[1]%60}s)")
        t += leg[1]
print(f"   RAPTOR arrival at W = {hm(t)}  (deadline {hm(eng.target_sec)})")

# R5 from the same origin at the same D*
olat, olon = server.ORIGIN_LL[CELL]
dstar = dt.datetime(eng.service_date.year, eng.service_date.month, eng.service_date.day) \
    + dt.timedelta(seconds=int(latest_home))
o = server.NET.snap_to_network(gpd.GeoSeries([Point(olon, olat)], crs=config.WGS)).iloc[0]
dest = gpd.GeoDataFrame({"id": ["d"]}, geometry=[Point(lon, lat)], crs=config.WGS)
# canonical params from the ONE template (single-departure window for the probe)
tk = network.routing_template(server.NET, dest, dstar, paths=True,
                              window=dt.timedelta(minutes=1))
tk.origin = o
r5it = server._recorded_itin(tk)
print(f"\nR5 @ same D*={hm(latest_home)}: total p50 = {r5it['total']}m")
print("R5 legs:")
for l in r5it["legs"]:
    if l["mode"] == "transit":
        print(f"   wait {l.get('wait',0)}m  RIDE {l['line']} {l['min']}m")
    else:
        print(f"   walk {l['min']}m")
print(f"\nDelta = RAPTOR {my['total']} - R5 {r5it['total']} = {my['total']-r5it['total']}m")
print("If RAPTOR rides/walks match R5 but RAPTOR waits are ~0 while R5 waits a headway ->")
print("  inherent best-case PERFECT-TIMING. If RAPTOR walks are SHORTER -> baked-walk discrepancy.")
