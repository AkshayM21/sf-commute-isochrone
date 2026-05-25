"""Is the RAPTOR path layer's R5 disagreement a BUG or the arrive-by anchor?

For a sample of cells, take RAPTOR's chosen latest-home departure D* and run R5 forward at the
SAME D* (single departure, recorded path). If RAPTOR's route == R5's route at the same D*, the
engine's routing is correct and the color-by-line gap vs the depart-after oracle is purely the
arrive-by-vs-depart-after anchor (different question), not a bug.
"""
import os, sys
import datetime as dt
from pathlib import Path
_mem = os.environ.get("R5_MAX_MEMORY")
if _mem and "--max-memory" not in sys.argv:
    sys.argv += ["--max-memory", _mem]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import server  # boots R5
from core import config, raptor_engine
import com.conveyal.r5 as r5

GOLDEN = ROOT / "tests" / "raptor_golden"
NAME = os.environ.get("WP", "downtown")
N = int(os.environ.get("N", "40"))

eng = raptor_engine.RaptorEngine(verbose=False)
z = np.load(GOLDEN / f"oracle_{NAME}.npz", allow_pickle=True)
lat, lon = float(z["lat"]), float(z["lon"])
pw = np.array([int(z["purewalk"][i]) for i in range(len(eng.cell_ids))], np.int64)
tree = eng.journey_tree(z["egress_g"], z["egress_w"], pw)
commute, dom = tree.commute_and_dominant()


def r5_route_at(olat, olon, dep_dt):
    """R5 recorded-path dominant route for a single departure dep_dt (window 0)."""
    o = server.NET.snap_to_network(gpd.GeoSeries([Point(olon, olat)], crs=config.WGS)).iloc[0]
    if o.is_empty:
        return None
    dest = gpd.GeoDataFrame({"id": ["d"]}, geometry=[Point(lon, lat)], crs=config.WGS)
    from core.network import _common
    from r5py.r5.regional_task import RegionalTask
    import datetime as _dt
    t = RegionalTask(server.NET, origin=o, destinations=dest, departure=dep_dt,
                     departure_time_window=_dt.timedelta(minutes=1),
                     max_time=_dt.timedelta(minutes=config.MAX_MIN),
                     transport_modes=server.network.MODES, percentiles=config.PERCENTILES,
                     speed_walking=config.WALK_KMH, max_public_transport_rides=8)
    t.destinations = dest
    t._regional_task.includePathResults = True
    t._regional_task.nPathsPerTarget = 8
    it = server._recorded_itin(t)
    if not it:
        return None, it
    return server._dominant_line(it), it


svc = eng.service_date
sample = [i for i in range(0, len(eng.cell_ids), max(1, len(eng.cell_ids) // N))][:N]
same = checked = 0
print(f"workplace={NAME}  comparing RAPTOR route vs R5 AT THE SAME departure D*")
print(f"{'cell':>5} {'D*':>6} {'RAPTOR':<14} {'R5@D*':<14} {'match'}")
for i in sample:
    if commute[i] < 0:
        continue
    tr = tree._trace(i)
    if tr is None:
        continue
    legs_raw, latest_home = tr
    olat, olon = server.ORIGIN_LL[eng.cell_ids[i]]
    dstar = dt.datetime(svc.year, svc.month, svc.day) + dt.timedelta(seconds=int(latest_home))
    r5res = r5_route_at(olat, olon, dstar)
    if r5res is None or r5res[0] is None:
        continue
    r5line = r5res[0]
    mine = dom[i]
    checked += 1
    ok = (mine == r5line)
    same += ok
    print(f"{eng.cell_ids[i]:>5} {latest_home//60%60:>2}m  {str(mine):<14} {str(r5line):<14} {'OK' if ok else 'x'}")
print(f"\nSAME-ANCHOR agreement: {same}/{checked} = {same/max(1,checked)*100:.0f}%")
print("(High => engine routing correct; the depart-after-oracle gap is the arrive-by anchor.)")
