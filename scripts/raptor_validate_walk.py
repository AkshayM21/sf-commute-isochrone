"""Validate the JVM-free walk router against R5's walk matrix (the egress W->stops oracle).

W->stops is the confound-free check: R5 and our router use the SAME exact GTFS stop coords and
the SAME workplace point (cell comparisons would add a snapping offset R5's snapped origins hide).
Two questions:
  1. MECHANICS — does our FLAT weight (grade-agnostic distance/4.8, R5's model) reproduce R5's
     walk times? (validates OSM parse + snapping + Dijkstra). Expect small MAE, ~0 bias.
  2. HILL IMPROVEMENT — does the grade-aware weight deviate in the RIGHT direction? On flat-grade
     trips it should still ~match R5; on STEEP trips it should be >= R5 (R5 ignores grade).

Usage: .venv/bin/python scripts/raptor_validate_walk.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from core import config, raptor_build, walk

GOLDEN = ROOT / "tests" / "raptor_golden"
CAP_REF = 30 * 60          # R5 oracle WALK_CAP_MIN


def main():
    wg = walk.WalkGraph.load()
    data = raptor_build.load_or_build(verbose=False)
    slat, slon = data["stop_lat"], data["stop_lon"]
    n_stops = data["n_stops"]
    valid = ~np.isnan(slat)
    gids = np.where(valid)[0]
    snode, sconn = wg.snap(np.column_stack((slon[gids], slat[gids])))   # [T,k]
    node_elev = wg.elev
    stop_node_elev = node_elev[snode[:, 0]]            # nearest-node elevation (grade proxy)
    gpos = {int(g): i for i, g in enumerate(gids)}     # gid -> row in the snapped arrays

    oracles = sorted(GOLDEN.glob("oracle_*.npz"))
    flat_err, hill_err, hill_sign = [], [], []
    seg_flat_err, seg_steep_sign = [], []
    for op in oracles:
        z = np.load(op, allow_pickle=True)
        wlat, wlon = float(z["lat"]), float(z["lon"])
        wnode, _ = wg.snap([(wlon, wlat)]); w_elev = float(node_elev[int(wnode[0, 0])])
        flat = wg.one_to_many((wlon, wlat), snode, sconn, CAP_REF, flat=True)
        hill = wg.one_to_many((wlon, wlat), snode, sconn, CAP_REF, flat=False)
        for g, r5sec in zip(np.asarray(z["egress_g"]), np.asarray(z["egress_w"])):
            i = gpos.get(int(g))
            if i is None:
                continue
            r5 = float(r5sec)
            if np.isfinite(flat[i]):
                flat_err.append(flat[i] - r5)
            if np.isfinite(hill[i]):
                hill_err.append(abs(hill[i] - r5)); hill_sign.append(hill[i] - r5)
                # grade proxy: |elev diff| / straight-line distance
                dx = (slon[int(g)] - wlon) * wg.mlon; dy = (slat[int(g)] - wlat) * wg.mlat
                dist = max(1.0, (dx * dx + dy * dy) ** 0.5)
                grade = abs(stop_node_elev[i] - w_elev) / dist
                if grade < 0.03 and np.isfinite(flat[i]):
                    seg_flat_err.append(abs(flat[i] - r5))
                elif grade > 0.08:
                    seg_steep_sign.append(hill[i] - r5)

    fe = np.array(flat_err); he = np.array(hill_err); hs = np.array(hill_sign)
    print(f"\nW->stops vs R5 over {len(oracles)} workplaces, {len(fe)} stop-pairs:")
    print(f"  FLAT (mechanics):  MAE {np.abs(fe).mean():6.1f}s  bias {fe.mean():+6.1f}s  "
          f"median|err| {np.median(np.abs(fe)):5.1f}s  within 30s {np.mean(np.abs(fe)<=30)*100:.0f}%")
    print(f"  HILL (production): MAE {he.mean():6.1f}s  bias {hs.mean():+6.1f}s  "
          f"(positive = ours longer on grade, the intended improvement)")
    sfe = np.array(seg_flat_err); sss = np.array(seg_steep_sign)
    if len(sfe):
        print(f"  flat-grade (<3%) trips: FLAT MAE {sfe.mean():.1f}s  (n={len(sfe)})")
    if len(sss):
        print(f"  steep-grade (>8%) trips: HILL bias {sss.mean():+.1f}s, "
              f"ours>=R5 in {np.mean(sss>=-5)*100:.0f}% (n={len(sss)})")
    ok = (np.abs(fe).mean() <= 45 and abs(fe.mean()) <= 20 and (len(sss) == 0 or np.mean(sss >= -5) >= 0.8))
    print("\nRESULT:", "PASS" if ok else "REVIEW",
          f"(flat MAE {np.abs(fe).mean():.0f}s, flat bias {fe.mean():+.0f}s)")


if __name__ == "__main__":
    main()
