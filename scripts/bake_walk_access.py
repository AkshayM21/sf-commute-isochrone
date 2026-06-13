"""Bake the cell->stop access table from the JVM-free walk router (replaces the R5 step).

Produces ``data/raptor_cache/access_walk[flat]_<gridm>m_<fp>.npz`` in the SAME CSR format the
engine consumes (cell_ids, access_off, access_to[gid], access_w[REFERENCE sec]), but from the
hill-aware pedestrian graph instead of an R5 walk matrix. Reference seconds at config.WALK_KMH;
the per-request speed scalar is applied by the engine.

Then validates end-to-end: runs the engine with the walk-baked access table (keeping each oracle's
R5 egress/pure-walk, to ISOLATE the access-table swap) and reports door-to-door p50 MAE vs the R5
oracle. WALK_FLAT=1 bakes the grade-agnostic table (mechanics check); default bakes the hill table.

Usage: .venv/bin/python scripts/bake_walk_access.py   [WALK_FLAT=1]
"""
import hashlib, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from core import config, grid as gridmod, feeds, raptor_build, walk, raptor_engine, raptor_golden

FLAT = os.environ.get("WALK_FLAT", "").lower() in ("1", "true", "yes")
GRID_M = int(os.environ.get("GRID_M", str(config.GRID_M)))
CACHE = config.DATA / "raptor_cache"
GOLDEN = ROOT / "tests" / "raptor_golden"
WALK_CAP_MIN = 30                          # match the R5 access bake; engine caps tighter at load


def log(*a):
    print(*a, flush=True)


def bake():
    gtfs = config.gtfs_paths()
    svc = feeds.pick_service_date(gtfs)
    data = raptor_build.load_or_build(gtfs, svc, verbose=False)
    n_stops = data["n_stops"]
    slat, slon = data["stop_lat"], data["stop_lon"]

    # canonical cell set/order = the existing R5 access table's cell_ids (apples-to-apples)
    fp = raptor_build._fingerprint(gtfs, svc.strftime("%Y%m%d"),
                                   raptor_build.band_seconds(), raptor_build.FOOTPATH_M)
    r5_acc = sorted(CACHE.glob(f"access_{GRID_M}m_{fp}.npz"))
    if not r5_acc:
        sys.exit(f"no R5 access table access_{GRID_M}m_{fp}.npz to take the cell set from")
    cell_ids = list(np.load(r5_acc[0], allow_pickle=True)["cell_ids"].astype(str))

    # cell centroids from the deterministic grid (id -> lon/lat)
    g = gridmod.build_grid(gridmod.load_neighborhoods(), GRID_M)
    id2ll = {str(i): (float(geom.x), float(geom.y)) for i, geom in zip(g["id"], g.geometry)}
    cell_ll = np.array([id2ll[c] for c in cell_ids], dtype=np.float64)

    wg = walk.WalkGraph.load()
    # fingerprint the graph these weights come from, so a walk_graph.npz rebuild (OSM repull,
    # DEM refetch, stairs-mult change) without a rebake is detectable at server boot
    graph_sha = hashlib.sha256((config.DATA / "walk_graph.npz").read_bytes()).hexdigest()
    valid = np.where(~np.isnan(slat))[0]                       # gids with coords
    stop_nodes, stop_conn = wg.snap(np.column_stack((slon[valid], slat[valid])))
    cell_nodes, cell_conn = wg.snap(cell_ll)

    t = time.time()
    cap_ref = WALK_CAP_MIN * 60
    M = wg.many_to_targets(cell_nodes, cell_conn, stop_nodes, stop_conn, cap_ref, flat=FLAT)
    log(f"[bake] {len(cell_ids)} cells x {len(valid)} stops in {time.time()-t:.1f}s "
        f"({'flat' if FLAT else 'hill'} weights)")

    off = np.zeros(len(cell_ids) + 1, dtype=np.int64)
    to_l, w_l = [], []
    for ci in range(len(cell_ids)):
        row = M[ci]
        keep = np.where(np.isfinite(row))[0]
        for j in keep:
            to_l.append(int(valid[j])); w_l.append(int(round(float(row[j]))))
        off[ci + 1] = len(to_l)
    access_to = np.array(to_l, dtype=np.int32); access_w = np.array(w_l, dtype=np.int32)
    log(f"[bake] {len(access_to)} pairs (avg {len(access_to)/len(cell_ids):.1f}/cell)")

    name = f"access_walk{'flat' if FLAT else ''}_{GRID_M}m_{fp}.npz"
    out = CACHE / name
    # write tmp + atomic rename so an interrupted bake can't leave a truncated zip at the
    # canonical (fingerprinted) path the engine's existence check trusts. Tmp name must end
    # in .npz — np.savez silently appends it otherwise.
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, cell_ids=np.array(cell_ids), access_off=off, access_to=access_to,
             access_w=access_w, grid_m=GRID_M, n_stops=n_stops, raptor_fp=fp,
             service_date=svc.strftime("%Y%m%d"), slope_aware=np.int8(0 if FLAT else 1),
             walk_ref_kmh=np.float32(config.WALK_KMH), walk_graph_sha=graph_sha)
    np.load(tmp, allow_pickle=True).close()      # cheap zip-integrity check before publish
    os.replace(tmp, out)
    log(f"[bake] saved {out.name} ({out.stat().st_size/1e6:.1f} MB)")
    return out


def validate(acc_path):
    eng = raptor_engine.RaptorEngine(access_path=acc_path, verbose=False)
    oracles = sorted(GOLDEN.glob("oracle_*.npz"))
    pos = {c: i for i, c in enumerate(eng.cell_ids)}
    all_err, all_signed = [], []
    log(f"\n{'workplace':<12}{'n':>6}{'MAE':>7}{'bias':>7}{'p95':>6}{'max':>5}  "
        f"(door-to-door p50 vs R5, walk-baked access + R5 egress)")
    for op in oracles:
        z = np.load(op, allow_pickle=True)
        pw = raptor_golden.purewalk_aligned(eng, z)
        res = eng.departafter(z["egress_g"], z["egress_w"], pw)
        p50 = [res[c][1] for c in eng.cell_ids]
        errs, signed = [], []
        for k, cid in enumerate(z["cell_ids"].astype(str)):
            i = pos.get(cid)
            if i is None:
                continue
            mine = p50[i]; mine = -1 if mine is None else int(mine); r5 = int(z["p50"][k])
            if r5 < 0 or mine < 0:
                continue
            errs.append(abs(mine - r5)); signed.append(mine - r5)
            all_err.append(abs(mine - r5)); all_signed.append(mine - r5)
        errs = np.array(errs); signed = np.array(signed)
        log(f"{str(z['name']):<12}{len(errs):>6}{errs.mean():>7.2f}{signed.mean():>+7.2f}"
            f"{np.percentile(errs,95):>6.1f}{errs.max():>5}")
    ae = np.array(all_err); asg = np.array(all_signed)
    log(f"\n{'AGGREGATE':<12}{len(ae):>6}{ae.mean():>7.2f}{asg.mean():>+7.2f}"
        f"{np.percentile(ae,95):>6.1f}{ae.max():>5}")
    log(f"(R5-access baseline was MAE 0.75; {'flat' if FLAT else 'hill'}-walk access shown above. "
        f"hill is EXPECTED to read longer on steep cells = more accurate, not an R5 match.)")


if __name__ == "__main__":
    validate(bake())
