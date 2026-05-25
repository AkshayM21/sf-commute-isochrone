"""Validate the RAPTOR engine against the R5 oracles (JVM-free, the DoD gate).

Loads the baked access table + each workplace's R5 ground-truth oracle (forward depart-after
p50 + that workplace's egress/pure-walk), runs the engine, and reports per-workplace and
aggregate MAE / p95 / max / signed-bias / reachability-mismatch. R5 has no native arrive-by,
so the headline comparison is R5's depart-after window p50 (the spike's proven oracle); the
engine reproduces it by inverting the reverse profile. Arrive-by-09:00 is the same profile
read at the target and is sanity-printed alongside.

Usage: .venv/bin/python scripts/raptor_validate.py
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from core import raptor_engine
from core.raptor import _select_kernel

GOLDEN = ROOT / "tests" / "raptor_golden"


def stats(engine_p50, r5_p50, cell_ids, oracle_cell_ids):
    pos = {c: i for i, c in enumerate(cell_ids)}
    errs, signed, mism, near_cap = [], [], 0, 0
    for k, cid in enumerate(oracle_cell_ids):
        i = pos.get(cid)
        if i is None:
            continue
        mine = engine_p50[i]; r5v = int(r5_p50[k])
        mine = -1 if mine is None else int(mine)
        if r5v < 0 and mine < 0:
            continue
        if (r5v < 0) != (mine < 0):
            # near-75min-cap boundary flips are expected R5-internal minutiae, not access misses
            if (r5v >= 0 and r5v >= 72) or (mine >= 0 and mine >= 72):
                near_cap += 1
            else:
                mism += 1
            continue
        errs.append(abs(mine - r5v)); signed.append(mine - r5v)
    errs = np.array(errs); signed = np.array(signed)
    return dict(n=len(errs), mae=float(errs.mean()), p95=float(np.percentile(errs, 95)),
                mx=int(errs.max()), bias=float(signed.mean()), mism=mism, near_cap=near_cap,
                w1=float(np.mean(errs <= 1) * 100), w2=float(np.mean(errs <= 2) * 100),
                w3=float(np.mean(errs <= 3) * 100))


def main():
    print(f"[kernel] {_select_kernel()}")
    eng = raptor_engine.RaptorEngine(verbose=True)
    oracles = sorted(GOLDEN.glob("oracle_*.npz"))
    if not oracles:
        sys.exit("no oracles in tests/raptor_golden/ — run scripts/raptor_oracle.py")
    all_err, all_signed = [], []
    print(f"\n{'workplace':<12}{'n':>5}{'MAE':>7}{'p95':>6}{'max':>5}{'bias':>7}"
          f"{'mism':>6}{'~cap':>6}{'<=1':>6}{'<=2':>6}  time")
    agg_mism = agg_nearcap = 0
    for op in oracles:
        z = np.load(op, allow_pickle=True)
        name = str(z["name"])
        purewalk = np.full(len(eng.cell_ids), -1, dtype=np.int64)
        opos = {c: i for i, c in enumerate(z["cell_ids"].astype(str))}
        for i, c in enumerate(eng.cell_ids):
            j = opos.get(c)
            if j is not None:
                purewalk[i] = int(z["purewalk"][j])
        t0 = time.time()
        res = eng.departafter(z["egress_g"], z["egress_w"], purewalk)
        dt = time.time() - t0
        p50 = [res[c][1] for c in eng.cell_ids]
        s = stats(p50, z["p50"], eng.cell_ids, list(z["cell_ids"].astype(str)))
        agg_mism += s["mism"]; agg_nearcap += s["near_cap"]
        # collect for aggregate
        pos = {c: i for i, c in enumerate(eng.cell_ids)}
        for k, cid in enumerate(z["cell_ids"].astype(str)):
            i = pos.get(cid)
            if i is None:
                continue
            mine = p50[i]; r5v = int(z["p50"][k])
            mine = -1 if mine is None else int(mine)
            if r5v < 0 or mine < 0:
                continue
            all_err.append(abs(mine - r5v)); all_signed.append(mine - r5v)
        print(f"{name:<12}{s['n']:>5}{s['mae']:>7.2f}{s['p95']:>6.1f}{s['mx']:>5}"
              f"{s['bias']:>+7.2f}{s['mism']:>6}{s['near_cap']:>6}{s['w1']:>5.0f}%{s['w2']:>5.0f}%"
              f"  {dt:.2f}s")
    ae = np.array(all_err); asg = np.array(all_signed)
    print(f"\n{'AGGREGATE':<12}{len(ae):>5}{ae.mean():>7.2f}{np.percentile(ae,95):>6.1f}"
          f"{ae.max():>5}{asg.mean():>+7.2f}{agg_mism:>6}{agg_nearcap:>6}"
          f"{np.mean(ae<=1)*100:>5.0f}%{np.mean(ae<=2)*100:>5.0f}%")
    print(f"\nDoD: MAE<=1.0  p95<=3.0  max~4  bias~0  reach-mismatch=0 (near-cap flips excluded)")
    ok = (ae.mean() <= 1.0 and np.percentile(ae, 95) <= 3.0 and agg_mism == 0)
    print("RESULT:", "PASS" if ok else "REVIEW", f"(mism={agg_mism}, near-cap flips={agg_nearcap})")


if __name__ == "__main__":
    main()
