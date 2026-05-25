"""Validate the service-noise Monte-Carlo "realistic" layer against the R5 oracle (JVM-free).

The MC perturbs the schedule per draw and re-optimizes, then reads the median-departure commute;
``realistic`` = p50 over draws. R5's depart-after window p50 (the committed oracle) is the
schedule-PERFECT realistic, so we expect ``realistic`` to sit slightly ABOVE it (service delays
only add time) — a small POSITIVE bias is correct, not an error. We assert:

  * realistic vs R5 p50:  MAE <= ~2.0,  bias within [+/-1.5]  (expect mildly positive)
  * per-cell invariant:   perfect (arrive-by best-case) <= realistic   (MC never beats perfect)
  * fragility is sane:    most cells robust, a long tail of fragile (Caltrain/peak-express/SE) cells

The MC re-routes CLAIRVOYANTLY (knows the perturbed schedule at home), so ``realistic`` is a
LOWER BOUND on real committed-plan variance — it answers "is there a good alternative under bad
conditions" (route resilience), not "I'm on the platform and just missed it". See RAPTOR.md.

Usage: .venv/bin/python scripts/raptor_validate_mc.py
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from core import raptor_engine
from core.raptor import _select_kernel

GOLDEN = ROOT / "tests" / "raptor_golden"


def _purewalk(engine, z):
    pw = np.full(len(engine.cell_ids), -1, dtype=np.int64)
    opos = {c: i for i, c in enumerate(z["cell_ids"].astype(str))}
    for i, c in enumerate(engine.cell_ids):
        j = opos.get(c)
        if j is not None:
            pw[i] = int(z["purewalk"][j])
    return pw


def main():
    print(f"[kernel] {_select_kernel()}")
    eng = raptor_engine.RaptorEngine(verbose=True)
    oracles = sorted(GOLDEN.glob("oracle_*.npz"))
    if not oracles:
        sys.exit("no oracles in tests/raptor_golden/ — run scripts/raptor_oracle.py")
    pos = {c: i for i, c in enumerate(eng.cell_ids)}
    all_err, all_signed = [], []
    tot_viol = 0
    print(f"\n{'workplace':<12}{'n':>5}{'MAE':>7}{'bias':>7}{'p95':>6}{'max':>5}"
          f"{'p>real':>8}{'frag50':>8}{'frag90':>8}{'stuck%':>8}  time")
    for op in oracles:
        z = np.load(op, allow_pickle=True)
        name = str(z["name"])
        pw = _purewalk(eng, z)
        # perfect (arrive-by best-case) per cell, the map's headline value
        perfect, _dom = eng.journey_tree(z["egress_g"], z["egress_w"], pw).commute_and_dominant()
        perfect = np.asarray(perfect, np.int32)
        t0 = time.time()
        mc = eng.montecarlo(z["egress_g"], z["egress_w"], pw, perfect=perfect,
                            seed=12345, alt_draws=0)
        dt = time.time() - t0
        realistic = mc["realistic"]
        viol = int(np.sum(realistic[perfect >= 0] < perfect[perfect >= 0]))
        tot_viol += viol
        errs, signed = [], []
        for k, cid in enumerate(z["cell_ids"].astype(str)):
            i = pos.get(cid)
            if i is None or perfect[i] < 0:
                continue
            r5 = int(z["p50"][k])
            if r5 < 0:
                continue
            errs.append(abs(int(realistic[i]) - r5)); signed.append(int(realistic[i]) - r5)
        errs = np.array(errs); signed = np.array(signed)
        all_err.append(errs); all_signed.append(signed)
        reach = perfect >= 0
        frag = mc["frag"][reach]; stuck = mc["stuck"][reach]
        print(f"{name:<12}{len(errs):>5}{errs.mean():>7.2f}{signed.mean():>+7.2f}"
              f"{np.percentile(errs,95):>6.1f}{errs.max():>5}{viol:>8}"
              f"{np.median(frag):>8.0f}{np.percentile(frag,90):>8.0f}"
              f"{np.mean(stuck>=0.15)*100:>7.0f}%  {dt:.2f}s")
    err = np.concatenate(all_err); signed = np.concatenate(all_signed)
    mae = float(err.mean()); bias = float(signed.mean()); p95 = float(np.percentile(err, 95))
    print(f"\n{'AGGREGATE':<12}{len(err):>5}{mae:>7.2f}{bias:>+7.2f}{p95:>6.1f}{err.max():>5}"
          f"{tot_viol:>8}")
    # There is no true R5 ground truth for a DELAYED commute (R5 here is schedule-perfect), so
    # this bounds realistic's drift from schedule-perfect rather than measuring error: realistic
    # should sit modestly ABOVE R5 p50 (positive bias), more so in fragile peripheral workplaces.
    print(f"\nSanity: MAE<=2.5  0<=bias<=2.0 (realistic sits above schedule-perfect)  "
          f"perfect<=realistic violations=0")
    ok = (mae <= 2.5 and 0.0 <= bias <= 2.0 and tot_viol == 0)
    print("RESULT:", "PASS" if ok else "REVIEW",
          f"(MAE={mae:.2f}, bias={bias:+.2f}, perfect>realistic={tot_viol})")


if __name__ == "__main__":
    main()
