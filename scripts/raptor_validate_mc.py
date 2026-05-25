"""Validate the service-noise Monte-Carlo "realistic" layer against the R5 oracle (JVM-free).

Reports BOTH realistic modes per workplace at the same seed:
  * COMMITTED (the default ``realistic``): fix departure + first leg from the published plan, then
    per draw re-optimize the TAIL from the actual late arrival. The honest "I missed my transfer
    and ate a headway" cost.
  * CLAIRVOYANT (``committed=False``): re-route the WHOLE journey with foreknowledge of the draw =
    an optimistic LOWER bound (route resilience: "is there a good alternative under bad conditions").

R5's depart-after window p50 here is schedule-PERFECT (no delays), so it is NOT ground truth for a
delayed commute — both modes should sit ABOVE it (positive bias), and committed above clairvoyant
(committing the first leg can only add missed-transfer pain). We assert:

  * committed_mean >= clairvoyant_mean   (committed is the more honest, higher number)
  * bias vs R5 p50 > 0                    (realistic sits above schedule-perfect)
  * per-cell invariant: perfect (arrive-by best-case) <= committed   (MC never beats perfect)

See RAPTOR.md for the committed-plan upgrade + its remaining caveat (the tail stays clairvoyant).

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
    all_signed = []
    tot_viol = 0
    comm_sum = clair_sum = ncell = 0
    # Per workplace, BOTH realistic modes at the same seed: COMMITTED (default — fix the first leg,
    # re-optimize the late tail) and CLAIRVOYANT (re-route the whole journey with foreknowledge =
    # the optimistic lower bound). Committed should sit ABOVE clairvoyant, which sits above R5 p50.
    print(f"\n{'workplace':<12}{'n':>5}{'clairv':>8}{'commit':>8}{'biasR5':>8}{'p>real':>8}"
          f"{'frag50':>8}{'frag90':>8}{'stuck%':>8}  time")
    for op in oracles:
        z = np.load(op, allow_pickle=True)
        name = str(z["name"])
        pw = _purewalk(eng, z)
        # perfect (arrive-by best-case) per cell, the map's headline value
        perfect, _dom = eng.journey_tree(z["egress_g"], z["egress_w"], pw).commute_and_dominant()
        perfect = np.asarray(perfect, np.int32)
        clair = eng.montecarlo(z["egress_g"], z["egress_w"], pw, perfect=perfect,
                               seed=12345, alt_draws=0, committed=False)
        t0 = time.time()
        mc = eng.montecarlo(z["egress_g"], z["egress_w"], pw, perfect=perfect,
                            seed=12345, alt_draws=0, committed=True)
        dt = time.time() - t0
        realistic = mc["realistic"]; reach = perfect >= 0
        viol = int(np.sum(realistic[reach] < perfect[reach]))
        tot_viol += viol
        signed = []
        for k, cid in enumerate(z["cell_ids"].astype(str)):
            i = pos.get(cid)
            if i is None or perfect[i] < 0:
                continue
            r5 = int(z["p50"][k])
            if r5 < 0:
                continue
            signed.append(int(realistic[i]) - r5)
        signed = np.array(signed); all_signed.append(signed)
        cm_mean = float(realistic[reach].mean()); cl_mean = float(clair["realistic"][reach].mean())
        comm_sum += int(realistic[reach].sum()); clair_sum += int(clair["realistic"][reach].sum())
        ncell += int(reach.sum())
        frag = mc["frag"][reach]; stuck = mc["stuck"][reach]
        print(f"{name:<12}{reach.sum():>5}{cl_mean:>8.1f}{cm_mean:>8.1f}{signed.mean():>+8.2f}"
              f"{viol:>8}{np.median(frag):>8.0f}{np.percentile(frag,90):>8.0f}"
              f"{np.mean(stuck>=0.15)*100:>7.0f}%  {dt:.2f}s")
    signed = np.concatenate(all_signed)
    bias = float(signed.mean())
    comm_mean, clair_mean = comm_sum / ncell, clair_sum / ncell
    print(f"\n{'AGGREGATE':<12}{ncell:>5}{clair_mean:>8.1f}{comm_mean:>8.1f}{bias:>+8.2f}{tot_viol:>8}")
    # No true R5 ground truth for a DELAYED commute (R5 here is schedule-perfect), so this bounds
    # committed's drift from schedule-perfect, not error. Committed sits ABOVE clairvoyant (it adds
    # missed-transfer pain the foreknowledge version dodges), which sits above R5 p50.
    print(f"\nSanity: committed_mean>=clairvoyant_mean  bias>0 (above schedule-perfect)  "
          f"perfect<=committed violations=0")
    ok = (comm_mean >= clair_mean and bias > 0.0 and tot_viol == 0)
    print("RESULT:", "PASS" if ok else "REVIEW",
          f"(committed={comm_mean:.2f} vs clairvoyant={clair_mean:.2f}, biasR5={bias:+.2f}, "
          f"perfect>committed={tot_viol})")


if __name__ == "__main__":
    main()
