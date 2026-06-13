"""Validate the committed-plan service-noise Monte-Carlo "realistic" layer vs the R5 oracle (JVM-free).

The committed-plan MC fixes departure + first leg from the published plan, then per draw re-optimizes
the TAIL from the actual late arrival — the honest "I missed my transfer and ate a headway" cost.

R5's depart-after window p50 here is schedule-PERFECT (no delays), so it is NOT ground truth for a
delayed commute — committed should sit ABOVE it (positive bias). We assert:

  * bias vs R5 p50 small                                (realistic tracks the schedule median)
  * PRE-floor committed p50 ("realistic_raw") sits ABOVE perfect on average — the non-vacuous
    direction. The SERVED realistic >= perfect by construction: montecarlo() floors the draws at
    perfect, so checking the served value would be a tautology (we still report it as a floor
    sanity column, not as an invariant of the kernel).

See RAPTOR.md for the committed-plan model + its remaining caveat (the tail re-optimizes clairvoyantly).

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
    all_signed, all_rawd = [], []
    tot_viol = 0
    comm_sum = ncell = 0
    # Committed-plan realistic per workplace (fix the first leg, re-optimize the late tail). It
    # should sit ABOVE R5 p50 (which is schedule-perfect here, NOT ground truth for a delayed
    # commute); rawbias is the PRE-floor committed-perfect mean (the real kernel signal).
    print(f"\n{'workplace':<12}{'n':>5}{'commit':>8}{'biasR5':>8}{'p>real':>8}{'rawbias':>9}"
          f"{'frag50':>8}{'frag90':>8}{'stuck%':>8}  time")
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
        realistic = mc["realistic"]; reach = perfect >= 0
        viol = int(np.sum(realistic[reach] < perfect[reach]))   # served floor (0 by construction)
        tot_viol += viol
        rawd = mc["realistic_raw"][reach].astype(np.int64) - perfect[reach]
        all_rawd.append(rawd)
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
        cm_mean = float(realistic[reach].mean())
        comm_sum += int(realistic[reach].sum())
        ncell += int(reach.sum())
        frag = mc["frag"][reach]; stuck = mc["stuck"][reach]
        print(f"{name:<12}{reach.sum():>5}{cm_mean:>8.1f}{signed.mean():>+8.2f}"
              f"{viol:>8}{rawd.mean():>+9.2f}{np.median(frag):>8.0f}{np.percentile(frag,90):>8.0f}"
              f"{np.mean(stuck>=0.15)*100:>7.0f}%  {dt:.2f}s")
    signed = np.concatenate(all_signed)
    rawd = np.concatenate(all_rawd)
    bias = float(signed.mean())
    rawbias = float(rawd.mean())
    comm_mean = comm_sum / ncell
    print(f"\n{'AGGREGATE':<12}{ncell:>5}{comm_mean:>8.1f}{bias:>+8.2f}{tot_viol:>8}{rawbias:>+9.2f}")
    # Committed = best (arrive-by) departure + small service delay, so it tracks R5's depart-window
    # p50 closely (|bias| small, either sign — R5 p50 already includes typical wait). The served
    # realistic >= perfect by construction (montecarlo() floors the draws), so the kernel-level
    # check is on the PRE-floor p50: delays must push committed ABOVE perfect on average (the
    # per-cell negative tail is the legit traced-tree-vs-sweep relaxation, see the zero-pert test).
    print(f"\nSanity: |bias vs R5 p50| <= 3; pre-floor committed-perfect mean > 0; "
          f"served floor holds (p>real = 0 by construction)")
    ok = (abs(bias) <= 3.0 and tot_viol == 0 and rawbias > 0.0)
    print("RESULT:", "PASS" if ok else "REVIEW",
          f"(committed={comm_mean:.2f}, biasR5={bias:+.2f}, rawbias={rawbias:+.2f}, "
          f"perfect>served={tot_viol})")


if __name__ == "__main__":
    main()
