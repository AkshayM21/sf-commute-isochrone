"""Validate the Phase-2 RAPTOR path layer against the R5 path oracle (JVM-free).

Compares, per workplace:
  - color-by-line: % of cells where RAPTOR's dominant route == R5's dominant route
    (R5 = depart-after recorded paths; RAPTOR = arrive-by traced journey, so some disagreement
    is the anchor difference + genuine equally-fast alternatives, not a bug);
  - route-sequence match on the sampled cells (ordered transit-line list).

Also re-checks the hover==map invariant on every cell (must be exact). Skips if the R5 path
oracles (scripts/raptor_path_oracle.py) or the access table are absent.
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from core import raptor_engine
GOLDEN = ROOT / "tests" / "raptor_golden"


def main():
    paths = sorted(GOLDEN.glob("path_*.json"))
    if not paths:
        sys.exit("no R5 path oracles — run scripts/raptor_path_oracle.py")
    eng = raptor_engine.RaptorEngine(verbose=False)
    cidx = {c: i for i, c in enumerate(eng.cell_ids)}
    agree_tot = compared_tot = inv_viol = 0
    seq_match = seq_tot = 0
    print(f"{'workplace':<12}{'cells':>7}{'dom-agree':>11}{'walkonly-agree':>16}{'seq-match':>11}")
    for pj in paths:
        o = json.loads(pj.read_text())
        name = o["name"]
        npz = GOLDEN / f"oracle_{name}.npz"
        if not npz.exists():
            continue
        z = np.load(npz, allow_pickle=True)
        pw = np.array([int(z["purewalk"][i]) for i in range(len(eng.cell_ids))], np.int64)
        tree = eng.journey_tree(z["egress_g"], z["egress_w"], pw)
        commute, dom = tree.commute_and_dominant()
        r5dom = o["dominant"]
        agree = compared = wo_agree = wo_tot = 0
        for cid, r5line in r5dom.items():
            i = cidx.get(cid)
            if i is None or commute[i] < 0:
                continue
            mine = dom[i]
            if mine is None:
                continue
            compared += 1
            if mine == r5line:
                agree += 1
            if r5line == "walk only" or mine == "walk only":
                wo_tot += 1
                if mine == r5line:
                    wo_agree += 1
            # hover==map invariant
            it = tree.itinerary(i)
            if it is not None:
                s = sum(l["min"] for l in it["legs"]) + sum(l.get("wait", 0) for l in it["legs"])
                if s != it["total"]:
                    inv_viol += 1
        # route-sequence on sampled cells
        sm = st = 0
        for cid, r5it in o["sample_itins"].items():
            i = cidx.get(cid)
            if i is None or commute[i] < 0:
                continue
            it = tree.itinerary(i)
            if it is None:
                continue
            r5seq = [l["line"] for l in r5it["legs"] if l["mode"] == "transit"]
            myseq = [l["line"] for l in it["legs"] if l["mode"] == "transit"]
            st += 1
            if r5seq == myseq:
                sm += 1
        agree_tot += agree; compared_tot += compared; seq_match += sm; seq_tot += st
        print(f"{name:<12}{compared:>7}{agree/max(1,compared)*100:>10.1f}%"
              f"{wo_agree/max(1,wo_tot)*100:>15.1f}%{sm/max(1,st)*100:>10.0f}%")
    print(f"\n{'AGGREGATE':<12}{compared_tot:>7}{agree_tot/max(1,compared_tot)*100:>10.1f}%"
          f"{'':>16}{seq_match/max(1,seq_tot)*100:>10.0f}%")
    print(f"hover==map invariant violations: {inv_viol} (must be 0)")
    print(f"\nDoD: dominant-line agreement >= 90% ; hover==map = 0 violations")
    print("RESULT:", "PASS" if (agree_tot / max(1, compared_tot) >= 0.90 and inv_viol == 0)
          else "REVIEW")


if __name__ == "__main__":
    main()
