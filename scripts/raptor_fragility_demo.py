"""Prototype: a per-cell COMMUTE FRAGILITY score, JVM-free, from the RAPTOR back-pointers.

Models the user's distinction: the perfect-timing commute is the BASE (your departure timing is
in your control); the risk worth surfacing is SERVICE noise you CAN'T control — chiefly, "if a
delay makes you miss a connection, how long until the next one?" For each transit leg we read the
recovery headway (gap to the next trip on that pattern at the board stop) straight from the loaded
GTFS. The fragility of a journey ~ the worst recovery headway among its TRANSFERS (0-transfer trips
are robust: only small ride-time noise). This is ~free (no Monte-Carlo, no forward search).
"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from core import raptor_engine, raptor_golden

GOLDEN = ROOT / "tests" / "raptor_golden"
NAME = os.environ.get("WP", "downtown")
eng = raptor_engine.RaptorEngine(verbose=False)
z = np.load(GOLDEN / f"oracle_{NAME}.npz", allow_pickle=True)
pw = raptor_golden.purewalk_aligned(eng, z)
tree = eng.journey_tree(z["egress_g"], z["egress_w"], pw)
d = eng.data
pat_dep, pat_nstops, pat_mat_off = d["pat_dep"], d["pat_nstops"], d["pat_mat_off"]
CAP = eng.max_min


def recovery_headway(pi, bpos, dep_sec):
    """Minutes until the NEXT trip on pattern pi departs the board stop (>= CAP if none = last).
    The ride tuple carries no trip index, so recover the next trip by binary search over the
    board-stop departure column (trips within a pattern are departure-sorted)."""
    ns = int(pat_nstops[pi]); mb = int(pat_mat_off[pi])
    nt = (int(pat_mat_off[pi + 1]) - mb) // ns
    deps = pat_dep[mb + bpos: mb + nt * ns: ns]       # departures at bpos, trips 0..nt-1
    t = int(np.searchsorted(deps, dep_sec, side="right"))   # first trip departing AFTER ours
    if t >= nt:
        return CAP                                    # last trip of the morning — miss it, stuck
    return (int(deps[t]) - dep_sec) // 60


def fragility(ci):
    tr = tree._trace(ci)
    if tr is None:
        return None
    legs_raw, lh = tr
    rides = [l for l in legs_raw if l[0] == "ride"]
    # ride tuple = ("ride", pi, dep_sec, arr_sec, bpos, apos, alight_stop): pattern, board
    # position, and OUR departure (the trip index is recovered inside recovery_headway)
    recov = [recovery_headway(l[1], l[4], l[2]) for l in rides]         # per-leg recovery headway
    names = [tree._name(l[1]) for l in rides]
    n_xfer = max(0, len(rides) - 1)
    # the uncontrollable risk = worst recovery among the TRANSFER connections (legs after the 1st)
    frag = max(recov[1:]) if len(recov) > 1 else 0
    return dict(legs=list(zip(names, recov)), n_xfer=n_xfer, frag=frag)


commute, _ = tree.commute_and_dominant()
rows = []
for ci in range(len(eng.cell_ids)):
    if commute[ci] < 0:
        continue
    f = fragility(ci)
    if f:
        rows.append((ci, int(commute[ci]), f))

frags = np.array([r[2]["frag"] for r in rows])
print(f"workplace={NAME}  cells={len(rows)}")
print(f"  0-transfer (robust) cells: {sum(1 for r in rows if r[2]['n_xfer']==0)} "
      f"({sum(1 for r in rows if r[2]['n_xfer']==0)/len(rows)*100:.0f}%)")
print(f"  fragility (worst miss-a-transfer recovery, min): "
      f"median {int(np.median(frags))}  p90 {int(np.percentile(frags,90))}  "
      f"robust(<=8m) {np.mean(frags<=8)*100:.0f}%  fragile(>=20m) {np.mean(frags>=20)*100:.0f}%")

print("\n  ROBUST examples (miss a connection -> short wait):")
for ci, c, f in sorted([r for r in rows if r[2]['n_xfer'] >= 1], key=lambda r: r[2]['frag'])[:4]:
    legs = "  ".join(f"{n}(miss=+{h}m)" for n, h in f["legs"])
    print(f"   cell {eng.cell_ids[ci]}: {c}m, {f['n_xfer']} xfer | {legs}")
print("\n  FRAGILE examples (miss a connection -> long wait / stuck):")
for ci, c, f in sorted([r for r in rows if r[2]['n_xfer'] >= 1], key=lambda r: -r[2]['frag'])[:6]:
    legs = "  ".join(f"{n}(miss=+{h}m)" for n, h in f["legs"])
    print(f"   cell {eng.cell_ids[ci]}: {c}m, {f['n_xfer']} xfer, fragility {f['frag']}m | {legs}")
