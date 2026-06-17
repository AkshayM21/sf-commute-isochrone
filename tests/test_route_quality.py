"""Permanent route-QUALITY regression suite for the JVM-free reverse-RAPTOR arrive-by engine.

These guard the whole CLASS of "dominated / illogical routing" bugs fixed in 2026-06-16 (commit
b260240: min-journey access-stop selection + no-overshoot final-ride alight, plus the 2026-06-16
``_build_stop_dominant`` no-overshoot/route-name tie-break fix that closed the alt-label gap) so
they cannot silently return. Each test SCANS reachable cells across the 5 golden workplaces
(``tests/raptor_golden/oracle_*.npz``) using the same harness ``scripts/core/raptor_golden.py``
and ``tests/test_raptor.py`` use.

The four bug classes (each gets a scanning test):
  1. OVERSHOOT       — a traced final ride rides PAST the stop closest to W and walks back
                       (longer ride AND longer egress walk; strictly dominated). STRICT == 0.
  2. PRIMARY < ALT   — the shown (primary) route is slower than a line in its own "also serves"
                       window. STRICT: primary <= every alt + 1 (the walk-prior eps allows 1 min).
  3. HOVER == MAP    — the displayed route's total minutes must equal the map cell value. STRICT == 0.
  4. WALK MONOTONE   — walking FASTER must not lengthen the commute. Partially inherent (the
                       single-departure "latest run before 09:00" jiggle); RATCHETED to a recorded
                       per-workplace baseline so it can never get WORSE (driven to 0 by the future
                       arrival-window phase).

Trees are precomputed ONCE per workplace (and per walk speed for the monotonicity test) in a
module-scoped fixture; the user explicitly accepts slow tests. The heaviest part (the full
5-workplace x 3-speed monotonicity scan) is gated behind ``@pytest.mark.slow`` but a 2-workplace
default scan keeps CI catching regressions. JVM-free; skips cleanly if the bakes/oracles are absent.
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
GOLDEN = os.path.join(_HERE, "raptor_golden")

# walk-speed scalars = 4.8 / pace (slow 4.0, med 4.8, fast 5.6 km/h); see core/walk.py + RAPTOR.md
SPEED_SCALAR = {"slow": 1.20, "med": 1.00, "fast": 0.857}
ALT_WINDOW_MIN = 5            # the alt dominance window (RAPTOR_ALT_WINDOW_MIN default)
_JBIG = np.int64(1) << 39     # the per-stop jtime "unreachable" sentinel


def _oracles():
    if not os.path.isdir(GOLDEN):
        return []
    return sorted(f for f in os.listdir(GOLDEN) if f.startswith("oracle_") and f.endswith(".npz"))


# A 2-workplace subset for the DEFAULT (non-slow) run, so plain `pytest` still catches a regression
# without the full 5x3 walk-speed sweep. downtown+bayview cover the densest transfer corridor and
# the transit-sparse SE periphery (where the residual jiggle lives).
_DEFAULT_SUBSET = ("oracle_downtown.npz", "oracle_bayview.npz")


@pytest.fixture(scope="module")
def engine():
    from core import raptor_engine
    try:
        return raptor_engine.RaptorEngine(verbose=False)
    except FileNotFoundError as e:                 # access table not baked in this checkout
        pytest.skip(f"RAPTOR access table not baked ({e}); run scripts/raptor_oracle.py")


@pytest.fixture(scope="module")
def trees(engine):
    """Per-workplace traced arrive-by tree at MED walk speed (the served default), built ONCE.
    Returns {workplace_name: (tree, commute_array, dominant_list, alt_window_list)}."""
    from core import raptor_golden
    out = {}
    for f in _oracles():
        z = np.load(os.path.join(GOLDEN, f), allow_pickle=True)
        pw = raptor_golden.purewalk_aligned(engine, z)
        tree = engine.journey_tree(z["egress_g"], z["egress_w"], pw)
        commute, dom = tree.commute_and_dominant()
        win = tree.alt_lines_window(commute, ALT_WINDOW_MIN)
        out[f.replace("oracle_", "").replace(".npz", "")] = (tree, commute, dom, win)
    return out


@pytest.fixture(scope="module")
def da_trees(engine):
    """Per-workplace DEPART-AFTER traced tree (Stage 1 of the depart-after map migration) at MED
    walk speed, built ONCE. Returns {workplace_name: (datree, painted_p50_dict, cell_ids)} where
    painted_p50_dict is ``engine.departafter``'s p50 ({cell_id: [p5, p50]}) — the independent map
    value the traced tree must match — and cell_ids is the engine cell order (== tree row order).
    hover==map is the gate (test_departafter_hover_equals_map)."""
    from core import raptor_golden
    out = {}
    for f in _oracles():
        z = np.load(os.path.join(GOLDEN, f), allow_pickle=True)
        pw = raptor_golden.purewalk_aligned(engine, z)
        datree = engine.journey_tree_departafter(z["egress_g"], z["egress_w"], pw, percentile=50)
        painted = engine.departafter(z["egress_g"], z["egress_w"], pw, percentiles=(5, 50))
        out[f.replace("oracle_", "").replace(".npz", "")] = (datree, painted, list(engine.cell_ids))
    return out


@pytest.fixture(scope="module")
def egress(engine):
    """Per-workplace egress arrays (egress_g, egress_w) from the oracles, by workplace name —
    needed by route_typicals (the per-route committed MC scores the tail from the same egress)."""
    out = {}
    for f in _oracles():
        z = np.load(os.path.join(GOLDEN, f), allow_pickle=True)
        out[f.replace("oracle_", "").replace(".npz", "")] = (z["egress_g"], z["egress_w"])
    return out


# ---------------------------------------------------------------------------------------------
# 1. hover == map (the foundational invariant)
# ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(not _oracles(), reason="no R5 oracles in tests/raptor_golden/")
def test_hover_equals_map(trees):
    """For every reachable cell across ALL 5 workplaces, the hover itinerary total must equal the
    map cell value EXACTLY. STRICT: 0 violations. (The breakdown and the map come from the same
    traced tree, so any drift is a reconstruction bug.)"""
    total_checked = total_viol = 0
    for name, (tree, commute, _dom, _win) in trees.items():
        checked = viol = 0
        for ci in range(tree.n_cells):
            if commute[ci] < 0:
                continue
            it = tree.itinerary(ci)
            checked += 1
            if it is None or it["total"] != int(commute[ci]):
                viol += 1
        assert checked > 500, f"{name}: only {checked} reachable cells — fixture too thin"
        assert viol == 0, f"{name}: hover!=map on {viol}/{checked} cells"
        total_checked += checked; total_viol += viol
    assert total_checked > 5000
    assert total_viol == 0


# ---------------------------------------------------------------------------------------------
# 2. no overshoot on the final ride
# ---------------------------------------------------------------------------------------------
def _egress_walk(tree, gid):
    """Per-stop egress WALK seconds (the table the engine builds), or None if not egress-reachable
    from W (the EGRESS_INF sentinel)."""
    from core.raptor_journey import EGRESS_INF
    w = int(tree.egress_sec[gid])
    return None if w >= EGRESS_INF else w


@pytest.mark.skipif(not _oracles(), reason="no R5 oracles in tests/raptor_golden/")
def test_no_overshoot(trees):
    """For every traced journey ending in a transit ride + egress, the final ride's alight must
    already MINIMIZE ``arr[p] + egress_walk(stop@p)`` over forward egress-reachable positions
    ``p >= board+1`` on that trip — i.e. an INDEPENDENT re-run of the minimization finds nothing
    strictly better than the emitted alight. STRICT: 0 overshoot cells.

    This re-derives the minimum from scratch (reading ``tree.egress_sec`` directly, not calling the
    production ``_min_overshoot_alight``) so the test is a genuine cross-check, not a tautology. The
    emitted alight is egress-reachable by construction, so its ``arr + egress_walk`` is well-defined;
    we assert no forward position beats it."""
    total_checked = total_overshoot = 0
    for name, (tree, commute, _dom, _win) in trees.items():
        d = tree.d
        pat_arr = d["pat_arr"]; pat_dep = d["pat_dep"]; pat_stops = d["pat_stops"]
        pat_nstops = d["pat_nstops"]; pat_mat_off = d["pat_mat_off"]
        pat_stop_off = d["pat_stop_off"]; pat_ntrips = d["pat_ntrips"]
        checked = overshoot = 0
        for ci in range(tree.n_cells):
            if commute[ci] < 0:
                continue
            tr = tree._trace(ci)
            if tr is None:
                continue
            legs_raw, _lh = tr
            # final ride = the last ride leg, iff its successor is the egress
            fin = None
            for idx in range(len(legs_raw) - 1, -1, -1):
                if legs_raw[idx][0] == "ride":
                    if idx + 1 < len(legs_raw) and legs_raw[idx + 1][0] == "egress":
                        fin = legs_raw[idx]
                    break
            if fin is None:
                continue
            checked += 1
            pi = fin[1]; dep_sec = fin[2]; bpos = fin[4]; apos = fin[5]; alight = fin[6]
            ns = int(pat_nstops[pi]); mb = int(pat_mat_off[pi]); sb = int(pat_stop_off[pi])
            # recover the trip from the board departure (unique by board position + dep time)
            trip = None
            for t in range(int(pat_ntrips[pi])):
                if int(pat_dep[mb + t * ns + bpos]) == dep_sec:
                    trip = t; break
            if trip is None:
                continue
            trow = mb + trip * ns
            # the emitted alight's downstream W-arrival (its egress is reachable by construction)
            cur_w = _egress_walk(tree, alight)
            assert cur_w is not None, f"{name} cell {ci}: emitted final alight not egress-reachable"
            cur_arrW = int(pat_arr[trow + apos]) + cur_w
            # independent minimization over forward egress-reachable positions
            best_arrW = cur_arrW
            for p in range(bpos + 1, ns):
                g = int(pat_stops[sb + p])
                w = _egress_walk(tree, g)
                if w is None:
                    continue
                cand = int(pat_arr[trow + p]) + w
                if cand < best_arrW:
                    best_arrW = cand
            if best_arrW < cur_arrW:
                overshoot += 1
        assert checked > 200, f"{name}: only {checked} transit+egress journeys — fixture too thin"
        total_checked += checked; total_overshoot += overshoot
        assert overshoot == 0, (
            f"{name}: {overshoot}/{checked} final rides overshoot the egress-optimal alight "
            "(no-overshoot regression)")
    assert total_checked > 2000


# ---------------------------------------------------------------------------------------------
# 3. primary is the fastest in its own compare window
# ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(not _oracles(), reason="no R5 oracles in tests/raptor_golden/")
def test_primary_is_fastest(trees):
    """For every reachable cell, the primary route's door-to-door minutes must be <= every line in
    that cell's alt window + 1 (the walk-prior eps allows a 1-min near-tie). STRICT: 0 violations.

    This was the documented KNOWN GAP (downtown cell ~1601): ``_build_stop_dominant``'s per-stop
    dominant LABEL disagreed with ``_dominant``'s traced-journey label because the per-stop pass
    measured the final ride at its OVERSHOOT length (the raw node-chain alight) instead of the
    no-overshoot alight ``_trace_from`` emits — so a faster line surfaced as an alt while a slower
    line was shown as primary. Fixed 2026-06-16 by applying ``_min_overshoot_alight`` to the final
    ride inside ``_build_stop_dominant`` AND matching ``_dominant``'s (route name, feed, route_id)
    tie-break. With the fix the primary is always the fastest in its window, so this is STRICT.

    Determinism note: the primary's displayed time is read from its entry in the window when present
    (the same time the breakdown shows); when absent it's the cell's map total. With the fix the
    primary is never absent-while-a-faster-alt-is-present (separately asserted by the audit)."""
    total_viol = 0; worst_gap = 0; worst = None
    for name, (tree, commute, dom, win) in trees.items():
        for ci in range(tree.n_cells):
            if commute[ci] < 0:
                continue
            cw = win[ci] or {}
            if not cw:
                continue
            prim_line = dom[ci]
            prim_t = cw[prim_line][0] if prim_line in cw else int(commute[ci])
            best_alt = min(m for m, _ in cw.values())
            gap = prim_t - best_alt
            if gap > 1:
                total_viol += 1
                if gap > worst_gap:
                    worst_gap = gap; worst = (name, ci, prim_line, prim_t, best_alt)
    assert total_viol == 0, (
        f"{total_viol} cells show a primary route slower than an alt by > 1 min "
        f"(worst: workplace={worst[0]} cell={worst[1]} primary={worst[2]!r} "
        f"{worst[3]}min vs best-alt {worst[4]}min — primary-slower-than-alt regression)")


# ---------------------------------------------------------------------------------------------
# 4. walk-speed monotonicity (RATCHET, not strict — the residual is inherent)
# ---------------------------------------------------------------------------------------------
# Per-workplace baselines for the faster-walk -> longer-commute residual, measured 2026-06-16 on
# the shipped engine (min-journey selection + no-overshoot alight + the _stop_dominant fix). This
# is the INHERENT single-departure "latest run before 09:00" jiggle: at a faster walk the arrive-by
# latest-departure can genuinely re-pick a different corridor whose single feasible journey is a
# little longer (a real routing effect of one-journey-per-access-stop, not a dominance bug). It will
# be driven to 0 by the future arrival-window phase. Each entry is (count_ceiling, max_increase_min)
# per adjacent faster step. The test asserts the engine never REGRESSES past these — a ratchet.
#
# inherent single-departure jiggle; tighten to 0 after the arrival-window phase.
_MONO_BASELINE = {
    "bayview":    {"slow->med": (39, 4),  "med->fast": (204, 7)},
    "caltrain":   {"slow->med": (90, 5),  "med->fast": (124, 8)},
    "downtown":   {"slow->med": (53, 7),  "med->fast": (64, 4)},
    "sunset":     {"slow->med": (405, 3), "med->fast": (238, 7)},
    "westportal": {"slow->med": (392, 9), "med->fast": (30, 7)},
}


def _commute_at(engine, z, scalar):
    from core import raptor_golden
    pw = raptor_golden.purewalk_aligned(engine, z)
    tree = engine.journey_tree(z["egress_g"], z["egress_w"], pw, walk_scalar=scalar)
    commute, _ = tree.commute_and_dominant()
    return commute


def _mono_one(engine, fname):
    """Return {step: (count, max_increase)} for one workplace's slow->med and med->fast steps."""
    z = np.load(os.path.join(GOLDEN, fname), allow_pickle=True)
    cm = {sp: _commute_at(engine, z, sc) for sp, sc in SPEED_SCALAR.items()}
    res = {}
    for a, b in (("slow", "med"), ("med", "fast")):
        ca, cb = cm[a], cm[b]
        both = (ca >= 0) & (cb >= 0)
        inc = both & (cb > ca)                 # faster walk (b) -> LONGER commute = the residual
        cnt = int(np.sum(inc))
        mx = int((cb - ca)[inc].max()) if cnt else 0
        res[f"{a}->{b}"] = (cnt, mx)
    return res


def _assert_mono_ratchet(name, res):
    base = _MONO_BASELINE[name]
    for step, (cnt, mx) in res.items():
        bc, bm = base[step]
        assert cnt <= bc, (
            f"{name} {step}: faster-walk-longer-commute cells {cnt} > baseline {bc} "
            "(walk-speed monotonicity REGRESSED — inherent jiggle, but it must not get worse)")
        assert mx <= bm, (
            f"{name} {step}: max commute increase on speed-up {mx}min > baseline {bm}min "
            "(walk-speed monotonicity REGRESSED)")


@pytest.mark.skipif(not _oracles(), reason="no R5 oracles in tests/raptor_golden/")
def test_walkspeed_monotonicity_default(engine):
    """Default (non-slow) walk-speed monotonicity ratchet on a 2-workplace subset, so plain pytest
    catches a regression without the full 5x3 sweep. See _MONO_BASELINE for the recorded ceilings;
    the residual is the inherent single-departure jiggle (tighten to 0 after the arrival-window
    phase)."""
    for fname in _DEFAULT_SUBSET:
        if fname not in _oracles():
            continue
        name = fname.replace("oracle_", "").replace(".npz", "")
        _assert_mono_ratchet(name, _mono_one(engine, fname))


@pytest.mark.slow
@pytest.mark.skipif(not _oracles(), reason="no R5 oracles in tests/raptor_golden/")
def test_walkspeed_monotonicity_all(engine):
    """Full 5-workplace x 3-speed walk-speed monotonicity ratchet (the heavy scan). Each adjacent
    faster step must not increase the faster-walk-longer-commute cell COUNT or its MAX magnitude
    past the recorded per-workplace baseline (_MONO_BASELINE). This is the inherent single-departure
    "latest run before 09:00" jiggle — it will be driven to 0 by the future arrival-window phase;
    until then it is RATCHETED so no change can make it worse."""
    for fname in _oracles():
        name = fname.replace("oracle_", "").replace(".npz", "")
        if name not in _MONO_BASELINE:
            pytest.skip(f"no recorded monotonicity baseline for workplace {name!r}")
        _assert_mono_ratchet(name, _mono_one(engine, fname))


# ---------------------------------------------------------------------------------------------
# 5. per-route TYPICAL (committed-plan p50) — the compare-list consistency fix
# ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(not _oracles(), reason="no R5 oracles in tests/raptor_golden/")
def test_per_route_typical_honors_perfect_le_committed(engine, trees, egress):
    """``RaptorEngine.route_typicals`` scores the PRIMARY + each alt of a cell with the SAME
    committed-plan MC as the served realistic map. Two invariants, scanned over cells with alts
    across the default workplace subset:

      1. perfect <= committed PER ROUTE: every route's typical (real) >= its OWN best-case minutes
         (the primary and each alt), so the compare card never shows a typical FASTER than the
         route's perfect-timing number. STRICT == 0 violations.
      2. internal consistency in BEST-CASE terms: the primary's best-case <= every alt's best-case
         + the walk-prior eps (1 min), i.e. the shown route is the fastest in its own window (the
         engine fix that motivated this display change). STRICT (allow the 1-min eps).

    Each cell's routes are ONE shared kernel call; the seed mirrors the server's per-workplace seed
    shape so this exercises the exact code path /itinerary?pin=1 hits."""
    from core import raptor_engine
    checked_routes = viol_floor = viol_primary = cells = 0
    for name in _DEFAULT_SUBSET:
        key = name.replace("oracle_", "").replace(".npz", "")
        if key not in trees:
            continue
        tree, commute, dom, win = trees[key]
        eg_g, eg_w = egress[key]
        # cells with >=2 windowed transit lines (a primary + >=1 alt), capped for runtime
        cand = [ci for ci, w in enumerate(win) if w and len(w) >= 2 and commute[ci] >= 0]
        for ci in cand[:60]:
            s_star, _aw, _lh, is_walk = tree._select(ci)
            if is_walk or s_star < 0:
                continue                                 # walk-only primary: no transit typical
            prim_line = dom[ci]
            alt_items = [(ln, mk) for ln, mk in win[ci].items() if ln != prim_line][:4]
            if not alt_items:
                continue
            stops = [int(s_star)] + [int(mk[1]) for _ln, mk in alt_items]
            floors = [int(commute[ci])] + [int(mk[0]) for _ln, mk in alt_items]
            pairs = engine.route_typicals(tree, ci, stops, eg_g, eg_w,
                                          perfect_route_mins=floors, seed=12345)
            cells += 1
            # 1. perfect <= committed per route
            for p, fl in zip(pairs, floors):
                if p is None:
                    continue
                checked_routes += 1
                if p[0] < fl:
                    viol_floor += 1
                assert p[1] >= 0, f"{key}/{ci}: negative fragility {p[1]}"
            # 2. primary best-case <= every alt best-case + eps (the engine consistency fix)
            for _ln, mk in alt_items:
                if int(commute[ci]) > int(mk[0]) + 1:
                    viol_primary += 1
    assert cells > 0, "no multi-line cells scanned — fixture/oracle problem"
    assert viol_floor == 0, f"{viol_floor}/{checked_routes} routes violate perfect<=committed"
    assert viol_primary == 0, f"{viol_primary} cells: primary best-case slower than an alt (> eps)"


# ---------------------------------------------------------------------------------------------
# 6. DEPART-AFTER hover == map (Stage 1 of the depart-after map migration)
# ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(not _oracles(), reason="no R5 oracles in tests/raptor_golden/")
def test_departafter_hover_equals_map(da_trees):
    """The depart-after tracer's foundational invariant: for EVERY reachable cell across ALL 5
    workplaces, the forward-traced journey total (anchored on the painted arrivalW value, traced
    from the painted access stop s* at the representative deadline T*) must equal the INDEPENDENT
    depart-after p50 the map paints (``engine.departafter``) EXACTLY, AND the traced legs must
    reconcile to that total. STRICT: 0 violations (proto6/verify_prior: 14626/14626).

    Two cross-checks per cell, so this is not a tautology:
      (a) hover==map: ``DepartAfterJourneyTree.commute_and_dominant``/``itinerary`` total ==
          ``engine.departafter(...)[cell][p50]`` (two independently-computed code paths — the
          assemble kernel vs. the tracer's penalized per-departure selection).
      (b) feasible + reconciling: the traced legs (access -> rides -> egress) sum (incl. waits) to
          the total, with >= 1 walk leg (an access leg always exists)."""
    total_checked = total_viol = total_legbad = 0
    for name, (datree, painted, cell_ids) in da_trees.items():
        commute, dom = datree.commute_and_dominant()
        checked = viol = legbad = 0
        for ci in range(datree.n_cells):
            pv = painted[cell_ids[ci]][1]                 # engine.departafter p50 (independent)
            cv = int(commute[ci]) if commute[ci] >= 0 else None
            if pv is None and cv is None:
                continue
            checked += 1
            if pv != cv:
                viol += 1
                continue
            it = datree.itinerary(ci)
            if it is None or it["total"] != cv:
                viol += 1
                continue
            legsum = sum(l["min"] for l in it["legs"]) + sum(l.get("wait", 0) for l in it["legs"])
            if legsum != it["total"]:
                legbad += 1
        assert checked > 500, f"{name}: only {checked} reachable cells — fixture too thin"
        assert viol == 0, f"{name}: depart-after hover!=map on {viol}/{checked} cells"
        assert legbad == 0, f"{name}: legs don't reconcile on {legbad}/{checked} cells"
        total_checked += checked; total_viol += viol; total_legbad += legbad
    assert total_checked > 5000
    assert total_viol == 0 and total_legbad == 0


def _independent_departafter_selection(datree, percentile=50.0):
    """A DELIBERATELY-SEPARATE numpy reimplementation of the depart-after served-percentile access-
    stop selection — NOT the kernel under test. For each cell it replicates
    ``raptor.assemble_departafter``'s per-departure penalized eps-window pick directly off the tree's
    ``arrivalW`` (time-optimal first, then ``argmin true_time + (beta-1)*aw`` within ``arr<=opt+eps``,
    then pure-walk on ``pw*beta``), takes the same ``method='lower'`` percentile minute, and anchors
    on the LATEST departure achieving it — returning per cell the painted minute and the chosen access
    stop ``s*`` (-1 = walk/unreachable). The hover==map test pins the VALUE; this pins the chosen
    STOP, the thing a tautological legs-sum check can't catch (a wrong s* still reconciles to the
    painted total). Cross-checked against ``datree._select_arrays()`` s* below."""
    aW = datree.arrivalW
    off = np.asarray(datree.access_off, np.int64)
    to = np.asarray(datree.access_to, np.int64)
    aw = np.asarray(datree.access_w, np.int64)
    pwk = np.asarray(datree.purewalk, np.int64)
    dep_grid = np.asarray(datree.dep_grid, np.int64)
    cell_deps = np.asarray(datree.cell_deps, np.int64)
    cap = int(datree.max_min); beta = float(datree.beta); bw = beta - 1.0
    epsi = int(round(float(datree.eps)))
    INF = 1 << 60
    nd = len(cell_deps); ndg = len(dep_grid); n = len(off) - 1
    painted = np.full(n, -1, np.int32)
    s_star = np.full(n, -1, np.int64)
    for ci in range(n):
        a0, a1 = int(off[ci]), int(off[ci + 1])
        gids = to[a0:a1]; awalk = aw[a0:a1]; pw = int(pwk[ci])
        ttm = np.empty(nd, np.float64); d_stop = np.full(nd, -1, np.int64)
        sub = aW[gids] if len(gids) else None
        for di in range(nd):
            D = int(cell_deps[di]); best = INF; bestsc = 1e18; bstop = -1
            if sub is not None:
                kk = np.searchsorted(dep_grid, D + awalk, side="left")
                ok = kk < ndg
                if ok.any():
                    rr = np.where(ok)[0]; vals = sub[rr, kk[rr]]; fin = vals < INF
                    if fin.any():
                        ff = rr[fin]; arrf = vals[fin].astype(np.int64)
                        opt = int(arrf.min())
                        sc = (arrf - D).astype(np.float64) + bw * awalk[ff].astype(np.float64)
                        sc = np.where(arrf <= opt + epsi, sc, np.inf)
                        j = int(np.argmin(sc)); bestsc = float(sc[j]); best = int(arrf[j])
                        bstop = int(gids[ff[j]])
            tt = (best - D) / 60.0 if best < INF else 1e18
            if pw >= 0 and pw * beta < (bestsc if best < INF else 1e18):
                tt = pw / 60.0; bstop = -1
            ttm[di] = np.ceil(min(tt, cap)); d_stop[di] = bstop
        v = int(np.percentile(ttm, percentile, method="lower"))
        if v >= cap:
            continue
        painted[ci] = v
        Di = next((di for di in range(nd - 1, -1, -1) if int(ttm[di]) == v), -1)
        if Di >= 0:
            s_star[ci] = int(d_stop[Di])
    return painted, s_star


@pytest.mark.skipif(not _oracles(), reason="no R5 oracles in tests/raptor_golden/")
def test_departafter_selection_matches_independent(da_trees):
    """C1 (NON-TAUTOLOGICAL): the tracer's chosen access stop ``s*`` AND traced dominant line must
    agree with a SEPARATE recompute of the penalized-eps argmin — pinning the WRONG-STOP class.

    The hover==map test forces legs to sum to the painted total (``reconcile_legs``), so it passes
    for ANY feasible trace even from the wrong stop. Here, for every reachable cell across all 5
    workplaces we (a) recompute the painted minute + representative-departure penalized argmin stop
    INDEPENDENTLY in numpy (``_independent_departafter_selection``) and assert it equals the kernel's
    emitted ``s*`` from ``datree._select_arrays()``; and (b) trace the journey from BOTH the kernel's
    s* and the independent s* and assert the dominant line matches. A re-derivation drift in the
    kernel (or a wrong-stop pick) flips s* (and usually the line) -> caught. STRICT: 0 mismatches.

    Sanity that this is not vacuous: it also asserts a deliberately-corrupted selection (s* shifted
    to a DIFFERENT in-cell access stop) DOES make the dominant line disagree on a real fraction of
    cells — i.e. the check has teeth (see test_departafter_selection_catches_wrong_stop)."""
    tot_checked = tot_smis = tot_lmis = 0
    for name, (datree, _painted_dict, _ids) in da_trees.items():
        s_star, _aw, _D, _T, is_walk, painted = datree._select_arrays()
        ind_painted, ind_s = _independent_departafter_selection(datree, percentile=50)
        checked = smis = lmis = 0
        for ci in range(datree.n_cells):
            if painted[ci] < 0:
                assert ind_painted[ci] < 0, f"{name}/{ci}: kernel unreachable but independent {ind_painted[ci]}"
                continue
            checked += 1
            assert int(ind_painted[ci]) == int(painted[ci]), \
                f"{name}/{ci}: independent painted {ind_painted[ci]} != kernel {painted[ci]}"
            if bool(is_walk[ci]):
                # walk-only: both must agree there's no transit access stop
                if int(ind_s[ci]) >= 0:
                    smis += 1
                continue
            if int(s_star[ci]) != int(ind_s[ci]):
                smis += 1
                continue
            # dominant line traced from the kernel s* must match the line traced from the indep s*
            kern_dom = datree._dominant(datree._trace_raw(ci)[0])
            T = int(_T[ci]); jt = datree._tree_at(T)
            # access-walk sec of the INDEPENDENT stop
            a0, a1 = int(datree.access_off[ci]), int(datree.access_off[ci + 1])
            aw_ind = 0
            for k in range(a0, a1):
                if int(datree.access_to[k]) == int(ind_s[ci]):
                    aw_ind = int(datree.access_w[k]); break
            tr_ind = jt._trace_from(int(ind_s[ci]), aw_ind, int(_D[ci]))
            ind_dom = datree._dominant(tr_ind[0]) if tr_ind else None
            if kern_dom != ind_dom:
                lmis += 1
        assert checked > 500, f"{name}: only {checked} reachable cells"
        assert smis == 0, f"{name}: kernel s* != independent s* on {smis}/{checked} cells"
        assert lmis == 0, f"{name}: dominant line drift on {lmis}/{checked} cells"
        tot_checked += checked; tot_smis += smis; tot_lmis += lmis
    assert tot_checked > 5000 and tot_smis == 0 and tot_lmis == 0


@pytest.mark.skipif(not _oracles(), reason="no R5 oracles in tests/raptor_golden/")
def test_departafter_selection_catches_wrong_stop(da_trees):
    """Proves the C1 cross-check has TEETH (not vacuous): a deliberately-corrupted selection that
    picks a DIFFERENT in-cell access stop than the penalized argmin must make the traced dominant
    line disagree with the independent reference on a meaningful fraction of multi-stop cells. If
    this can't find disagreement, the dominant-line cross-check above could never fail and would be
    a tautology — so we assert it CAN."""
    name, (datree, _pd, _ids) = next(iter(da_trees.items()))
    s_star, _aw, _D, _T, is_walk, painted = datree._select_arrays()
    flipped = 0; tried = 0
    for ci in range(datree.n_cells):
        if painted[ci] < 0 or bool(is_walk[ci]) or int(s_star[ci]) < 0:
            continue
        a0, a1 = int(datree.access_off[ci]), int(datree.access_off[ci + 1])
        # pick a DIFFERENT in-cell access stop (the corruption), reachable at this T*
        T = int(_T[ci]); jt = datree._tree_at(T)
        alt_stop = alt_aw = None
        for k in range(a0, a1):
            g = int(datree.access_to[k])
            if g != int(s_star[ci]) and int(jt.best[g]) > (-(1 << 50)):
                alt_stop = g; alt_aw = int(datree.access_w[k]); break
        if alt_stop is None:
            continue
        tried += 1
        kern_dom = datree._dominant(datree._trace_raw(ci)[0])
        tr_bad = jt._trace_from(alt_stop, alt_aw, int(_D[ci]))
        bad_dom = datree._dominant(tr_bad[0]) if tr_bad else None
        if bad_dom != kern_dom:
            flipped += 1
        if flipped >= 20 and tried >= 50:
            break
    assert tried >= 20, f"{name}: too few multi-stop cells to corrupt ({tried})"
    assert flipped > 0, (f"{name}: a wrong-stop pick NEVER changed the dominant line over {tried} "
                         f"cells — the C1 cross-check would be vacuous")


@pytest.mark.skipif(not _oracles(), reason="no R5 oracles in tests/raptor_golden/")
def test_departafter_walkspeed_monotonicity(engine):
    """C2: walking FASTER must never LENGTHEN the depart-after commute (a faster walk can only board
    sooner / egress sooner). Scans the painted depart-after map across slow->med->fast over the
    default workplace subset and asserts every reachable cell is monotone non-increasing as speed
    rises. Mirrors the arrive-by monotonicity guard; EXPECTED true-zero for depart-after (the
    percentile-over-window value has no single-departure 'latest run' jiggle), so STRICT == 0."""
    from core import raptor_golden
    order = ["slow", "med", "fast"]            # increasing speed = non-increasing commute
    grand = viol = 0
    for fname in _DEFAULT_SUBSET:
        if fname not in set(_oracles()):
            continue
        z = np.load(os.path.join(GOLDEN, fname), allow_pickle=True)
        pw = raptor_golden.purewalk_aligned(engine, z)
        cm = {}
        for sp in order:
            datree = engine.journey_tree_departafter(z["egress_g"], z["egress_w"], pw,
                                                     percentile=50, walk_scalar=SPEED_SCALAR[sp])
            cm[sp] = datree.commute()
        n = len(cm["slow"])
        for ci in range(n):
            for a, b in zip(order, order[1:]):
                va, vb = int(cm[a][ci]), int(cm[b][ci])
                if va < 0 or vb < 0:
                    continue                    # reachability can only improve with faster walk
                grand += 1
                if vb > va:                     # faster walk made it LONGER -> violation
                    viol += 1
    assert grand > 1000, f"only {grand} comparisons — fixture too thin"
    assert viol == 0, f"depart-after walk-speed monotonicity: {viol}/{grand} faster-walk-longer cells"


@pytest.mark.slow
@pytest.mark.skipif(not _oracles(), reason="no R5 oracles in tests/raptor_golden/")
def test_departafter_hover_equals_map_all_speeds(engine):
    """The depart-after hover==map invariant across slow/med/fast walk speeds (the walk-speed toggle
    must hold end-to-end), over the default workplace subset. STRICT: 0 violations per speed."""
    from core import raptor_golden
    grand = 0
    for fname in _DEFAULT_SUBSET:
        if fname not in set(_oracles()):
            continue
        z = np.load(os.path.join(GOLDEN, fname), allow_pickle=True)
        pw = raptor_golden.purewalk_aligned(engine, z)
        ids = engine.cell_ids
        for sp, scalar in SPEED_SCALAR.items():
            datree = engine.journey_tree_departafter(z["egress_g"], z["egress_w"], pw,
                                                     percentile=50, walk_scalar=scalar)
            painted = engine.departafter(z["egress_g"], z["egress_w"], pw, percentiles=(5, 50),
                                         walk_scalar=scalar)
            commute, _dom = datree.commute_and_dominant()
            checked = viol = 0
            for ci in range(datree.n_cells):
                pv = painted[ids[ci]][1]
                cv = int(commute[ci]) if commute[ci] >= 0 else None
                if pv is None and cv is None:
                    continue
                checked += 1
                if pv != cv:
                    viol += 1
                    continue
                it = datree.itinerary(ci)
                if it is None or it["total"] != cv:
                    viol += 1
            assert checked > 500, f"{fname}/{sp}: only {checked} reachable cells"
            assert viol == 0, f"{fname}/{sp}: depart-after hover!=map on {viol}/{checked}"
            grand += checked
    assert grand > 3000
