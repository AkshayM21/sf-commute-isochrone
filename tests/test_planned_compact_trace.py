"""Differential contract for the isolated planned-overlay compact parent-chain primitive."""
import threading

import numpy as np

from core.raptor_planned import (
    PLANNED_TRACE_INVALID_BOARD,
    PLANNED_TRACE_MALFORMED,
    PLANNED_TRACE_OK,
    PLANNED_TRACE_REANCHORED,
    PLANNED_TRACE_UNREACHABLE,
    trace_planned_stop_templates_python,
)


EGRESS_INF = np.int64(1 << 40)


def _one_ride_fixture():
    # stop 0 walks 30s to stop 1, rides pattern 0, then walks from its best downstream stop.
    # stop 3 has a malformed self-cycle; stop 4 is a direct 45s egress chain.
    return {
        "tree_best": np.array([60, 60, -1, 50, 50], np.int64),
        "best_node": np.array([0, 0, -1, 3, 4], np.int32),
        "nd_kind": np.array([2, 1, 0, 2, 0], np.int8),
        "nd_stop": np.array([0, 1, 3, 0, 4], np.int32),
        "nd_pat": np.array([-1, 0, -1, -1, -1], np.int32),
        "nd_trip": np.array([-1, 0, -1, -1, -1], np.int32),
        "nd_board": np.array([-1, 0, -1, -1, -1], np.int32),
        "nd_alight": np.array([-1, 2, -1, -1, -1], np.int32),
        "nd_to": np.array([1, -1, -1, 1, -1], np.int32),
        "nd_egress": np.array([-1, -1, 100, -1, 45], np.int64),
        "nd_next": np.array([1, 2, -1, 3, -1], np.int32),
        "pat_nstops": np.array([3], np.int32),
        "pat_ntrips": np.array([1], np.int32),
        "pat_mat_off": np.array([0], np.int64),
        "pat_stop_off": np.array([0], np.int64),
        "pat_stops": np.array([1, 2, 3], np.int32),
        "pat_dep": np.array([100, 200, 205], np.int64),
        "pat_arr": np.array([100, 200, 205], np.int64),
        "tr_off": np.array([0, 1, 1, 1, 1, 1], np.int64),
        "tr_to": np.array([1], np.int32),
        "tr_time": np.array([30], np.int64),
        "egress_sec": np.array([EGRESS_INF, EGRESS_INF, 0, 100, 45], np.int64),
        "pattern_structural_rank": np.array([0], np.int64),
    }


def _call(fn, fixture, stops, boards, floor=50, max_rides=8):
    f = fixture
    return fn(
        np.asarray(stops, np.int64), np.asarray(boards, np.int64), int(floor),
        f["tree_best"], f["best_node"], f["nd_kind"], f["nd_stop"], f["nd_pat"],
        f["nd_trip"], f["nd_board"], f["nd_alight"], f["nd_to"], f["nd_egress"],
        f["nd_next"], f["pat_nstops"], f["pat_ntrips"], f["pat_mat_off"], f["pat_stop_off"],
        f["pat_stops"], f["pat_dep"], f["pat_arr"], f["tr_off"], f["tr_to"],
        f["tr_time"], f["egress_sec"], f["pattern_structural_rank"], EGRESS_INF,
        int(max_rides))


def _assert_parallel_equal(actual, expected):
    assert len(actual) == len(expected) == 12
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)


def test_compact_trace_oracle_statuses_reanchor_and_exact_no_overshoot():
    result = _call(
        trace_planned_stop_templates_python, _one_ride_fixture(),
        stops=[0, 0, 2, 3, 4], boards=[50, 90, 50, 50, 50])
    (status, effective, root, tail, transit, final_walk, ride_count,
     ride_pi, ride_trip, ride_bpos, ride_apos, representative) = result

    np.testing.assert_array_equal(
        status,
        [PLANNED_TRACE_OK, PLANNED_TRACE_REANCHORED, PLANNED_TRACE_UNREACHABLE,
         PLANNED_TRACE_MALFORMED, PLANNED_TRACE_OK])
    np.testing.assert_array_equal(effective, [50, 60, 50, 50, 50])
    np.testing.assert_array_equal(root, [0, 0, -1, 3, 4])
    # The final ride's tree alight is position 2 (finish 305), but exact no-overshoot chooses
    # position 1 (finish 200). Tail includes the 30s transfer walk and scheduled wait from B.
    assert (tail[0], transit[0], final_walk[0]) == (150, 100, 0)
    assert (ride_count[0], ride_pi[0, 0], ride_trip[0, 0]) == (1, 0, 0)
    assert (ride_bpos[0, 0], ride_apos[0, 0], representative[0]) == (0, 1, 0)
    assert (tail[4], transit[4], final_walk[4], ride_count[4]) == (45, 0, 45, 0)


def test_compact_trace_oracle_drops_reanchor_below_window_floor():
    result = _call(
        trace_planned_stop_templates_python, _one_ride_fixture(),
        stops=[0], boards=[90], floor=70)
    assert result[0].tolist() == [PLANNED_TRACE_INVALID_BOARD]
    # A rejected re-anchor never silently changes the requested anchor.
    assert result[1].tolist() == [90]


def test_compact_trace_oracle_preserves_missing_footpath_zero_rule_and_tape_overflow():
    fixture = _one_ride_fixture()
    fixture["tr_to"] = np.array([99], np.int32)  # no 0->1 edge: scalar tracer deliberately uses 0
    ok = _call(trace_planned_stop_templates_python, fixture, stops=[0], boards=[90])
    assert ok[0].tolist() == [PLANNED_TRACE_OK]
    assert ok[3].tolist() == [110]                # wait 10 + ride 100 + final walk 0

    overflow = _call(
        trace_planned_stop_templates_python, fixture, stops=[0], boards=[90], max_rides=0)
    assert overflow[0].tolist() == [PLANNED_TRACE_MALFORMED]

    fixture = _one_ride_fixture()
    fixture["nd_trip"][1] = 1  # pat_ntrips[0] == 1: never spill into a following schedule row
    invalid_trip = _call(
        trace_planned_stop_templates_python, fixture, stops=[0], boards=[50])
    assert invalid_trip[0].tolist() == [PLANNED_TRACE_MALFORMED]
    from core.raptor_planned_numba import trace_planned_stop_templates
    _assert_parallel_equal(
        _call(trace_planned_stop_templates, fixture, stops=[0], boards=[50]), invalid_trip)


def test_compact_trace_oracle_no_overshoot_uses_walk_then_later_position_ties():
    fixture = _one_ride_fixture()
    fixture["nd_alight"][1] = 1
    fixture["pat_arr"] = np.array([100, 200, 205], np.int64)
    fixture["egress_sec"] = np.array([EGRESS_INF, EGRESS_INF, 10, 5, 45], np.int64)
    # Both finish at 210, so the lower egress walk wins: position 2.
    result = _call(trace_planned_stop_templates_python, fixture, stops=[0], boards=[50])
    assert result[10][0, 0] == 2
    assert result[5][0] == 5

    # Equal finish and equal walk choose the later position, matching _min_overshoot_alight.
    fixture["pat_arr"] = np.array([100, 200, 200], np.int64)
    fixture["egress_sec"][3] = 10
    result = _call(trace_planned_stop_templates_python, fixture, stops=[0], boards=[50])
    assert result[10][0, 0] == 2
    assert result[5][0] == 10


def test_compact_trace_compiled_matches_oracle_for_every_field():
    from core.raptor_planned_numba import trace_planned_stop_templates

    fixture = _one_ride_fixture()
    expected = _call(
        trace_planned_stop_templates_python, fixture,
        stops=[0, 0, 2, 3, 4], boards=[50, 90, 50, 50, 50])
    actual = _call(
        trace_planned_stop_templates, fixture,
        stops=[0, 0, 2, 3, 4], boards=[50, 90, 50, 50, 50])
    _assert_parallel_equal(actual, expected)


def test_compact_trace_tape_keeps_occurrences_and_structural_rank_only_breaks_ties():
    fixture = {
        "tree_best": np.array([50], np.int64),
        "best_node": np.array([0], np.int32),
        "nd_kind": np.array([1, 1, 0], np.int8),
        "nd_stop": np.array([0, 1, 2], np.int32),
        "nd_pat": np.array([0, 1, -1], np.int32),
        "nd_trip": np.array([0, 0, -1], np.int32),
        "nd_board": np.array([0, 0, -1], np.int32),
        "nd_alight": np.array([1, 1, -1], np.int32),
        "nd_to": np.array([-1, -1, -1], np.int32),
        "nd_egress": np.array([-1, -1, 0], np.int64),
        "nd_next": np.array([1, 2, -1], np.int32),
        "pat_nstops": np.array([2, 2], np.int32),
        "pat_ntrips": np.array([1, 1], np.int32),
        "pat_mat_off": np.array([0, 2], np.int64),
        "pat_stop_off": np.array([0, 2], np.int64),
        "pat_stops": np.array([0, 1, 1, 2], np.int32),
        "pat_dep": np.array([100, 160, 200, 260], np.int64),
        "pat_arr": np.array([100, 160, 200, 260], np.int64),
        "tr_off": np.array([0, 0, 0, 0], np.int64),
        "tr_to": np.empty(0, np.int32),
        "tr_time": np.empty(0, np.int64),
        "egress_sec": np.array([EGRESS_INF, EGRESS_INF, 0], np.int64),
        "pattern_structural_rank": np.array([9, 2], np.int64),
    }
    oracle = _call(trace_planned_stop_templates_python, fixture, stops=[0], boards=[50])
    assert oracle[0].tolist() == [PLANNED_TRACE_OK]
    assert oracle[6].tolist() == [2]
    assert oracle[7][0, :2].tolist() == [0, 1]  # occurrences never collapse to a label/id
    assert oracle[11].tolist() == [1]           # equal ride duration; structural rank 2 beats 9

    from core.raptor_planned_numba import trace_planned_stop_templates
    _assert_parallel_equal(
        _call(trace_planned_stop_templates, fixture, stops=[0], boards=[50]), oracle)


def _overlay_tree(reanchor=False):
    """One-cell citywide-overlay fixture backed by the same compact arrays as the oracle tests."""
    from core.raptor_journey import DepartAfterJourneyTree

    fixture = _one_ride_fixture()
    data = dict(fixture)
    data.update({
        "n_stops": 5,
        "line_table": [("fixture-feed", "fixture-route", "ALPHA")],
        "pat_line": np.array([0], np.int32),
    })
    par = {name: fixture[name] for name in (
        "best_node", "nd_kind", "nd_stop", "nd_pat", "nd_trip", "nd_board",
        "nd_alight", "nd_to", "nd_egress", "nd_next")}
    calls = []

    class FakeJourneyTree:
        best = fixture["tree_best"]
        egress_sec = fixture["egress_sec"]

        def __init__(self):
            self.par = par

        def _trace_from(self, stop, access_walk, home):
            calls.append((int(stop), int(access_walk), int(home)))
            if int(stop) in (0, 1):
                return ([
                    ("access", int(access_walk)),
                    ("walk_t", 30, 0, 1),
                    ("ride", 0, 100, 200, 0, 1, 2),
                    ("egress", 0, 2),
                ], int(home))
            if int(stop) == 4:
                return ([("access", int(access_walk)), ("egress", 45, 4)], int(home))
            return None

    tree = DepartAfterJourneyTree.__new__(DepartAfterJourneyTree)
    tree.d = data
    tree.line_table = data["line_table"]
    tree.pat_line = data["pat_line"]
    tree._planned_pattern_identity_cache = {}
    tree._planned_structural_rank_cache = None
    tree._planned_validated_stop = {}
    # Keep this hand-built tree aligned with the production object's cache ownership contract.
    tree._planned_validated_stop_lock = threading.RLock()
    tree.cell_deps = np.array([50, 90], np.int64)
    tree.max_min = 120
    tree.n_cells = 1
    tree.arrivalW = np.zeros((5, 1), np.int64)
    tree.access_off = np.array([0, 3], np.int64)
    tree.access_to = np.array([0, 1, 4], np.int64)
    tree.access_w = np.array([10, 20, 20], np.int64)
    tree._tree_at = lambda _deadline: FakeJourneyTree()
    tree._select_arrays = lambda: (
        np.array([0]), np.array([10]), np.array([40]), np.array([300]),
        np.array([False]), np.array([4], np.int32))
    best_B = np.array([90 if reanchor else 50, 50, -1, -1, 50], np.int64)
    best_T = np.array([300, 300, -1, -1, 95], np.int64)
    stop_tail = np.full(5, EGRESS_INF, np.int64)
    usable = best_B >= 0
    stop_tail[usable] = best_T[usable] - best_B[usable]
    tree._planned_stop_anchors = lambda: (stop_tail, best_B, best_T)
    return tree, calls


def _two_ride_overlay_tree():
    """One overlay candidate with a same-labelled, two-ride transfer route."""
    from core.raptor_journey import DepartAfterJourneyTree

    fixture = {
        "tree_best": np.array([50, 50, 50, 50], np.int64),
        "best_node": np.array([0, 1, 2, 3], np.int32),
        "nd_kind": np.array([2, 1, 1, 0], np.int8),
        "nd_stop": np.array([0, 1, 2, 3], np.int32),
        "nd_pat": np.array([-1, 0, 1, -1], np.int32),
        "nd_trip": np.array([-1, 0, 0, -1], np.int32),
        "nd_board": np.array([-1, 0, 0, -1], np.int32),
        "nd_alight": np.array([-1, 1, 1, -1], np.int32),
        "nd_to": np.array([1, -1, -1, -1], np.int32),
        "nd_egress": np.array([-1, -1, -1, 0], np.int64),
        "nd_next": np.array([1, 2, 3, -1], np.int32),
        "pat_nstops": np.array([2, 2], np.int32),
        "pat_ntrips": np.array([1, 1], np.int32),
        "pat_mat_off": np.array([0, 2], np.int64),
        "pat_stop_off": np.array([0, 2], np.int64),
        "pat_stops": np.array([1, 2, 2, 3], np.int32),
        "pat_dep": np.array([100, 200, 210, 260], np.int64),
        "pat_arr": np.array([100, 200, 210, 260], np.int64),
        "tr_off": np.array([0, 1, 1, 1, 1], np.int64),
        "tr_to": np.array([1], np.int32),
        "tr_time": np.array([30], np.int64),
        "egress_sec": np.array([EGRESS_INF, EGRESS_INF, EGRESS_INF, 0], np.int64),
    }
    data = dict(fixture)
    data.update({
        # The display label intentionally repeats; route identity must still retain both rides.
        "n_stops": 4,
        "line_table": [("fixture-feed", "route-a", "ALPHA"),
                       ("fixture-feed", "route-b", "ALPHA")],
        "pat_line": np.array([0, 1], np.int32),
    })
    par = {name: fixture[name] for name in (
        "best_node", "nd_kind", "nd_stop", "nd_pat", "nd_trip", "nd_board",
        "nd_alight", "nd_to", "nd_egress", "nd_next")}
    calls = []

    class FakeJourneyTree:
        best = fixture["tree_best"]
        egress_sec = fixture["egress_sec"]

        def __init__(self):
            self.par = par

        def _trace_from(self, stop, access_walk, home):
            calls.append((int(stop), int(access_walk), int(home)))
            if int(stop) != 0:
                return None
            return ([
                ("access", int(access_walk)),
                ("walk_t", 30, 0, 1),
                ("ride", 0, 100, 200, 0, 1, 2),
                ("ride", 1, 210, 260, 0, 2, 3),
                ("egress", 0, 3),
            ], int(home))

    tree = DepartAfterJourneyTree.__new__(DepartAfterJourneyTree)
    tree.d = data
    tree.line_table = data["line_table"]
    tree.pat_line = data["pat_line"]
    tree._planned_pattern_identity_cache = {}
    tree._planned_structural_rank_cache = None
    tree._planned_validated_stop = {}
    tree._planned_validated_stop_lock = threading.RLock()
    tree.cell_deps = np.array([50], np.int64)
    tree.max_min = 120
    tree.n_cells = 1
    tree.arrivalW = np.zeros((4, 1), np.int64)
    tree.access_off = np.array([0, 1], np.int64)
    tree.access_to = np.array([0], np.int64)
    tree.access_w = np.array([10], np.int64)
    tree._tree_at = lambda _deadline: FakeJourneyTree()
    tree._select_arrays = lambda: (
        np.array([0]), np.array([10]), np.array([40]), np.array([300]),
        np.array([False]), np.array([5], np.int32))
    tree._planned_stop_anchors = lambda: (
        np.array([250, EGRESS_INF, EGRESS_INF, EGRESS_INF], np.int64),
        np.array([50, -1, -1, -1], np.int64),
        np.array([300, -1, -1, -1], np.int64))
    return tree, calls


def test_planned_template_mode_defaults_to_compact_and_invalid_values_roll_back(monkeypatch):
    from core.raptor_journey import DepartAfterJourneyTree

    monkeypatch.delenv("RAPTOR_PLANNED_TEMPLATE_MODE", raising=False)
    assert DepartAfterJourneyTree._planned_template_mode() == "compact"
    monkeypatch.setenv("RAPTOR_PLANNED_TEMPLATE_MODE", "compact")
    assert DepartAfterJourneyTree._planned_template_mode() == "compact"
    monkeypatch.setenv("RAPTOR_PLANNED_TEMPLATE_MODE", "legacy")
    assert DepartAfterJourneyTree._planned_template_mode() == "legacy"
    monkeypatch.setenv("RAPTOR_PLANNED_TEMPLATE_MODE", "not-a-mode")
    assert DepartAfterJourneyTree._planned_template_mode() == "legacy"


def test_compact_default_and_explicit_selector_preserve_legacy_rollback(monkeypatch):
    monkeypatch.delenv("RAPTOR_PLANNED_TEMPLATE_MODE", raising=False)
    default_tree, default_calls = _overlay_tree()
    assert default_tree._planned_alt_lines_window(np.array([4]), 2) == [{"ALPHA": [3, 0]}]
    assert default_calls == []

    monkeypatch.setenv("RAPTOR_PLANNED_TEMPLATE_MODE", "compact")
    explicit_tree, explicit_calls = _overlay_tree()
    assert explicit_tree._planned_alt_lines_window(np.array([4]), 2) == [{"ALPHA": [3, 0]}]
    assert explicit_calls == []

    monkeypatch.setenv("RAPTOR_PLANNED_TEMPLATE_MODE", "legacy")
    legacy_tree, legacy_calls = _overlay_tree()
    assert legacy_tree._planned_alt_lines_window(np.array([4]), 2) == [{"ALPHA": [3, 0]}]
    assert len(legacy_calls) == 3

    monkeypatch.setenv("RAPTOR_PLANNED_TEMPLATE_MODE", "mistyped-compact")
    invalid_tree, invalid_calls = _overlay_tree()
    assert invalid_tree._planned_alt_lines_window(np.array([4]), 2) == [{"ALPHA": [3, 0]}]
    assert len(invalid_calls) == 3


def test_compact_two_ride_same_label_transfer_matches_scalar_overlay(monkeypatch):
    monkeypatch.setenv("RAPTOR_PLANNED_TEMPLATE_MODE", "legacy")
    legacy, legacy_calls = _two_ride_overlay_tree()
    expected = legacy._planned_alt_lines_window(np.array([5]), 2)
    assert expected == [{"ALPHA > ALPHA": [4, 0]}]
    assert legacy_calls == [(0, 0, 50)]

    monkeypatch.setenv("RAPTOR_PLANNED_TEMPLATE_MODE", "compact")
    compact, compact_calls = _two_ride_overlay_tree()
    assert compact._planned_alt_lines_window(np.array([5]), 2) == expected
    assert compact_calls == []
    lazy = compact._planned_validated_stop[0]
    assert lazy.label == "ALPHA > ALPHA"
    assert len(lazy.route_key) == 2


def test_compact_overlay_matches_legacy_without_materializing_candidate_raw_tails(monkeypatch):
    legacy, legacy_calls = _overlay_tree()
    monkeypatch.setenv("RAPTOR_PLANNED_TEMPLATE_MODE", "legacy")
    expected = legacy._planned_alt_lines_window(np.array([4]), 2)
    assert expected == [{"ALPHA": [3, 0]}]
    assert len(legacy_calls) == 3                 # both transit stops plus walk-only stop

    compact, compact_calls = _overlay_tree()
    monkeypatch.setenv("RAPTOR_PLANNED_TEMPLATE_MODE", "compact")
    actual = compact._planned_alt_lines_window(np.array([4]), 2)
    assert actual == expected
    assert compact_calls == []                   # citywide overlay used only compact arrays

    # A later selected pin/itinerary boundary materializes exactly its chosen stop once and
    # presents the unchanged scalar five-tuple to downstream code.
    lazy = compact._planned_validated_stop[0]
    materialized = compact._planned_template_with_raw(0, lazy)
    assert len(compact_calls) == 1
    assert materialized == legacy._planned_validated_stop[0]
    assert compact._planned_template_with_raw(0, lazy) == materialized
    assert len(compact_calls) == 1


def test_compact_overlay_reanchor_matches_legacy_effective_board_and_order(monkeypatch):
    legacy, _legacy_calls = _overlay_tree(reanchor=True)
    monkeypatch.setenv("RAPTOR_PLANNED_TEMPLATE_MODE", "legacy")
    expected = legacy._planned_alt_lines_window(np.array([4]), 2)

    compact, compact_calls = _overlay_tree(reanchor=True)
    monkeypatch.setenv("RAPTOR_PLANNED_TEMPLATE_MODE", "compact")
    actual = compact._planned_alt_lines_window(np.array([4]), 2)
    assert actual == expected == [{"ALPHA": [3, 0]}]
    assert compact._planned_template_summary(compact._planned_validated_stop[0])[0] == 60
    assert compact_calls == []


def test_compact_lazy_materialization_falls_back_when_boardable_route_summary_changed(monkeypatch):
    compact, compact_calls = _overlay_tree()
    monkeypatch.setenv("RAPTOR_PLANNED_TEMPLATE_MODE", "compact")
    assert compact._planned_alt_lines_window(np.array([4]), 2) == [{"ALPHA": [3, 0]}]
    lazy = compact._planned_validated_stop[0]

    class ChangedJourneyTree:
        best = np.array([60, 60, -1, 50, 50], np.int64)

        def _trace_from(self, stop, access_walk, home):
            compact_calls.append((int(stop), int(access_walk), int(home)))
            # Still boardable from B=50 and still publicly ALPHA, but it now rides to position 2.
            # The compact summary captured position 1, so accepting this raw tail would combine a
            # stale structural route key with a different itinerary.
            return ([
                ("access", int(access_walk)),
                ("walk_t", 30, 0, 1),
                ("ride", 0, 100, 205, 0, 2, 3),
                ("egress", 100, 3),
            ], int(home))

    changed = ChangedJourneyTree()
    compact._tree_at = lambda _deadline: changed
    materialized = compact._planned_template_with_raw(0, lazy)

    # One failed lazy check plus one scalar-oracle trace.  The cache ends on the scalar tuple, whose
    # raw route and structural key agree with each other rather than the stale compact summary.
    assert compact_calls == [(0, 0, 50), (0, 0, 50)]
    assert materialized is compact._planned_validated_stop[0]
    B, T, raw_tail, label, route_key = materialized
    assert (B, T, label) == (50, 300, "ALPHA")
    assert raw_tail[-2][5] == 2
    assert route_key == compact._planned_route_identity(raw_tail)
    assert route_key != lazy.route_key


def test_compact_stale_lazy_holder_does_not_delete_newer_scalar_cache_row(monkeypatch):
    compact, compact_calls = _overlay_tree()
    monkeypatch.setenv("RAPTOR_PLANNED_TEMPLATE_MODE", "compact")
    assert compact._planned_alt_lines_window(np.array([4]), 2) == [{"ALPHA": [3, 0]}]
    lazy = compact._planned_validated_stop[0]

    # Model another consumer winning the fallback race on this same tree.  The older lazy holder
    # has already learned that its raw chain is unusable, while the shared cache now contains the
    # authoritative scalar result.
    lazy._raw_tail = None
    scalar = (
        50,
        300,
        (("walk_t", 30, 0, 1), ("ride", 0, 100, 200, 0, 1, 2),
         ("egress", 0, 2)),
        "ALPHA",
        compact._planned_route_identity(
            (("walk_t", 30, 0, 1), ("ride", 0, 100, 200, 0, 1, 2),
             ("egress", 0, 2))),
    )
    compact._planned_validated_stop[0] = scalar

    assert compact._planned_template_with_raw(0, lazy) is scalar
    assert compact._planned_validated_stop[0] is scalar
    # The already-resolved scalar cache row prevents a duplicate raw trace.
    assert compact_calls == []


def test_compact_concurrent_materialization_publishes_one_scalar_cache_row(monkeypatch):
    """Concurrent stale lazy readers normalize the shared row, never reintroduce a lazy holder.

    Both consumers start from the exact same compact summary and deadline tree.  The start barrier
    makes the cache publication race reproducible without relying on scheduler timing: whichever
    materializer wins must publish the canonical scalar tuple, and the other must return that same
    row rather than leaving (or restoring) the compact holder in the stop cache.
    """
    compact, _compact_calls = _overlay_tree()
    monkeypatch.setenv("RAPTOR_PLANNED_TEMPLATE_MODE", "compact")
    assert compact._planned_alt_lines_window(np.array([4]), 2) == [{"ALPHA": [3, 0]}]
    lazy = compact._planned_validated_stop[0]

    # Keep both callers on one logical T* tree; the fixture normally constructs an equivalent
    # fake tree for each lookup, which would make this concurrency contract less explicit.
    same_tree = compact._tree_at(lazy.deadline)
    compact._tree_at = lambda _deadline: same_tree

    start = threading.Barrier(2)
    results = [None, None]
    errors = []

    def materialize(slot):
        try:
            start.wait(timeout=5)
            results[slot] = compact._planned_template_with_raw(0, lazy)
        except BaseException as exc:  # keep worker failures visible in the main test thread
            errors.append(exc)

    workers = [threading.Thread(target=materialize, args=(slot,)) for slot in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)

    assert not any(worker.is_alive() for worker in workers)
    assert errors == []
    scalar = compact._planned_validated_stop[0]
    assert isinstance(scalar, tuple)
    assert results == [scalar, scalar]
    assert results[0] is results[1] is scalar


def test_compact_delayed_group_publish_cannot_return_lazy_holder_after_failed_materialization(
        monkeypatch):
    """A slow bulk extractor must not slip a compact holder into a scalar-only fallback.

    The slow thread observes an empty stop cache and computes its compact summary, but pauses
    before its final ``setdefault``.  The foreground request first publishes a holder, discovers
    that its lazy raw tail is stale, and forces scalar fallback.  Before that fallback reads the
    cache, the slow group is allowed to publish.  The pre-lock implementation returns the newly
    published holder directly from ``_validated_stop_template_from_anchor``; a scalar consumer
    then crashes when it tuple-unpacks it.  A cache lock may serialize the fallback ahead of the
    slow publication; the short wait intentionally allows that correct ordering too.
    """
    import core.raptor as raptor
    import core.raptor_planned_numba as planned_numba

    compact, _calls = _overlay_tree()
    monkeypatch.setenv("RAPTOR_PLANNED_TEMPLATE_MODE", "compact")
    monkeypatch.setattr(raptor, "_select_kernel", lambda: "numba")

    slow_scanned = threading.Event()
    release_slow = threading.Event()
    slow_finished = threading.Event()
    slow_errors = []
    oracle = trace_planned_stop_templates_python

    def staged_trace(*args, **kwargs):
        if threading.current_thread().name == "delayed-compact-group":
            slow_scanned.set()
            assert release_slow.wait(timeout=5)
        return oracle(*args, **kwargs)

    monkeypatch.setattr(planned_numba, "trace_planned_stop_templates", staged_trace)
    _tail, best_board, best_deadline = compact._planned_stop_anchors()

    def publish_delayed_group():
        try:
            compact._validated_stop_templates_grouped_compact(
                np.array([0], np.int64), best_board, best_deadline)
        except BaseException as exc:
            slow_errors.append(exc)
        finally:
            slow_finished.set()

    slow = threading.Thread(target=publish_delayed_group, name="delayed-compact-group")
    slow.start()
    assert slow_scanned.wait(timeout=5)

    assert compact._planned_alt_lines_window(np.array([4]), 2) == [{"ALPHA": [3, 0]}]
    stale = compact._planned_validated_stop[0]
    stale._raw_tail = None                       # force the stale-holder scalar-oracle path
    scalar_oracle = compact._validated_stop_template_from_anchor

    def release_then_fallback(*args, **kwargs):
        # Without cache serialization the delayed group can publish its holder before the scalar
        # oracle owns the row.  With serialization it waits, then preserves the scalar winner.
        release_slow.set()
        slow_finished.wait(timeout=0.25)
        return scalar_oracle(*args, **kwargs)

    monkeypatch.setattr(compact, "_validated_stop_template_from_anchor", release_then_fallback)
    result = compact._planned_template_with_raw(0, stale)
    slow.join(timeout=5)

    assert not slow.is_alive()
    assert slow_errors == []
    # This is the scalar-only boundary used by _validated_stop_choice: never leak a holder here.
    assert isinstance(result, tuple)
    assert result is compact._planned_validated_stop[0]
    assert not hasattr(result, "materialize_raw_tail")
