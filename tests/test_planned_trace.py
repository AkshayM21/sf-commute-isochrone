"""Planned-mode trace/display regressions (review 2026-07-11, Fixer B: B1-B4).

Pure-python fixtures in the style of test_route_family_selection.py: a minimal synthetic
DepartAfterJourneyTree whose per-cell selection (``_sel``) and per-T* JourneyTree are fabricated
directly, so NO RAPTOR kernel, NO numba, NO JVM runs — only the reconstruction/display machinery
under test (planned-primary selection / ``_fold_first_visible_wait`` / ``_validated_stop_anchor`` /
``_fill_committed_leg`` and the real ``_clock``/``_format``/``_geometry``/``reconcile_legs``).

The scenario grid (seconds): one cell, one access stop (gid 0, 180 s walk), boarding-window
minute B = 28800. The kernel anchor is ``D* = B - access_walk = 28620`` and the painted planned
minute is ``ceil((T - B + access_walk)/60)`` (``select_planned_departafter``'s objective).
"""
import os
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from core.raptor_journey import EGRESS_INF, JourneyTree
from core.raptor_journey_da import DepartAfterJourneyTree

_INF = np.int64(1 << 60)
_NEGBIG = -(1 << 40)

# Two patterns: pattern 0 = Muni "22" over stop gids 0..3, pattern 1 = Muni "19" over gids 4..5.
_DATA = {
    "line_table": [("muni_current", "22", "22", "bus"), ("muni_current", "19", "19", "bus")],
    "pat_line": np.array([0, 1], np.int64),
    "pat_stop_off": np.array([0, 4], np.int64),
    "pat_stops": np.array([0, 1, 2, 3, 4, 5], np.int64),
    "stop_lat": np.array([37.70, 37.71, 37.72, 37.73, 37.74, 37.75]),
    "stop_lon": np.array([-122.40, -122.41, -122.42, -122.43, -122.44, -122.45]),
    "stop_name": np.array(["Origin Platform", "Transfer Platform", "Civic Platform",
                           "North Terminal", "Connector Platform", "East Terminal"]),
}

# The clean primary journey: 3-min access walk, 2-min genuine pre-board wait (board 28920 at
# window minute B=28800), an EXACT 12-minute ride (720 s), 2-min egress. Arrives W at 29760.
RIDE_EXACT = ("ride", 0, 28920, 29640, 0, 3, 3)
LEGS_CLEAN = [("access", 180), RIDE_EXACT, ("egress", 120, 3)]

# A raw chain whose first board departs BEFORE the anchored window minute 28800 (the cummax'd
# profile row vs raw single-deadline tree divergence B3 guards against).
LEGS_BAD = [("access", 180), ("ride", 0, 28700, 29640, 0, 3, 3), ("egress", 120, 3)]

# Tiny-first-hop journey for B4: a 60 s hop on the "22" (displayed un-folded in planned mode),
# a 1-min transfer walk, then the significant "19" ride. Arrives W at 30000.
RIDE_TINY = ("ride", 0, 28920, 28980, 0, 1, 1)
RIDE_BIG = ("ride", 1, 29100, 29940, 0, 1, 5)
LEGS_TINY_FIRST = [("access", 180), RIDE_TINY, ("walk_t", 60, 1, 4), RIDE_BIG, ("egress", 60, 5)]


class _Prov:
    """Geometry provider stub: every walk kind returns a small non-empty polyline."""

    def access(self, ci, s):
        return [[37.700, -122.400], [37.701, -122.401]], False

    def purewalk(self, ci):
        return [[37.700, -122.400], [37.702, -122.402]], True

    def transfer(self, s, j):
        return [[37.710, -122.410], [37.711, -122.411]], False

    def egress(self, s):
        return [[37.730, -122.430], [37.731, -122.431]], False


def _stub_jt(legs_raw, best0=_NEGBIG, max_min=75, data=_DATA):
    """A real JourneyTree (real _clock/_format/_geometry) with a fabricated node-chain trace."""
    jt = JourneyTree.__new__(JourneyTree)
    jt.d = data
    jt.line_table = data["line_table"]
    jt.pat_line = data["pat_line"]
    jt.max_min = max_min
    best = np.full(8, _NEGBIG, np.int64)
    best[0] = best0
    jt.best = best
    if legs_raw is not None:
        jt._trace_from = lambda s, aw, home, _lr=list(legs_raw): (list(_lr), int(home))
    return jt


def _mk_tree(legs_raw, aw=180, B=28800, T=29760, cell_deps=(28800,), dep_grid=None,
             jt_best0=_NEGBIG, max_min=75, data=_DATA):
    """One-cell planned DepartAfterJourneyTree with a prefilled kernel selection anchored at B."""
    cell_deps = np.asarray(cell_deps, np.int64)
    dep_grid = cell_deps.copy() if dep_grid is None else np.asarray(dep_grid, np.int64)
    arrivalW = np.full((8, len(dep_grid)), _INF, np.int64)
    arrivalW[0, :] = T                                   # stop 0 reaches W at T from every minute
    tree = DepartAfterJourneyTree(
        data, [0, 1], [0], [aw], [-1], arrivalW, dep_grid, cell_deps, max_min,
        egress_g=[3], egress_w=[120], planned=True)
    Dstar = B - aw
    painted = (T - B + aw + 59) // 60                    # the kernel's anchored planned minute
    tree._sel = (np.array([0], np.int64), np.array([aw], np.int64),
                 np.array([Dstar], np.int64), np.array([T], np.int64),
                 np.array([False]), np.array([painted], np.int32))
    jt = _stub_jt(legs_raw, best0=jt_best0, max_min=max_min, data=data)
    tree._trees[int(T)] = jt
    return tree, jt


def _legs_sum(res):
    return sum(l["min"] + l.get("wait", 0) for l in res["legs"])


def _publish_planned_selection(monkeypatch, tree, painted, s, aw, D, T, is_walk=False):
    """Drive the real planned selection boundary from one synthetic provisional row."""
    from core import raptor as R
    monkeypatch.setattr(R, "select_planned_departafter", lambda *args, **kwargs: (
        np.array([painted], np.int32), np.array([s], np.int64), np.array([aw], np.int64),
        np.array([D], np.int64), np.array([T], np.int64), np.array([is_walk], bool)))
    tree._sel = None
    return tree._select_arrays()


def test_geometry_exposes_gtfs_boarding_actions_without_service_heuristics():
    """Ride geometry carries stop/direction facts needed by the plain-language inspector.

    These fields come from the pattern and stop table; the renderer never infers them from a
    concrete service name.  Sparse legacy data without stop_name remains covered by other fixtures.
    """
    jt = _stub_jt(None)
    legs = [{"mode": "transit", "line": "fixture-service", "min": 2,
             "segs": [("ride", 0, 0, 1)]}]
    geom = jt._geometry(0, legs, _Prov(), s_star=0)
    ride = geom[0]
    assert ride["board"] == {"name": "Origin Platform", "lat": 37.7, "lon": -122.4}
    assert ride["alight"] == {"name": "Transfer Platform", "lat": 37.71, "lon": -122.41}
    assert ride["toward"] == "North Terminal"
    assert ride["route_id"] == _DATA["line_table"][0][1]


# --------------------------------------------------------------------------- B1
def test_planned_first_wait_folds_into_access_walk_no_phantom_ride_minutes():
    """The exact-minute scheduled ride stays exact; the genuine 2-min pre-board wait lands in the
    access walk; the legs sum to the painted total (which the old home shift broke: it zeroed the
    visible wait and reconcile_legs pushed the residual into phantom walk/ride minutes)."""
    tree, _jt = _mk_tree(LEGS_CLEAN)
    res = tree.itinerary(0)
    painted = (29760 - 28800 + 180 + 59) // 60           # 19 min
    assert res["total"] == painted == 19
    assert _legs_sum(res) == res["total"]
    transit = [l for l in res["legs"] if l["mode"] == "transit"]
    assert len(transit) == 1
    assert transit[0]["min"] == 12                       # scheduled 12:00 ride displays 12, not 13
    assert transit[0].get("wait", 0) == 0                # the fold zeroed the visible first wait
    walks = [l for l in res["legs"] if l["mode"] == "walk"]
    assert walks[0]["min"] == 5                          # 3-min access walk + 2-min folded wait
    assert walks[0]["physical_min"] == 3.0               # actual street walk, before the fold
    assert walks[0]["schedule_allowance_min"] == 2.0     # controllable pre-board slack
    assert walks[-1]["min"] == 2                         # egress untouched
    assert "wait_clamped" not in res


def test_planned_committed_boarding_key_contract_holds():
    """Invariant (4): commit_home + walk0 == the committed first ride's scheduled departure, with
    commit_home the SAME D* anchor the map paints (so MC commutes measure from the painted anchor)."""
    tree, _jt = _mk_tree(LEGS_CLEAN)
    cl = tree.committed_first_legs()
    assert cl["commit_kind"][0] == 2
    assert cl["commit_pi"][0] == 0
    assert int(cl["commit_home"][0]) == 28800 - 180      # D* = B - access_walk, NOT the vehicle dep
    assert int(cl["commit_walk0"][0]) == 300             # access walk + the absorbed pre-board wait
    assert int(cl["commit_home"][0]) + int(cl["commit_walk0"][0]) == 28920   # == first ride dep


def test_planned_committed_via_stops_key_contract_holds():
    tree, _jt = _mk_tree(LEGS_CLEAN)
    out = tree.committed_legs_via_stops(0, [0])
    assert out["commit_kind"][0] == 2
    assert int(out["commit_home"][0]) + int(out["commit_walk0"][0]) == 28920
    assert int(out["commit_home"][0]) == 28620           # the validated anchor's own home


# --------------------------------------------------------------------------- B2
def test_fold_prefers_preceding_walk_leg():
    out = [{"mode": "walk", "line": None, "sec": 180, "segs": [("access",)]},
           {"mode": "transit", "line": "22", "sec": 720, "wait_sec": 120,
            "segs": [("ride", 0, 0, 3)]}]
    DepartAfterJourneyTree._fold_first_visible_wait(out)
    assert len(out) == 2
    assert out[0]["sec"] == 300                          # wait folded into the existing walk
    assert out[0]["physical_sec"] == 180
    assert out[0]["schedule_allowance_sec"] == 120
    assert out[0]["segs"] == [("access",)]               # geometry source unchanged
    assert out[1]["wait_sec"] == 0


def test_fold_inserted_walk_leg_carries_access_segs_in_geometry_mode():
    out = [{"mode": "transit", "line": "22", "sec": 720, "wait_sec": 120,
            "segs": [("ride", 0, 0, 3)]}]
    DepartAfterJourneyTree._fold_first_visible_wait(out)
    assert out[0]["mode"] == "walk" and out[0]["sec"] == 120
    assert out[0]["physical_sec"] == 0
    assert out[0]["schedule_allowance_sec"] == 120
    assert out[0]["segs"] == [("access",)]               # B2: not a pts-less minutes-bearing leg
    assert out[1]["wait_sec"] == 0


def test_fold_inserted_walk_leg_has_no_segs_outside_geometry_mode():
    out = [{"mode": "transit", "line": "22", "sec": 720, "wait_sec": 120}]
    DepartAfterJourneyTree._fold_first_visible_wait(out)
    assert out[0]["mode"] == "walk" and out[0]["sec"] == 120
    assert out[0]["physical_sec"] == 0
    assert out[0]["schedule_allowance_sec"] == 120
    assert "segs" not in out[0]                          # no internal plumbing leak into responses


def test_walk_truth_fields_preserve_fractional_physical_walk_and_allowance():
    """The public split is exact decimal minutes, not a second independent rounding scheme."""
    out = [{"mode": "walk", "line": None, "sec": 150},
           {"mode": "transit", "line": "fixture", "sec": 600, "wait_sec": 90}]
    DepartAfterJourneyTree._fold_first_visible_wait(out)
    res = {"legs": [{"mode": "walk", "line": None, "min": 4}]}
    geom = [{"mode": "walk", "name": None, "min": 4, "pts": []}]
    DepartAfterJourneyTree._attach_walk_truth(res, out, geom)
    assert out[0]["sec"] == 240
    assert res["legs"][0]["physical_min"] == 2.5
    assert res["legs"][0]["schedule_allowance_min"] == 1.5
    assert geom[0]["physical_min"] == 2.5
    assert geom[0]["schedule_allowance_min"] == 1.5


def test_walk_truth_fields_preserve_multi_minute_allowance():
    out = [{"mode": "walk", "line": None, "sec": 180},
           {"mode": "transit", "line": "fixture", "sec": 600, "wait_sec": 240}]
    DepartAfterJourneyTree._fold_first_visible_wait(out)
    res = {"legs": [{"mode": "walk", "line": None, "min": 7}]}
    DepartAfterJourneyTree._attach_walk_truth(res, out)
    assert res["legs"][0]["physical_min"] == 3.0
    assert res["legs"][0]["schedule_allowance_min"] == 4.0


def test_planned_candidate_rank_uses_exact_time_then_physical_access():
    """Minute ties cannot let a longer access variant win on an incidental formatter signature."""
    # Both candidates display 20 minutes and board at second 300. The first is physically closer;
    # the different homes/arrivals keep their exact door-to-door seconds equal for this generic
    # scheduled-fixture comparison.
    closer = {
        "total": 20, "home": 100, "board_anchor": 300, "route_key": (("A",),),
        "raw": [("access", 200), ("ride", 0, 300, 1300, 0, 1, 1), ("egress", 0, 1)],
    }
    farther = {
        "total": 20, "home": 0, "board_anchor": 300, "route_key": (("B",),),
        "raw": [("access", 300), ("ride", 1, 300, 1200, 0, 1, 1), ("egress", 0, 1)],
    }
    assert DepartAfterJourneyTree._planned_candidate_better(closer, farther)

    # Exact seconds remain stronger than the public rounded-minute tie.
    faster_exact = dict(farther, raw=[("access", 300),
                                      ("ride", 1, 300, 1190, 0, 1, 1),
                                      ("egress", 0, 1)])
    assert DepartAfterJourneyTree._planned_candidate_better(faster_exact, closer)


def test_fold_touches_only_the_first_transit_wait():
    out = [{"mode": "walk", "line": None, "sec": 60},
           {"mode": "transit", "line": "22", "sec": 300, "wait_sec": 90},
           {"mode": "transit", "line": "19", "sec": 600, "wait_sec": 120}]
    DepartAfterJourneyTree._fold_first_visible_wait(out)
    assert out[0]["sec"] == 150
    assert out[1]["wait_sec"] == 0
    assert out[2]["wait_sec"] == 120                     # transfer waits are real commute time


def test_zero_access_walk_geometry_stays_aligned_with_legs():
    """Full integration of B2: with a 0-second access walk the fold must insert the standalone
    walk leg, and _geometry must emit real points for it (legs/geom 1:1, no empty-pts minutes)."""
    tree, _jt = _mk_tree([("access", 0), RIDE_EXACT, ("egress", 120, 3)], aw=0)
    res = tree.itinerary(0, geom_provider=_Prov())
    painted = (29760 - 28800 + 0 + 59) // 60             # 16 min
    assert res["total"] == painted == 16
    assert _legs_sum(res) == res["total"]
    assert len(res["geom"]) == len(res["legs"])
    for leg, g in zip(res["legs"], res["geom"]):
        assert g["mode"] == leg["mode"]
        assert g["min"] == leg["min"]
        assert "segs" not in leg                         # plumbing stripped from the response
    assert res["legs"][0]["mode"] == "walk"              # the folded 2-min pre-board wait
    assert res["legs"][0]["min"] == 2
    assert res["legs"][0]["physical_min"] == 0.0
    assert res["legs"][0]["schedule_allowance_min"] == 2.0
    assert len(res["geom"][0]["pts"]) >= 2               # carries the access polyline, not []
    assert res["geom"][0]["physical_min"] == 0.0
    assert res["geom"][0]["schedule_allowance_min"] == 2.0
    assert len(res["geom"][1]["pts"]) == 4               # the ride's 4 pattern stops


# --------------------------------------------------------------------------- B3
def test_primary_deadline_quantization_does_not_become_schedule_allowance():
    """A late profile deadline does not add slack beyond the raw selected primary."""
    # The profile deadline is one minute after this tree's actual raw arrival.  Planned publication
    # uses the raw 19-minute trace; only the genuine 2-minute first-board wait is allowance.
    tree, _jt = _mk_tree(LEGS_CLEAN, T=29820)
    res = tree.itinerary(0, geom_provider=_Prov())

    assert res["total"] == _legs_sum(res) == 19
    access, _ride, egress = res["legs"]
    assert access["mode"] == "walk"
    assert access["physical_min"] == 3.0
    assert access["schedule_allowance_min"] == 2.0
    assert access["physical_min"] + access["schedule_allowance_min"] == access["min"]
    assert egress["mode"] == "walk" and egress["min"] == 2
    assert res["geom"][0]["physical_min"] == 3.0
    assert res["geom"][0]["schedule_allowance_min"] == 2.0


def test_planned_via_stop_uses_raw_seconds_not_profile_deadline_slack():
    """The explicit-stop formatter never turns deadline quantization into allowance."""
    tree, _jt = _mk_tree(LEGS_CLEAN, T=29820)
    res = tree.itinerary_via_stop(0, 0, geom_provider=_Prov(), percentile="planned")

    assert res["total"] == _legs_sum(res) == 19
    access, _ride, egress = res["legs"]
    assert access["physical_min"] == 3.0
    assert access["schedule_allowance_min"] == 2.0
    assert access["physical_min"] + access["schedule_allowance_min"] == access["min"]
    assert egress["mode"] == "walk" and egress["min"] == 2
    assert res["geom"][0]["schedule_allowance_min"] == 2.0


def test_planned_branch_ignores_deadline_slack_when_raw_chain_has_no_access():
    """A caller-supplied deadline target cannot invent an allowance-only access leg."""
    raw = [RIDE_EXACT, ("egress", 120, 3)]
    tree, _jt = _mk_tree(raw, aw=0, B=28920, T=29760)
    res = tree._planned_itinerary_from_anchor(
        0, 0, 0, 28920, 29760, geom_provider=_Prov(), planned_total=15,
        planned_target_sec=900)

    assert res["total"] == _legs_sum(res) == 14
    _ride, egress = res["legs"]
    assert not any("schedule_allowance_min" in leg for leg in res["legs"])
    assert egress["mode"] == "walk" and egress["min"] == 2
    assert len(res["geom"]) == 2


def test_positive_target_residual_keeps_exact_source_walk_split():
    """The pre-format source leg keeps its seconds identity before rounded display fields exist."""
    out = [{"mode": "transit", "line": "fixture", "sec": 720, "wait_sec": 0,
            "segs": [("ride", 0, 0, 3)]}]
    DepartAfterJourneyTree._reconcile_planned_target(out, total_sec=720, target_sec=780)

    access = out[0]
    assert access["segs"] == [("access",)]
    assert access["physical_sec"] + access["schedule_allowance_sec"] == access["sec"] == 60


def test_format_planned_raw_rejects_caller_deadline_slack():
    """Raw branch publication is authoritative even when a caller supplies a longer target."""
    raw = [RIDE_EXACT, ("egress", 120, 3)]
    tree, jt = _mk_tree(raw, aw=0, B=28920, T=29760)
    res = tree._format_planned_raw(
        0, 0, raw, 28920, jt, geom_provider=_Prov(), planned_total=16,
        planned_target_sec=901)

    assert res["total"] == _legs_sum(res) == 14
    _ride, egress = res["legs"]
    assert not any("schedule_allowance_min" in leg for leg in res["legs"])
    assert egress["mode"] == "walk" and egress["min"] == 2


def test_format_planned_raw_uses_1140_second_trace_not_1200_second_deadline():
    """A source branch uses raw elapsed seconds even when its profile deadline is later."""
    tree, jt = _mk_tree(LEGS_CLEAN, T=29820)
    res = tree._format_planned_raw(
        0, 0, LEGS_CLEAN, 28620, jt, planned_total=20, planned_target_sec=1200)

    assert res["total"] == _legs_sum(res) == 19
    access, _ride, egress = res["legs"]
    assert access["physical_min"] == 3.0
    assert access["schedule_allowance_min"] == 2.0
    assert egress["mode"] == "walk" and egress["min"] == 2


def test_format_planned_raw_does_not_turn_ordinary_ceil_rounding_into_allowance():
    """An exact 14m01s target displays 15m without inventing 59s of schedule slack."""
    raw = [RIDE_EXACT, ("egress", 121, 3)]
    tree, jt = _mk_tree(raw, aw=0, B=28920, T=29761)
    res = tree._format_planned_raw(0, 0, raw, 28920, jt, planned_total=15)

    assert res["total"] == _legs_sum(res) == 15
    assert all("schedule_allowance_min" not in leg for leg in res["legs"])


def test_short_profile_target_cannot_shave_or_drop_a_raw_route():
    """Raw route truth wins over a shorter profile target without changing physical walking."""
    raw = [("access", 180), ("ride", 0, 28800, 29520, 0, 3, 3), ("egress", 120, 3)]
    tree, _jt = _mk_tree(raw, B=28800, T=29580)

    via = tree.itinerary_via_stop(0, 0, percentile="planned")
    branch = tree._planned_itinerary_from_anchor(
        0, 0, 180, 28800, 29580, planned_total=16)
    assert via["total"] == branch["total"] == 17
    assert via["legs"][0]["min"] == branch["legs"][0]["min"] == 3
    assert all("schedule_allowance_min" not in leg for leg in via["legs"] + branch["legs"])


def test_negative_target_residual_never_changes_physical_or_egress_seconds():
    """Only explicit allowance is eligible for a too-short target."""
    out = [
        {"mode": "walk", "line": None, "sec": 200,
         "physical_sec": 180, "schedule_allowance_sec": 20},
        {"mode": "transit", "line": "fixture", "sec": 720, "wait_sec": 0},
        {"mode": "walk", "line": None, "sec": 120},
    ]
    leftover = DepartAfterJourneyTree._reconcile_planned_target(
        out, total_sec=1040, target_sec=900)

    assert leftover == 120
    assert out[0]["physical_sec"] == 180
    assert out[0]["schedule_allowance_sec"] == 0
    assert out[0]["sec"] == 180
    assert out[-1]["sec"] == 120


def test_raw_chain_guard_detects_pre_anchor_board():
    assert not DepartAfterJourneyTree._raw_chain_valid_after_start(LEGS_BAD, 28800)
    assert DepartAfterJourneyTree._raw_chain_valid_after_start(LEGS_BAD, 28700)
    assert DepartAfterJourneyTree._raw_chain_valid_after_start(LEGS_BAD, 28650)


def test_primary_reselects_nearest_truthful_window_before_map_publish(monkeypatch):
    tree, _jt = _mk_tree(LEGS_BAD, cell_deps=(28620, 28680, 28800),
                         dep_grid=(28620, 28680, 28800), jt_best0=28680)
    s, aw, D, T, is_walk, painted = _publish_planned_selection(
        monkeypatch, tree, painted=19, s=0, aw=180, D=28620, T=29760)
    assert (int(s[0]), int(aw[0]), int(D[0]), int(T[0]), bool(is_walk[0]), int(painted[0])) == (
        0, 180, 28680 - 180, 29760, False, 21)
    legs_raw, home = tree._trace_raw(0)
    assert legs_raw == LEGS_BAD
    assert home == 28680 - 180
    assert 0 not in tree._wait_clamped
    res = tree.itinerary(0)
    assert tree.commute()[0] == res["total"] == 21
    assert _legs_sum(res) == res["total"]
    assert "wait_clamped" not in res


def test_primary_reselects_18_min_stale_anchor_to_truthful_22_min_primary(monkeypatch):
    """The former 18→22 formatter mismatch is corrected atomically in map selection."""
    legs = [("access", 180), ("ride", 0, 28565, 29640, 0, 3, 3), ("egress", 60, 3)]
    tree, _jt = _mk_tree(legs, cell_deps=(28560, 28800), dep_grid=(28560, 28800),
                         T=29700, jt_best0=28560)
    _s, _aw, D, _T, _walk, painted = _publish_planned_selection(
        monkeypatch, tree, painted=18, s=0, aw=180, D=28620, T=29700)
    assert int(D[0]) == 28560 - 180
    assert int(painted[0]) == 22
    _legs_raw, home = tree._trace_raw(0)
    assert home == 28560 - 180
    assert 0 not in tree._wait_clamped
    res = tree.itinerary(0)
    assert tree.commute()[0] == res["total"] == _legs_sum(res) == 22
    assert "wait_clamped" not in res
    cl = tree.committed_first_legs()                     # contract holds on the re-anchored home
    assert int(cl["commit_home"][0]) == 28380
    assert int(cl["commit_home"][0]) + int(cl["commit_walk0"][0]) == 28565


def test_primary_reselection_keeps_a_truthful_branch_between_stale_and_fallback_totals(monkeypatch):
    """A raw-19-minute valid branch wins over stale 18-minute / fallback-22-minute anchors."""
    invalid = [("access", 180), ("ride", 0, 28565, 29640, 0, 3, 3), ("egress", 60, 3)]
    valid = [("access", 180), RIDE_EXACT, ("egress", 120, 3)]
    tree, jt = _mk_tree(invalid, cell_deps=(28560, 28800), dep_grid=(28560, 28800), T=29700)
    tree.access_off = np.array([0, 2], np.int64)
    tree.access_to = np.array([0, 1], np.int64)
    tree.access_w = np.array([180, 180], np.int64)
    tree.n_cells = 1
    tree.arrivalW[1, :] = 29820
    jt._trace_from = lambda s, aw, home: (list(valid if int(s) == 1 else invalid), int(home))
    tree._trees[29820] = jt

    s, _aw, D, T, _walk, painted = _publish_planned_selection(
        monkeypatch, tree, painted=18, s=0, aw=180, D=28620, T=29700)
    assert (int(s[0]), int(D[0]), int(T[0]), int(painted[0])) == (1, 28620, 29820, 19)
    res = tree.itinerary(0)
    assert tree.commute()[0] == res["total"] == _legs_sum(res) == 19
    assert res["legs"][0]["schedule_allowance_min"] == 2.0


def test_primary_promotes_faster_second_level_reanchor_over_valid_slower_choice(monkeypatch):
    """Primary and via-stop alternatives share the same tree-best re-anchor universe."""
    route22 = lambda aw: [
        ("access", aw), ("ride", 0, 28565, 29640, 0, 3, 3), ("egress", 60, 3)]
    route19 = lambda aw: [
        ("access", aw), ("ride", 1, 28655, 29640, 0, 1, 5), ("egress", 60, 5)]
    tree, jt = _mk_tree(
        route22(180), cell_deps=(28560, 28800), dep_grid=(28560, 28800),
        T=29700, jt_best0=28560)
    tree.access_off = np.array([0, 2], np.int64)
    tree.access_to = np.array([0, 4], np.int64)
    tree.access_w = np.array([180, 180], np.int64)
    tree.arrivalW[4, :] = 29700
    jt.best[4] = 28650
    jt._trace_from = lambda s, aw, home: (
        route22(int(aw)) if int(s) == 0 else route19(int(aw)), int(home))

    s, _aw, D, _T, _walk, painted = _publish_planned_selection(
        monkeypatch, tree, painted=18, s=0, aw=180, D=28620, T=29700)
    assert (int(s[0]), int(D[0]), int(painted[0])) == (4, 28650 - 180, 21)
    primary = tree.itinerary(0)
    alt22 = tree.itinerary_via_stop(0, 0, percentile="planned")
    alt19 = tree.itinerary_via_stop(0, 4, percentile="planned")
    assert tree.commute()[0] == primary["total"] == alt19["total"] == 21
    assert primary["total"] <= alt22["total"] == 22
    assert {leg["line"] for leg in primary["legs"] if leg["mode"] == "transit"} == {"19"}


def test_primary_raw_seconds_beat_later_deadline_and_59_min_branch(monkeypatch):
    """Live-pattern shape: profile 3581/raw 3471 paints 58 and beats a raw-3531 branch."""
    primary = [("access", 0), ("ride", 0, 0, 3471, 0, 3, 3)]
    branch = [("access", 0), ("ride", 0, 0, 3531, 0, 3, 3)]
    tree, jt = _mk_tree(primary, aw=0, B=0, T=3581, cell_deps=(0,), dep_grid=(0,))
    _publish_planned_selection(monkeypatch, tree, painted=60, s=0, aw=0, D=0, T=3581)

    res = tree.itinerary(0)
    branch_res = tree._format_planned_raw(
        0, 0, branch, 0, jt, planned_total=59, planned_target_sec=3531)
    assert tree.commute()[0] == res["total"] == _legs_sum(res) == 58
    assert branch_res["total"] == _legs_sum(branch_res) == 59
    assert res["total"] < branch_res["total"]


def test_primary_reselection_considers_purewalk_when_transit_primary_is_stale(monkeypatch):
    """A 20-min direct walk beats a truthful 22-min transit fallback."""
    legs = [("access", 180), ("ride", 0, 28565, 29640, 0, 3, 3), ("egress", 60, 3)]
    tree, _jt = _mk_tree(legs, cell_deps=(28560, 28800), dep_grid=(28560, 28800),
                         T=29700, jt_best0=28560)
    tree.purewalk = np.array([1200], np.int64)
    s, _aw, D, _T, is_walk, painted = _publish_planned_selection(
        monkeypatch, tree, painted=18, s=0, aw=180, D=28620, T=29700)
    assert (int(s[0]), int(D[0]), bool(is_walk[0]), int(painted[0])) == (-1, 28800 - 1200, True, 20)
    res = tree.itinerary(0)
    assert tree.commute()[0] == res["total"] == _legs_sum(res) == 20


def test_primary_reselection_preserves_physical_access_walk(monkeypatch):
    """The upstream correction never rewrites 300 seconds of genuine access walking."""
    legs = [("access", 300), ("ride", 0, 28565, 29640, 0, 3, 3), ("egress", 60, 3)]
    tree, _jt = _mk_tree(legs, aw=300, cell_deps=(28560, 28800),
                         dep_grid=(28560, 28800), T=29700, jt_best0=28560)
    _publish_planned_selection(monkeypatch, tree, painted=20, s=0, aw=300, D=28500, T=29700)
    res = tree.itinerary(0)
    assert res["total"] == _legs_sum(res) == 24
    access = res["legs"][0]
    assert access["mode"] == "walk"
    # The corrected 24-min target keeps both the 300 s physical access and its 5 s pre-board
    # schedule allowance; neither component is rewritten by formatting.
    assert access["physical_min"] == 5.0
    assert access["schedule_allowance_min"] == 5 / 60.0
    assert access["physical_min"] + access["schedule_allowance_min"] == 305 / 60.0


def test_dropped_subminute_access_truth_does_not_attach_to_egress():
    """A zero-rounded annotated access walk must disappear with its metadata, not relabel egress."""
    # 5 s physical + 1 s allowance = 0.1 min; the 54 s egress has the larger remainder and gets
    # the sole rounded minute. Thus only the unannotated egress walk survives formatting.
    legs = [("access", 5), ("ride", 0, 28801, 29521, 0, 3, 3), ("egress", 54, 3)]
    tree, _jt = _mk_tree(legs, aw=5, T=29575)
    res = tree.itinerary(0, geom_provider=_Prov())
    walks = [leg for leg in res["legs"] if leg["mode"] == "walk"]
    assert len(walks) == 1 and walks[0]["min"] == 1
    assert "physical_min" not in walks[0]
    assert "schedule_allowance_min" not in walks[0]
    geom_walks = [leg for leg in res["geom"] if leg["mode"] == "walk"]
    assert len(geom_walks) == 1
    assert "physical_min" not in geom_walks[0]
    assert "schedule_allowance_min" not in geom_walks[0]


def test_primary_without_truthful_anchor_is_consistently_unreachable(monkeypatch):
    tree, _jt = _mk_tree(LEGS_BAD)                       # window = the single invalid minute
    _s, _aw, _D, _T, _walk, painted = _publish_planned_selection(
        monkeypatch, tree, painted=19, s=0, aw=180, D=28620, T=29760)
    assert int(painted[0]) == -1
    assert tree.commute()[0] == -1
    assert tree._trace_raw(0) is None
    assert tree.itinerary(0) is None


def test_validated_stop_anchor_reanchors_to_tree_best_inside_window():
    tree, jt = _mk_tree(LEGS_BAD, cell_deps=(28620, 28800), dep_grid=(28620, 28800),
                        jt_best0=28650)
    va = tree._validated_stop_anchor(0, 0, 180)
    assert va is not None
    pm, D, T, legs_raw, jt2 = va
    assert jt2 is jt
    assert D == 28650 - 180                              # re-anchored on the tree's own best[s]
    assert pm == (29760 - 28650 + 180 + 59) // 60 == 22  # minute recomputed (can only grow)
    it = tree.itinerary_via_stop(0, 0, percentile="planned")
    assert it["total"] == 22                             # strip total == the validated chip minute
    assert _legs_sum(it) == it["total"]


def test_validated_stop_anchor_drops_when_reanchor_leaves_window():
    tree, _jt = _mk_tree(LEGS_BAD, jt_best0=28650)       # window starts at 28800 > best[s]
    assert tree._validated_stop_anchor(0, 0, 180) is None
    assert tree.itinerary_via_stop(0, 0, percentile="planned") is None


def test_planned_alt_window_drops_unboardable_stop_and_keeps_reanchored_one():
    dropped, _ = _mk_tree(LEGS_BAD, jt_best0=28650)
    assert dropped.alt_lines_window(np.array([19]), 8)[0] is None
    kept, _ = _mk_tree(LEGS_BAD, cell_deps=(28620, 28800), dep_grid=(28620, 28800),
                       jt_best0=28650)
    win = kept.alt_lines_window(np.array([19]), 8)
    assert win[0] is not None
    assert list(win[0]["22"]) == [22, 0]                 # the VALIDATED minute, not the raw 19


def test_planned_alt_window_equal_label_tie_uses_minute_then_structural_route_key():
    """Vectorization must not inject stop id into the old public-label collision tie-break."""
    tree = DepartAfterJourneyTree.__new__(DepartAfterJourneyTree)
    tree.n_cells = 1
    tree.max_min = 75
    tree.access_off = np.array([0, 2], np.int64)
    tree.access_to = np.array([0, 1], np.int64)
    tree.access_w = np.array([60, 0], np.int64)
    tree.arrivalW = np.zeros((2, 1), np.int64)
    tree._select_arrays = lambda: (
        np.array([0]), np.array([60]), np.array([0]), np.array([0]),
        np.array([False]), np.array([10]))
    tree._planned_stop_anchors = lambda: (
        np.array([540, 600], np.int64), None, None)
    templates = {
        # Both display as ALPHA at 10m. The lexicographically earlier structural route is stop 1,
        # even though stop 0 has the smaller numeric id.
        0: (0, 540, (("walk", 540),), "ALPHA", ("z-route",)),
        1: (0, 600, (("walk", 600),), "ALPHA", ("a-route",)),
    }
    tree._validated_stop_template = lambda stop: templates[int(stop)]

    assert tree._planned_alt_lines_window(np.array([10]), 3)[0] == {"ALPHA": [10, 1]}


def test_planned_alt_window_uses_perfect_base_and_painted_sentinel_fallback():
    """A valid caller baseline widens its cell's window; -1 keeps painted fallback.

    The production planned endpoint intentionally passes its painted scheduled value as
    ``perfect``, so this seam is normally numerically invisible.  Preserve the general contract:
    a different non-negative perfect value must be used for that row, while a negative sentinel
    must not make an otherwise painted-reachable row unbounded or unreachable.
    """
    tree = DepartAfterJourneyTree.__new__(DepartAfterJourneyTree)
    tree.n_cells = 2
    tree.max_min = 75
    tree.access_off = np.array([0, 1, 2], np.int64)
    tree.access_to = np.array([0, 1], np.int64)
    tree.access_w = np.array([0, 0], np.int64)
    tree.arrivalW = np.zeros((2, 1), np.int64)
    tree._select_arrays = lambda: (
        np.array([0, 1]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0]),
        np.array([False, False]), np.array([10, 10]))
    # Each candidate takes 22 minutes.  The first cell receives a valid perfect=20 baseline,
    # putting it in the +3-minute window; the second receives -1 and must retain painted=10.
    tree._planned_stop_anchors = lambda: (
        np.array([22 * 60, 22 * 60], np.int64), None, None)
    tree._validated_stop_template = lambda stop: (
        0, 22 * 60, (("walk", 22 * 60),), "ALPHA", ("route", int(stop)))

    out = tree._planned_alt_lines_window(np.array([20, -1]), 3)
    assert out[0] == {"ALPHA": [22, 0]}
    assert out[1] is None


def test_grouped_planned_templates_share_deadline_tree_but_keep_b3_per_stop():
    """Grouping is only a tree lookup optimization; B3 stays stop-specific.

    Stops 0 and 1 share T=900. Stop 1's requested B=200 cannot board its 150-second ride, so it
    must still run the existing tree-local re-anchor to B=140 instead of inheriting stop 0's
    valid B=100 or being silently dropped. Stop 2 proves a second deadline gets one separate tree.
    This is deliberately an all-Python fake tree: no RAPTOR/Numba execution is needed to protect
    the extraction contract.
    """
    tree = DepartAfterJourneyTree.__new__(DepartAfterJourneyTree)
    tree._planned_validated_stop = {}
    tree.cell_deps = np.array([100], np.int64)
    tree._planned_route_label = lambda raw: "fixture"
    tree._planned_route_identity = lambda raw: tuple(raw)
    tree_calls = []
    trace_calls = []

    class _DeadlineTree:
        def __init__(self, deadline):
            self.deadline = int(deadline)
            self.best = np.array([100, 140, 190, -1], np.int64)

        def _trace_from(self, stop, access_walk, board):
            stop = int(stop); board = int(board)
            trace_calls.append((self.deadline, stop, board))
            dep = 220 if self.deadline == 1200 else 150
            return ([('access', int(access_walk)), ('ride', stop, dep, 300, 0, 1, 9),
                     ('egress', 0, 9)], board)

    def tree_at(deadline):
        tree_calls.append(int(deadline))
        return _DeadlineTree(deadline)

    tree._tree_at = tree_at
    templates = tree._validated_stop_templates_grouped(
        np.array([0, 1, 2, 3], np.int64),
        np.array([100, 200, 190, -1], np.int64),
        np.array([900, 900, 1200, -1], np.int64))

    assert tree_calls == [900, 1200]                     # once per representative deadline
    assert trace_calls == [(900, 0, 100), (900, 1, 200), (1200, 2, 190)]
    assert templates[0][0] == 100
    assert templates[1][0] == 140                        # B3 fallback from the shared T=900 tree
    assert templates[2][0] == 190
    assert templates[3] is None


def test_planned_stop_anchor_vectorized_preserves_latest_board_and_sentinels():
    """The vectorized anchor pass keeps the scalar latest-B tie and unusable-stop contract."""
    # Stop 0 has equal 420 s tails at B=10 and B=20: the later B=20 must win.  Stop 1
    # is unreachable.  Stop 2's tempting negative-B probe wins the scalar objective, then gets
    # discarded by its historical final ``bB >= 0`` guard rather than falling through to B=0.
    arrival = np.array([
        [600, 430, 440, 500],
        [_INF, _INF, _INF, _INF],
        [80, 130, 900, _INF],
    ], np.int64)
    kk = np.array([0, 1, 2, 3, 99], np.int64)  # out-of-profile probe is ignored
    board = np.array([-10, 10, 20, 30, 40], np.int64)

    expected = DepartAfterJourneyTree._planned_stop_anchors_oracle(arrival, kk, board)
    actual = DepartAfterJourneyTree._planned_stop_anchors_vectorized(arrival, kk, board)

    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(actual[0], np.array([420, _INF, _INF], np.int64))
    np.testing.assert_array_equal(actual[1], np.array([20, -1, -1], np.int64))
    np.testing.assert_array_equal(actual[2], np.array([440, -1, -1], np.int64))


def test_planned_stop_anchor_vectorized_matches_oracle_on_random_profiles():
    """Differential coverage for irregular grids, duplicate boards, ties, and INF holes."""
    rng = np.random.default_rng(20260807)
    for _ in range(100):
        n_stops = int(rng.integers(0, 13))
        n_grid = int(rng.integers(0, 17))
        n_probe = int(rng.integers(0, 17))
        arrival = rng.integers(0, 80_000, size=(n_stops, n_grid), dtype=np.int64)
        if n_grid:
            arrival[rng.random(size=arrival.shape) < 0.30] = _INF
        # `searchsorted` emits non-negative positions, but duplicate cell departures are allowed
        # and are where the scalar's stable probe ordering matters.
        kk = rng.integers(0, n_grid + 4, size=n_probe, dtype=np.int64)
        board = rng.integers(-120, 80_000, size=n_probe, dtype=np.int64)
        # Force an exact equal-cost collision with different boarding minutes.
        if n_stops and n_grid >= 2 and n_probe >= 2:
            kk[:2] = 0
            board[0] = 100
            board[1] = 200
            arrival[0, 0] = 1_000
            kk[1] = 1
            arrival[0, 1] = 1_100

        expected = DepartAfterJourneyTree._planned_stop_anchors_oracle(arrival, kk, board)
        actual = DepartAfterJourneyTree._planned_stop_anchors_vectorized(arrival, kk, board)
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got, want)


def test_deadline_tree_waiter_survives_immediate_lru_eviction(monkeypatch):
    """Concurrent variance/pin readers receive the owner's tree even after cache churn."""
    from core import raptor as R
    from core import raptor_journey_da as rda

    tree = DepartAfterJourneyTree.__new__(DepartAfterJourneyTree)
    tree._trees = OrderedDict()
    tree._tree_cache_max = 0
    tree._trees_lock = threading.Lock()
    tree._tree_flights = {}
    tree.d = {}
    tree.egress_g = np.array([0], np.int64)
    tree.egress_w = np.array([0], np.int64)
    tree.max_rounds = 1
    tree.board_slack = 60
    tree.access_off = np.array([0, 0], np.int64)
    tree.access_to = np.array([], np.int64)
    tree.access_w = np.array([], np.int64)
    tree.purewalk = np.array([-1], np.int64)
    tree.max_min = 75
    tree.beta = 1.0
    tree.eps = 60.0

    entered = threading.Event()
    release = threading.Event()
    built = object()

    def traced(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return object()

    monkeypatch.setattr(R, "reverse_raptor_traced_fast", traced)
    monkeypatch.setattr(rda, "JourneyTree", lambda *args, **kwargs: built)

    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(tree._tree_at, 100)
        assert entered.wait(timeout=2)
        flight = tree._tree_flights[100]
        waiter_entered = threading.Event()
        original_wait = flight["event"].wait

        def observed_wait(*args, **kwargs):
            waiter_entered.set()
            return original_wait(*args, **kwargs)

        flight["event"].wait = observed_wait
        waiter = pool.submit(tree._tree_at, 100)
        assert waiter_entered.wait(timeout=2)
        release.set()
        assert owner.result(timeout=2) is built
        assert waiter.result(timeout=2) is built
    assert tree._trees == {}
    assert tree._tree_flights == {}


# ----------------------------------------------------------- branch sig (A7 follow-on)
# Schedule arrays for the branch-enumeration helpers: pattern 0 ("22") has 1 trip over its 4
# stops, pattern 1 ("19") 1 trip over its 2 stops; arr == dep (zero dwell) keeps the math flat.
_DATA2 = dict(
    _DATA,
    pat_nstops=np.array([4, 2], np.int64),
    pat_ntrips=np.array([1, 1], np.int64),
    pat_mat_off=np.array([0, 4], np.int64),
    pat_dep=np.array([28920, 29040, 29160, 29280, 29400, 29940], np.int64),
    pat_arr=np.array([28920, 29040, 29160, 29280, 29400, 29940], np.int64),
)


def test_walk_finish_branch_sig_comes_from_legs_without_geometry():
    """With geom_provider=None (the server's enumeration call shape after A7), branch candidates
    have no ``geom`` — the rank ``sig`` must fall back to the LEGS like the main enumeration does,
    not silently collapse to ``()`` (which sorts before every legs-based sig in
    ``_planned_candidate_rank`` and would flip tie-breaks)."""
    tree, jt = _mk_tree(None, data=_DATA2)
    eg = np.full(6, EGRESS_INF, np.int64)
    eg[2] = 300; eg[3] = 600; eg[5] = 60                 # walkable alights mid-"22" + at the "19" end
    jt.egress_sec = eg
    cand = {
        "stop": 0, "home": 28620, "jt": jt,
        "raw": [("access", 180), ("ride", 0, 28920, 29040, 0, 1, 1),
                ("walk_t", 60, 1, 4), ("ride", 1, 29400, 29940, 0, 1, 5),
                ("egress", 60, 5)],
        "it": {"total": 23, "legs": [
            {"mode": "walk", "line": None, "min": 3},
            {"mode": "transit", "line": "22", "min": 2, "wait": 0},
            {"mode": "walk", "line": None, "min": 1},
            {"mode": "transit", "line": "19", "min": 9, "wait": 0},
            {"mode": "walk", "line": None, "min": 1}]},
    }
    out = tree._planned_walk_finish_branches(0, [cand], 30, geom_provider=None)
    assert len(out) == 2                                 # the two egress-reachable "22" alights
    for b in out:
        assert b["line"] == "22"
        assert b["sig"] != ()                            # the regression: sig collapsed to ()
        assert b["sig"] == DepartAfterJourneyTree._geom_route_sig(b["it"]["legs"])


def test_walk_finish_lazy_materialization_matches_eager_oracle_and_skips_losing_probes():
    """Repeated scheduled probes of one structural walk finish keep the eager winner.

    Two first-trip instances produce the same structural alights.  The later instance has more
    in-vehicle time at the same displayed total/final walk, so it wins the exact rank before
    formatting.  The local eager loop is deliberately scalar: it materializes all four proposals;
    the implementation must retain its public candidates while formatting only the two winners.
    """
    data = dict(
        _DATA2,
        pat_ntrips=np.array([2, 1], np.int64),
        pat_mat_off=np.array([0, 8], np.int64),
        pat_dep=np.array([
            28920, 29040, 29160, 29280,       # first 22 trip: 120 s between stops
            30000, 30150, 30300, 30450,       # second 22 trip: 150 s between stops
            29400, 29940,
        ], np.int64),
        pat_arr=np.array([
            28920, 29040, 29160, 29280,
            30000, 30150, 30300, 30450,
            29400, 29940,
        ], np.int64),
    )
    tree, jt = _mk_tree(None, data=data)
    tree.max_min = 75
    eg = np.full(6, EGRESS_INF, np.int64)
    # The two trips differ by 30 s at later alights.  These deliberately non-minute egresses
    # keep their CEIL totals equal, while preserving an exact raw-transit rank difference.
    eg[2] = 301; eg[3] = 541
    jt.egress_sec = eg

    def seed(dep, arr1, home):
        return {
            "stop": 0, "home": home, "jt": jt,
            "raw": [("access", 180), ("ride", 0, dep, arr1, 0, 1, 1),
                    ("walk_t", 60, 1, 4), ("ride", 1, 29400, 29940, 0, 1, 5),
                    ("egress", 60, 5)],
            "it": {"total": 23, "legs": [
                {"mode": "transit", "line": "22", "min": 2, "wait": 0},
                {"mode": "transit", "line": "19", "min": 9, "wait": 0},
            ]},
        }

    seeds = [seed(28920, 29040, 28620), seed(30000, 30150, 29760)]
    calls = []

    def fake_format(ci, stop, raw, home, source, geom_provider=None, planned_total=None,
                    planned_target_sec=None):
        calls.append((int(raw[1][2]), int(raw[1][5])))
        return {"total": int(planned_total), "legs": [
            {"mode": "transit", "line": "22", "min": 5, "wait": 0},
            {"mode": "walk", "line": None, "min": 5},
        ]}

    tree._format_planned_raw = fake_format

    def eager_oracle():
        proposals = []
        for cand in seeds:
            first = cand["raw"][1]
            first_pi, first_dep, first_bpos = int(first[1]), int(first[2]), int(first[4])
            ns0 = int(data["pat_nstops"][first_pi])
            mb0 = int(data["pat_mat_off"][first_pi]); sb0 = int(data["pat_stop_off"][first_pi])
            deps = data["pat_dep"][mb0 + first_bpos: mb0 + first_bpos
                                     + int(data["pat_ntrips"][first_pi]) * ns0: ns0]
            trip = int(np.searchsorted(deps, first_dep, side="left"))
            for xapos in range(first_bpos + 1, ns0):
                stop = int(data["pat_stops"][sb0 + xapos]); egress = int(jt.egress_sec[stop])
                if egress >= EGRESS_INF:
                    continue
                arr = int(data["pat_arr"][mb0 + trip * ns0 + xapos])
                raw = [("access", 180),
                       ("ride", first_pi, first_dep, arr, first_bpos, xapos, stop),
                       ("egress", egress, stop)]
                total = int(np.ceil(tree._planned_raw_total_sec(raw, cand["home"]) / 60.0))
                it = fake_format(0, cand["stop"], raw, cand["home"], jt,
                                 planned_total=total)
                geom = it["legs"]
                proposals.append({
                    "line": tree._geom_route_label(geom), "total": total, "it": it,
                    "stop": cand["stop"], "sig": tree._geom_route_sig(geom), "raw": raw,
                    "route_key": tree._planned_route_identity(raw), "home": cand["home"], "jt": jt,
                })
        best = {}
        for proposal in proposals:
            key = proposal["route_key"]
            if tree._planned_candidate_better(proposal, best.get(key)):
                best[key] = proposal
        return list(best.values())

    expected = eager_oracle()
    assert len(calls) == 4                               # old path formats every scheduled probe
    calls.clear()
    actual = tree._planned_walk_finish_branches(0, seeds, 30, geom_provider=None)

    def observable(rows):
        return [{key: row[key] for key in ("line", "total", "raw", "route_key", "home", "sig")}
                for row in rows]

    assert observable(actual) == observable(expected)
    # Exact elapsed seconds now settle the equal-minute candidates before formatting. The later
    # trip wins the nearer finish; the earlier trip wins the farther one.
    assert calls == [(30000, 2), (28920, 3)]


def test_walk_finish_lazy_materialization_uses_board_anchor_before_display_signature():
    """A later viable board breaks an old display-only tie without formatting the loser."""
    data = dict(
        _DATA2,
        pat_ntrips=np.array([2, 1], np.int64),
        pat_mat_off=np.array([0, 8], np.int64),
        pat_dep=np.array([
            28920, 29040, 29160, 29280,
            30000, 30120, 30240, 30360,       # same ride seconds as the first trip
            29400, 29940,
        ], np.int64),
        pat_arr=np.array([
            28920, 29040, 29160, 29280,
            30000, 30120, 30240, 30360,
            29400, 29940,
        ], np.int64),
    )
    tree, jt = _mk_tree(None, data=data)
    tree.max_min = 75
    eg = np.full(6, EGRESS_INF, np.int64)
    eg[2] = 301; eg[3] = 541
    jt.egress_sec = eg

    def seed(dep, arr1, home):
        return {
            "stop": 0, "home": home, "jt": jt,
            "raw": [("access", 180), ("ride", 0, dep, arr1, 0, 1, 1),
                    ("walk_t", 60, 1, 4), ("ride", 1, 29400, 29940, 0, 1, 5),
                    ("egress", 60, 5)],
            "it": {"total": 23, "legs": [
                {"mode": "transit", "line": "22", "min": 2, "wait": 0},
                {"mode": "transit", "line": "19", "min": 9, "wait": 0},
            ]},
        }

    seeds = [seed(28920, 29040, 28620), seed(30000, 30120, 29700)]
    calls = []

    def fake_format(ci, stop, raw, home, source, geom_provider=None, planned_total=None,
                    planned_target_sec=None):
        departure = int(raw[1][2])
        calls.append((departure, int(raw[1][5])))
        # The later probe intentionally has a lexically smaller final display signature.  This
        # simulates a rounding/wait distinction that the raw rank cannot inspect.
        shown_min = 2 if departure == 28920 else 1
        return {"total": int(planned_total), "legs": [
            {"mode": "transit", "line": "22", "min": shown_min, "wait": 0},
            {"mode": "walk", "line": None, "min": 5},
        ]}

    tree._format_planned_raw = fake_format
    actual = tree._planned_walk_finish_branches(0, seeds, 30, geom_provider=None)
    # The two later-board candidates have equal displayed minutes and exact ride duration, but
    # the canonical quality tuple picks their later board before display-signature work.
    assert calls == [(30000, 2), (30000, 3)]
    assert {int(row["raw"][1][2]) for row in actual} == {30000}
    assert all(row["sig"] != () for row in actual)


def test_branch_closure_discovers_same_service_reboard_from_walk_sibling_once():
    """A two-ride seed can reveal a one-seat sibling, then a same-service one-tail reboard.

    All service names are arbitrary fixture data.  The regression is the structural sequence:
    two rides -> same first ride plus walk -> reboard that service on another pattern. The display
    label is deliberately repeated: public text must neither reject the tail nor identify it. The
    bounded closure also deduplicates duplicate seeds and does not cycle through the shapes.
    """
    data = {
        "line_table": [
            ("fixture", "orbit", "ORBIT", "tram"),
            ("fixture", "spur", "SPUR", "bus"),
        ],
        # Pattern 2 is another topology/direction of the SAME opaque ORBIT route.
        "pat_line": np.array([0, 1, 0], np.int64),
        "pat_stop_off": np.array([0, 4, 6], np.int64),
        "pat_stops": np.array([0, 1, 2, 3, 4, 5, 2, 6], np.int64),
        "pat_nstops": np.array([4, 2, 2], np.int64),
        "pat_ntrips": np.array([1, 1, 1], np.int64),
        "pat_mat_off": np.array([0, 4, 6], np.int64),
        "pat_dep": np.array([
            28920, 29040, 29160, 29280,   # ORBIT
            29400, 29940,                 # SPUR (the seed's existing tail)
            29340, 29820,                 # ORBIT reboard (the sibling-discovered tail)
        ], np.int64),
        "pat_arr": np.array([
            28920, 29040, 29160, 29280,
            29400, 29940,
            29340, 29820,
        ], np.int64),
        # Route-at-stop index. The second ORBIT pattern boards directly at stop 2.
        "ras_off": np.array([0, 1, 2, 4, 5, 6, 7, 8], np.int64),
        "ras_pat": np.array([0, 0, 0, 2, 0, 1, 1, 2], np.int64),
        "ras_pos": np.array([0, 1, 2, 0, 3, 0, 1, 1], np.int64),
        # No inter-stop transfer edges are needed for the alternate tail.
        "tr_off": np.zeros(8, np.int64),
        "tr_to": np.array([], np.int64),
        "tr_time": np.array([], np.int64),
    }
    tree, jt = _mk_tree(None, data=data)
    eg = np.full(8, EGRESS_INF, np.int64)
    eg[2] = 600                         # eligible long walk from an ORBIT alight
    eg[3] = 300
    eg[6] = 60                          # short egress after ARC
    jt.egress_sec = eg

    seed = {
        "line": "ORBIT > SPUR",
        "total": 23,
        "stop": 0,
        "home": 28620,
        "jt": jt,
        "sig": (("transit", "ORBIT", 2, 0), ("transit", "SPUR", 9, 1)),
        "raw": [
            ("access", 180),
            ("ride", 0, 28920, 29040, 0, 1, 1),
            ("walk_t", 60, 1, 4),
            ("ride", 1, 29400, 29940, 0, 1, 5),
            ("egress", 60, 5),
        ],
        "it": {"total": 23, "legs": [
            {"mode": "walk", "line": None, "min": 3},
            {"mode": "transit", "line": "ORBIT", "min": 2, "wait": 0},
            {"mode": "walk", "line": None, "min": 1},
            {"mode": "transit", "line": "SPUR", "min": 9, "wait": 1},
            {"mode": "walk", "line": None, "min": 1},
        ]},
    }

    # Tail generation cannot act on the two-ride seed by itself.
    assert tree._planned_one_tail_branches(0, [seed], 30, geom_provider=None) == []

    closed = tree._planned_branch_closure(0, [seed, dict(seed)], 30, geom_provider=None)
    labels = {cand["line"] for cand in closed.values()}
    assert labels == {"ORBIT", "ORBIT > SPUR", "ORBIT > ORBIT"}
    # Duplicate seed removed; two genuine one-seat ORBIT alight shapes survive for the server's
    # structural dominance pass, alongside the seed and its reboard sibling.
    assert len(closed) == 4
    reboard = next(cand for cand in closed.values() if cand["line"] == "ORBIT > ORBIT")
    assert reboard["total"] <= 30
    assert reboard["sig"] != ()
    assert len(reboard["route_key"]) == 2                # A -> A is not collapsed to one ride


def _identity_tree(display_a="ALPHA", display_cross="ALPHA"):
    """Enumerator-only fixture with symbolic services, directions, a FIFO copy, and a loop."""
    data = {
        "line_table": [
            ("feed-one", "route-a", display_a, "bus"),
            ("feed-two", "route-a", display_cross, "bus"),
        ],
        # 0 forward, 1 reverse, 2 cross-feed, 3 FIFO-split copy of 0, 4 loop.
        "pat_line": np.array([0, 0, 1, 0, 0], np.int64),
        "pat_stop_off": np.array([0, 4, 8, 12, 16, 20], np.int64),
        "pat_nstops": np.array([4, 4, 4, 4, 4], np.int64),
        "pat_stops": np.array([
            0, 1, 2, 3,
            3, 2, 1, 0,
            10, 11, 12, 13,
            0, 1, 2, 3,
            4, 5, 4, 6,
        ], np.int64),
    }
    tree = DepartAfterJourneyTree.__new__(DepartAfterJourneyTree)
    tree.d = data
    tree.line_table = data["line_table"]
    tree.pat_line = data["pat_line"]
    return tree


def _identity_ride(pi, bpos=0, apos=3, dep=100, arr=200, alight=3):
    return ("ride", pi, dep, arr, bpos, apos, alight)


def test_planned_enumerator_identity_is_rename_invariant_and_preserves_structure():
    tree = _identity_tree("ALPHA", "ALPHA")
    forward = [_identity_ride(0)]
    original = tree._planned_route_identity(forward)

    renamed = _identity_tree("SYMBOL-X", "SYMBOL-X")
    assert renamed._planned_route_identity(forward) == original

    # Same underlying service, opposite topology/direction: not one candidate.
    assert tree._planned_route_identity([_identity_ride(1, alight=0)]) != original
    # Same public label and route id in another feed: not one candidate.
    assert tree._planned_route_identity([_identity_ride(2, alight=13)]) != original
    # FIFO-split pattern copies have the same service + topology + ridden positions: true duplicate.
    assert tree._planned_route_identity([_identity_ride(3)]) == original

    # Repeated stop ids in a loop are disambiguated by positions, not by the public service text.
    first_loop = [_identity_ride(4, 0, 2, alight=4)]
    later_loop = [_identity_ride(4, 2, 3, alight=6)]
    assert tree._planned_route_identity(first_loop) != tree._planned_route_identity(later_loop)

    # A genuine same-service reboard retains both ride occurrences and is labelled honestly.
    reboard = [_identity_ride(0, 0, 2, alight=2),
               _identity_ride(0, 2, 3, dep=240, arr=300, alight=3)]
    assert len(tree._planned_route_identity(reboard)) == 2
    assert tree._planned_route_identity(reboard) != original
    assert tree._planned_route_label(reboard) == "ALPHA > ALPHA"
    assert DepartAfterJourneyTree._geom_route_label([
        {"mode": "transit", "name": "ALPHA"},
        {"mode": "transit", "name": "ALPHA"},
    ]) == "ALPHA > ALPHA"


def test_planned_closure_dedupes_true_shape_not_display_label():
    tree = _identity_tree()
    tree._planned_walk_finish_branches = lambda *args, **kwargs: []
    tree._planned_one_tail_branches = lambda *args, **kwargs: []

    forward = [_identity_ride(0)]
    same_shape_later_trip = [_identity_ride(3, dep=400, arr=500)]  # FIFO copy, same topology
    reverse_same_label = [_identity_ride(1, alight=0)]
    seeds = [
        {"line": "ALPHA", "total": 20, "raw": forward, "sig": ("fast",)},
        {"line": "RENAMED", "total": 22, "raw": same_shape_later_trip, "sig": ("slow",)},
        {"line": "ALPHA", "total": 21, "raw": reverse_same_label, "sig": ("reverse",)},
    ]

    closed = tree._planned_branch_closure(0, seeds, 30)
    assert len(closed) == 2
    assert {cand["total"] for cand in closed.values()} == {20, 21}
    assert all(cand["line"] != "RENAMED" for cand in closed.values())


# --------------------------------------------------------------------------- B4
def test_fill_committed_leg_legacy_rule_skips_tiny_hop_unchanged():
    tr = (list(LEGS_TINY_FIRST), 28620)
    out = JourneyTree._empty_committed(1)
    JourneyTree._fill_committed_leg(out, 0, tr)          # arrive-by/legacy default
    assert out["commit_pi"][0] == 1                      # first SIGNIFICANT ride (the "19")
    assert int(out["commit_home"][0]) + int(out["commit_walk0"][0]) == 29100


def test_fill_committed_leg_planned_rule_commits_to_tiny_first_ride():
    tr = (list(LEGS_TINY_FIRST), 28620)
    out = JourneyTree._empty_committed(1)
    JourneyTree._fill_committed_leg(out, 0, tr, include_tiny=True)
    assert out["commit_pi"][0] == 0                      # the displayed (un-folded) tiny "22" hop
    assert int(out["commit_home"][0]) + int(out["commit_walk0"][0]) == 28920


def test_planned_committed_first_leg_matches_displayed_first_ride():
    """B4 end-to-end: planned display shows the tiny hop as its first transit leg (fold_tiny is
    off), so the MC must commit to that same boarding — not the later significant ride."""
    tree, _jt = _mk_tree(LEGS_TINY_FIRST, T=30000)
    res = tree.itinerary(0)
    painted = (30000 - 28800 + 180 + 59) // 60           # 23 min
    assert res["total"] == painted == 23
    assert _legs_sum(res) == res["total"]
    transit = [l for l in res["legs"] if l["mode"] == "transit"]
    assert [t["line"] for t in transit] == ["22", "19"]  # tiny hop displayed first
    assert transit[0].get("wait", 0) == 0                # first wait folded into the walk
    assert transit[1].get("wait", 0) == 1                # the real transfer wait survives
    cl = tree.committed_first_legs()
    assert cl["commit_pi"][0] == 0                       # committed == the displayed first ride
    assert int(cl["commit_home"][0]) + int(cl["commit_walk0"][0]) == 28920


def test_compiled_one_tail_discovery_matches_python_oracle_row_for_row():
    """The pin accelerator preserves schedule iteration order and every tie-breaking field."""
    from core.raptor_planned import (
        discover_one_tail_variants,
        discover_one_tail_variants_python,
    )

    data = {
        "pat_nstops": np.array([3, 3, 2], np.int64),
        "pat_ntrips": np.array([1, 2, 1], np.int64),
        "pat_stop_off": np.array([0, 3, 6], np.int64),
        "pat_mat_off": np.array([0, 3, 9], np.int64),
        "pat_stops": np.array([0, 1, 2, 1, 3, 5, 4, 5], np.int64),
        "pat_dep": np.array([100, 200, 300, 250, 350, 450, 400, 500, 600,
                             350, 500], np.int64),
        "pat_arr": np.array([100, 200, 300, 250, 350, 450, 400, 500, 600,
                             350, 500], np.int64),
        "ras_off": np.array([0, 1, 3, 4, 5, 6, 8], np.int64),
        "ras_pat": np.array([0, 0, 1, 0, 1, 2, 1, 2], np.int64),
        "ras_pos": np.array([0, 1, 0, 2, 1, 0, 2, 1], np.int64),
        "tr_off": np.array([0, 0, 1, 2, 2, 2, 2], np.int64),
        "tr_to": np.array([4, 4], np.int64),
        "tr_time": np.array([30, 800], np.int64),       # second transfer exceeds the 12m cap
    }
    egress = np.array([EGRESS_INF, EGRESS_INF, 120, 100, EGRESS_INF, 30], np.int64)
    expected = discover_one_tail_variants_python(
        data, egress, first_pi=0, first_dep=100, first_bpos=0,
        board_slack=60, egress_inf=EGRESS_INF)
    actual = discover_one_tail_variants(
        data, egress, first_pi=0, first_dep=100, first_bpos=0,
        board_slack=60, egress_inf=EGRESS_INF)

    np.testing.assert_array_equal(actual, expected)
    assert expected.shape == (2, 12)


def test_planned_access_closure_is_station_local_not_whole_service_route():
    tree = DepartAfterJourneyTree.__new__(DepartAfterJourneyTree)
    tree.access_off = np.array([0, 4], np.int64)
    tree.access_to = np.array([0, 1, 2, 3], np.int64)
    tree.d = {
        "stop_lat": np.array([37.0, 37.0008, 37.0060, 37.0003]),
        "stop_lon": np.array([-122.0, -122.0, -122.0, -122.0002]),
    }

    closed = tree.planned_access_stop_closure(0, {0, 99}, station_radius_m=300)

    assert closed == {0, 1, 3}
    assert 2 not in closed       # a farther stop on the same service is another boarding corridor


def test_planned_branch_access_filter_is_label_and_chip_cap_independent():
    tree = DepartAfterJourneyTree.__new__(DepartAfterJourneyTree)
    tree.access_off = np.array([0, 4], np.int64)
    tree.access_to = np.array([0, 1, 2, 3], np.int64)
    tree.access_w = np.array([60, 0, 0, 0], np.int64)
    tree.max_min = 75
    tree._planned_stop_anchors = lambda: (
        np.array([540, 540, 1200, 600], np.int64), None, None)
    calls = []
    templates = {
        0: (0, 540, (("walk", 540),), "SAME LABEL", ("corridor-z",)), # 10m: keep
        1: (0, 900, (("walk", 900),), "SAME LABEL", ("corridor-a",)), # 15m: drop
        2: (0, 1200, (("walk", 1200),), "FAR", ("corridor-far",)),    # 20m: drop
        3: None,                                                # unboardable: drop
    }

    def validated(stop):
        calls.append(int(stop))
        return templates[int(stop)]

    tree._validated_stop_template = validated
    assert tree.planned_branch_access_stops(0, 10, 2) == {0}
    assert calls == [0, 1, 2, 3]


# --------------------------------------------------------- planned branch probe collapse (P3)
def test_branch_probe_collapse_reuses_stop_deadline_tail_without_losing_earlier_valid_board():
    """A tree path is fixed by (stop, T), but B3 validation is fixed by the requested B.

    The later 500-second board below is invalid for the shared 490-second ride, while the earlier
    480-second board is valid.  A tempting ``(stop, T)`` collapse that keeps only the latest B
    would hide the real earlier branch.  This is a pure helper test: no RAPTOR/Numba execution.
    """
    from core.raptor_planned import evaluate_planned_branch_probes

    tails = {
        (7, 900): (("ride", "shared", 490, 700, 0, 1, 9), ("egress", 30, 9)),
        (8, 900): (("ride", "other", 510, 720, 0, 1, 10), ("egress", 20, 10)),
    }
    probes = [
        (7, 500, 900, 60),      # invalid: latest B misses the shared 490 departure
        (8, 510, 900, 30),      # different stop: must retain this position in output order
        (7, 480, 900, 60),      # earlier B is valid on the SAME (stop, T) tail
        (7, 480, 900, 120),     # same board/tail, distinct access/home anchor: preserve it
    ]
    calls = []

    def trace_tail(stop, deadline):
        calls.append((stop, deadline))
        return tails[(stop, deadline)]

    def valid_after_start(tail, board):
        return int(tail[0][2]) >= int(board)

    actual = list(evaluate_planned_branch_probes(probes, trace_tail, valid_after_start))
    assert calls == [(7, 900), (8, 900)]                 # shared tree tail traced exactly once
    assert [(s, B, T, aw) for s, B, T, aw, _tail in actual] == [
        (8, 510, 900, 30), (7, 480, 900, 60), (7, 480, 900, 120),
    ]

    # Differential oracle: the legacy loop retraced every probe, then built access/home from it.
    def legacy():
        out = []
        for stop, board, deadline, access_walk in probes:
            tail = tails[(stop, deadline)]
            if valid_after_start(tail, board):
                out.append((stop, board, deadline, access_walk, tail))
        return out

    def observable(rows):
        return [
            {
                "raw": (("access", access_walk), *tail),
                "home": board - access_walk,
                # In production route identity ignores access and trip clock, so this also
                # guards the exact inputs used by the existing structural dedupe/tie path.
                "route_key": tuple(leg[1] for leg in tail if leg[0] == "ride"),
            }
            for _stop, board, _deadline, access_walk, tail in rows
        ]

    assert observable(actual) == observable(legacy())


def test_branch_enumerator_probe_collapse_matches_old_loop_oracle():
    """The real enumerator keeps legacy candidates while reducing ``_trace_from`` calls.

    This exercises the integration rather than only the generic helper.  Both board times point
    to one deadline tree; 480 can board the fixture vehicle while 500 cannot.  The legacy oracle
    calls the old access/home-shaped trace once per probe, while the implementation traces a
    canonical access-free tail once and reconstructs the exact surviving raw/home candidate.
    """
    class _BranchTree:
        def __init__(self):
            self.calls = []

        def _trace_from(self, stop, access_walk, latest_home):
            self.calls.append((int(stop), int(access_walk), int(latest_home)))
            return [
                ("access", int(access_walk)),
                ("ride", 0, 490, 700, 0, 1, 9),
                ("egress", 30, 9),
            ], int(latest_home)

    jt = _BranchTree()
    tree = DepartAfterJourneyTree.__new__(DepartAfterJourneyTree)
    tree.planned = True
    tree.max_min = 75
    tree.access_off = np.array([0, 1], np.int64)
    tree.access_to = np.array([7], np.int64)
    tree.access_w = np.array([60], np.int64)
    tree.dep_grid = np.array([480, 500], np.int64)
    tree.cell_deps = np.array([480, 500], np.int64)
    tree.arrivalW = np.full((8, 2), _INF, np.int64)
    tree.arrivalW[7] = np.array([900, 900], np.int64)
    tree._tree_at = lambda deadline: jt
    tree._planned_route_identity = lambda raw: tuple(
        (leg[1], leg[4], leg[5]) for leg in raw if leg[0] == "ride")
    tree._format_planned_raw = lambda ci, stop, raw, home, source, geom_provider=None, planned_total=None, planned_target_sec=None: {
        "total": int(planned_total),
        "legs": [{"mode": "transit", "line": "fixture", "min": int(planned_total)}],
    }
    tree._planned_branch_closure = lambda ci, candidates, max_total, geom_provider=None: {
        cand["route_key"]: cand for cand in candidates
    }

    actual = tree.planned_branch_itineraries(0, 10, 10, geom_provider=None)
    assert jt.calls == [(7, 0, 0)]

    # The pre-collapse loop is the direct oracle.  It must retain exactly the same route key,
    # raw access prefix, home anchor, and ranking fields as the optimized enumerator.
    jt.calls.clear()
    expected = []
    for B in (480, 500):
        T = 900
        awk = 60
        raw, home = jt._trace_from(7, awk, B - awk)
        if not tree._raw_chain_valid_after_start(raw, B):
            continue
        planned_total = (tree._planned_raw_total_sec(raw, home) + 59) // 60
        expected.append({
            "total": planned_total,
            "raw": raw,
            "home": home,
            "route_key": tree._planned_route_identity(raw),
        })
    assert jt.calls == [(7, 60, 420), (7, 60, 440)]

    def observable(candidates):
        return [{key: cand[key] for key in ("total", "raw", "home", "route_key")}
                for cand in candidates]

    assert observable(actual) == expected
