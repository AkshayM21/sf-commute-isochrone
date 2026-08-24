"""Unit tests for the server-free planned-journey primitives."""

from scripts.core.planned_journey_primitives import (
    alight_tail_better,
    attach_walk_truth,
    fold_first_visible_wait,
    geom_route_label,
    geom_route_sig,
    planned_candidate_better,
    planned_candidate_quality,
    planned_raw_total_sec,
    raw_access_sec,
    raw_final_walk_sec,
    raw_transit_sec,
    raw_transfer_count,
    raw_transfer_walk_sec,
    reconcile_planned_target,
)


def test_fold_first_wait_preserves_physical_walk_and_adds_allowance_only_leg():
    out = [{"mode": "transit", "wait_sec": 45, "segs": [("ride",)]}]
    fold_first_visible_wait(out)
    assert out == [
        {
            "mode": "walk", "line": None, "sec": 45,
            "physical_sec": 0, "schedule_allowance_sec": 45,
            "segs": [("access",)],
        },
        {"mode": "transit", "wait_sec": 0, "segs": [("ride",)]},
    ]


def test_walk_truth_and_reconcile_keep_allowance_out_of_physical_time():
    out = [{
        "mode": "walk", "sec": 150,
        "physical_sec": 90, "schedule_allowance_sec": 60,
    }]
    assert reconcile_planned_target(out, 150, 180) == 0
    assert out[0]["physical_sec"] == 90
    assert out[0]["schedule_allowance_sec"] == 90
    res = {"legs": [{"mode": "walk", "min": 3}]}
    attach_walk_truth(res, out)
    assert res["legs"][0]["physical_min"] == 1.5
    assert res["legs"][0]["schedule_allowance_min"] == 1.5


def test_geometry_helpers_preserve_label_and_signature_shape():
    geom = [
        {"mode": "walk", "line": None, "min": 2},
        {"mode": "transit", "name": "K", "min": 7, "wait": 1},
        {"mode": "transit", "line": "19", "min": 4},
    ]
    assert geom_route_label(geom) == "K > 19"
    assert geom_route_sig(geom) == (
        ("walk", "", 2, 0), ("transit", "K", 7, 1), ("transit", "19", 4, 0)
    )


def test_raw_metrics_match_historical_tuple_contract():
    raw = [
        ("access", 120),
        ("ride", 3, 1000, 1300, 0, 2, 11),
        ("walk_t", 45, 11, 12),
        ("ride", 4, 1400, 1600, 0, 1, 12),
        ("egress", 180, 12),
    ]
    assert raw_access_sec(raw) == 120
    assert raw_transfer_walk_sec(raw) == 45
    assert raw_transfer_count(raw) == 1
    assert raw_transit_sec(raw) == 500
    assert raw_final_walk_sec(raw) == 180
    assert planned_raw_total_sec(raw, 880) == 900


def test_candidate_rank_helpers_keep_exact_time_before_display_ties():
    early = {
        "raw": [("access", 120), ("ride", 1, 1000, 1500, 0, 1, 2)],
        "home": 880, "total": 11, "route_key": (("a",),),
    }
    late = {
        "raw": [("access", 180), ("ride", 1, 1000, 1500, 0, 1, 2)],
        "home": 820, "total": 11, "route_key": (("a",),),
    }
    assert planned_candidate_quality(early) < planned_candidate_quality(late)
    assert planned_candidate_better(early, late)
    assert alight_tail_better(1260, 4, 540, (1260, 3, 11, 660))
