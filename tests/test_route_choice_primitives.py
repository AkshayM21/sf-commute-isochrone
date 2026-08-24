"""Direct tests for the server-free route-choice primitives."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

from core.route_choice_primitives import (
    alt_access_walk_min,
    alt_metric_total,
    alt_physical_walk_min,
    alt_quality_rank,
    recommend_route_choice,
    route_label,
    route_sig,
    route_trace_sig,
)


def _route(name, total, *, real=None, walk=0, physical=None, frag=0):
    leg = {"mode": "walk", "min": walk, "pts": []}
    if physical is not None:
        leg["physical_min"] = physical
    transit = {"mode": "transit", "name": name, "min": 10, "wait": 1,
               "pts": [(37.7, -122.4), (37.71, -122.39)]}
    out = {
        "frag": frag,
        "typical": {"total": total, "legs": [leg, transit]},
        "best": {"total": total, "legs": [leg, transit]},
    }
    if real is not None:
        out["real"] = real
    return out


def test_route_signatures_and_labels_are_display_stable():
    legs = [
        {"mode": "walk", "min": 2},
        {"mode": "transit", "line": "A", "min": 7, "wait": 1,
         "pts": [(37.7, -122.4), (37.71, -122.39)]},
    ]
    assert route_label(legs) == "A"
    assert route_sig(legs) == (("walk", "", 2, 0), ("transit", "A", 7, 1))
    assert route_trace_sig(legs)[1][-1] == "377000_-1224000_377100_-1223900"


def test_walk_metrics_keep_schedule_allowance_out_of_physical_walk():
    route = _route("A", 22, walk=12, physical=2)
    assert alt_access_walk_min(route) == 2
    assert alt_physical_walk_min(route) == 2


def test_recommendation_uses_active_metric_then_physical_walk():
    realistic = _route("A", 22, real=21, walk=5, physical=5)
    best = _route("B", 20, real=24, walk=2, physical=2)
    assert alt_metric_total(realistic, "r") == 21
    assert recommend_route_choice(realistic, [best], "r") is realistic
    assert recommend_route_choice(realistic, [best], "b") is best


def test_quality_rank_accepts_explicit_tie_break_callback():
    first = _route("A", 22)
    second = _route("B", 22)
    rank = lambda route: "a" if route is first else "b"
    assert alt_quality_rank(first, selection_tie_key=rank) < alt_quality_rank(
        second, selection_tie_key=rank)
