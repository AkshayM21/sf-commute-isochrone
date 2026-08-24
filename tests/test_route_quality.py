"""Structural route-quality contracts for the graph-native engine.

The old data-snapshot parity scans lived here.  Current route geometry, family
selection, and transfer behavior are covered by synthetic graph-native fixtures,
which are deterministic and do not require downloaded artifacts.
"""
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "scripts"))


def _leg(name, minutes, pts=None):
    return {"mode": "transit", "name": name, "line": name, "min": minutes,
            "pts": pts or [[37.70, -122.45], [37.71, -122.44]]}


def test_reconcile_legs_preserves_map_total_without_overshoot():
    from core.raptor_journey import reconcile_legs

    legs = [{"mode": "walk", "min": 3}, _leg("22", 12), {"mode": "walk", "min": 2}]
    result = reconcile_legs(legs, 20)
    assert sum(int(leg["min"]) + int(leg.get("wait", 0))
               for leg in result["legs"]) == 20
    assert result["legs"][1]["name"] == "22"


def test_min_overshoot_alight_prefers_earliest_feasible_finish_then_walk():
    from core.raptor_journey import EGRESS_INF, _min_overshoot_alight

    # One route has two forward stops. Their total finishes tie; the helper keeps
    # the shorter egress and later-position tie-break.
    pat_arr = np.array([100, 200, 205], np.int64)
    pat_stops = np.array([0, 1, 2], np.int64)
    egress = np.array([EGRESS_INF, 10, 5], np.int64)
    result = _min_overshoot_alight(
        pat_arr, pat_stops, egress, 0, 0, 0, 3, 0, 1 << 40)
    assert result == (2, 5)


def test_min_overshoot_alight_handles_unreachable_egress_safely():
    from core.raptor_journey import EGRESS_INF, _min_overshoot_alight

    pat_arr = np.array([100, 200], np.int64)
    pat_stops = np.array([0, 1], np.int64)
    egress = np.array([EGRESS_INF, EGRESS_INF], np.int64)
    assert _min_overshoot_alight(
        pat_arr, pat_stops, egress, 0, 0, 0, 2, 0, 1 << 40) == (0, 1 << 40)
