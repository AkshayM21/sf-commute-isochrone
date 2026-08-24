"""Direct regression tests for the pure traced-journey response seam."""

import os
import sys

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts"))

from core import route_response  # noqa: E402
from core import raptor_journey  # noqa: E402


def test_reconcile_legs_preserves_schema_rounding_and_waits():
    legs = [
        {"mode": "walk", "line": None, "min": 0, "segs": [("access",)]},
        {"mode": "transit", "line": "A", "min": 4, "wait": 1},
        {"mode": "walk", "line": None, "min": 2},
    ]
    result = route_response.reconcile_legs(legs, 8)
    assert result == {
        "total": 8,
        "xfers": 0,
        "legs": [
            {"mode": "transit", "line": "A", "min": 4, "wait": 1},
            {"mode": "walk", "line": None, "min": 3},
        ],
    }
    assert raptor_journey.reconcile_legs(
        [{"mode": "transit", "line": "A", "min": 4, "wait": 1}], 6
    ) == {
        "total": 6,
        "xfers": 0,
        "legs": [{"mode": "transit", "line": "A", "min": 4, "wait": 1},
                 {"mode": "walk", "line": None, "min": 1}],
    }


def test_format_legs_matches_historical_largest_remainder_and_metadata():
    out = [
        {"mode": "walk", "line": None, "sec": 30, "segs": [("access",)]},
        {"mode": "transit", "line": "A", "sec": 150, "wait_sec": 30},
        {"mode": "walk", "line": None, "sec": 30, "segs": [("egress", 9)]},
    ]
    expected = {
        "total": 4,
        "xfers": 0,
        "legs": [
            {"mode": "walk", "line": None, "min": 1, "segs": [("access",)]},
            {"mode": "transit", "line": "A", "min": 3, "wait": 0},
            {"mode": "walk", "line": None, "min": 0, "segs": [("egress", 9)]},
        ],
    }
    # The final zero-minute walk is dropped by reconciliation, including its geometry.
    expected["legs"].pop()
    assert route_response.format_legs(out, 4) == expected
    assert raptor_journey.JourneyTree._format(object(), out, 4) == expected


def test_format_legs_honors_historical_reconcile_monkeypatch(monkeypatch):
    calls = []

    def fake(legs, total):
        calls.append((legs, total))
        return {"total": total, "xfers": 99, "legs": legs}

    monkeypatch.setattr(raptor_journey, "reconcile_legs", fake)
    out = [{"mode": "walk", "line": None, "sec": 60}]
    assert raptor_journey.JourneyTree._format(object(), out, 1)["xfers"] == 99
    assert calls and calls[0][1] == 1


def test_push_walk_merges_seconds_and_geometry_descriptors():
    out = []
    route_response._push_walk(out, 20, ("access",))
    route_response._push_walk(out, 40, ("walkt", 1, 2))
    route_response._push_walk(out, 0, ("egress", 3))
    assert out == [{
        "mode": "walk", "line": None, "sec": 60,
        "segs": [("access",), ("walkt", 1, 2)],
    }]
    # Historical module name remains a dynamic compatibility wrapper.
    out2 = []
    raptor_journey._push_walk(out2, 1)
    assert out2 == [{"mode": "walk", "line": None, "sec": 1}]


def test_footpath_sec_and_no_overshoot_alight_tie_breaks():
    off = np.array([0, 2, 2], dtype=np.int64)
    to = np.array([1, 2], dtype=np.int64)
    times = np.array([17, 23], dtype=np.int64)
    assert route_response._footpath_sec(off, to, times, 0, 2) == 23
    assert route_response._footpath_sec(off, to, times, 1, 1) == 0

    # Candidate positions 1 and 2 finish at the same absolute time. Position 2 wins
    # when both egress durations are equal, preserving the legacy later-position tie-break.
    arr = np.array([100, 200, 200], dtype=np.int64)
    stops = np.array([10, 11, 12], dtype=np.int64)
    egress = np.full(13, route_response.EGRESS_INF, dtype=np.int64)
    egress[10] = 100
    egress[11] = 50
    egress[12] = 50
    assert route_response._min_overshoot_alight(arr, stops, egress, 0, 0, 0, 3, 1, 150) == (2, 50)
