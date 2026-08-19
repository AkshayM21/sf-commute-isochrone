"""Regression cases for `_profile`'s monotone selected-trip search prune.

The reverse scan retains the highest trip selected downstream even when that trip cannot board at
the current position.  These intentionally small cases exercise that non-obvious invariant, the
FIFO successor rejection, exact equality at the successor, and the already-last-trip fast path.
"""
import os
import sys

import numpy as np
import pytest


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "scripts"))


def _data(final_arrival):
    """One FIFO three-trip line, with seed transfers to its upstream stops."""
    # At the downstream egress, the 600-second deadline selects trip 1 unless the final arrival
    # is lowered to 520.  A seed transfer supplies `prev` at the upstream positions so the same
    # pattern scan must revisit its selected trip there.
    schedule = np.array([
        100, 180, 300,
        200, 300, 500,
        400, 510, final_arrival,
    ], dtype=np.int64)
    return {
        "n_stops": 3,
        "pat_nstops": np.array([3], dtype=np.int32),
        "pat_ntrips": np.array([3], dtype=np.int32),
        "pat_stop_off": np.array([0], dtype=np.int32),
        "pat_mat_off": np.array([0], dtype=np.int32),
        "pat_stops": np.array([0, 1, 2], dtype=np.int32),
        "pat_dep": schedule.copy(),
        "pat_arr": schedule.copy(),
        "ras_off": np.array([0, 1, 2, 3], dtype=np.int32),
        "ras_pat": np.array([0, 0, 0], dtype=np.int32),
        "ras_pos": np.array([0, 1, 2], dtype=np.int32),
        # One snapshot seed pass: stop 2 (the egress) reaches stops 0 and 1.  Test cases choose
        # the durations, so a `prev` label can intentionally be earlier than the old trip's
        # arrival at the current position.
        "tr_off": np.array([0, 0, 0, 2], dtype=np.int32),
        "tr_to": np.array([0, 1], dtype=np.int32),
        "tr_time": np.array([560, 0], dtype=np.int64),
    }


@pytest.mark.parametrize(
    ("name", "final_arrival", "to_stop_1", "expected_stop_0"),
    [
        # The selected downstream trip is 1.  At stop 1, trip 2 arrives exactly at `prev - 60`,
        # so the right-side upper-bound must advance to trip 2 (not reject equality).
        ("successor_tie_advances", 700, 30, 400),
        # Trip 1 selected downstream is not boardable at stop 1 (arrival 300 > key 240), and
        # neither is its successor.  It must nevertheless remain selected, just as max(old, idx)
        # did before the prune, and propagate its 200-second upstream departure.
        ("unboardable_old_trip_is_retained", 700, 300, 200),
        # The egress itself selects the last trip.  The fast path must not read past its arrival
        # column, and that last trip continues to propagate upstream.
        ("already_last_trip", 520, 300, 400),
    ],
)
def test_profile_monotone_trip_prune_matches_python_reference(
        name, final_arrival, to_stop_1, expected_stop_0):
    from core import raptor as R
    from core import raptor_numba as RN

    if R._select_kernel() != "numba":
        pytest.skip("numba not installed")
    data = _data(final_arrival)
    data["tr_time"][1] = to_stop_1
    deadlines = np.array([600], dtype=np.int64)
    egress_g = np.array([2], dtype=np.int32)
    egress_w = np.array([0], dtype=np.int64)

    actual = RN.reverse_profile(data, egress_g, egress_w, deadlines, 60, 2)
    reference = R.reverse_profile(
        data, egress_g, egress_w, deadlines, board_slack=60, max_rounds=2, kernel="python")
    assert np.array_equal(actual, reference), name
    assert int(actual[0, 0]) == expected_stop_0, name


def _random_fifo_data(rng, n_trips):
    """Small valid FIFO network with arbitrary trip count, including zero.

    Each position's arrival column is sorted, and each trip progresses forward along the pattern.
    A few ordinary footpaths make successive marked-stop rounds possible without relying on a
    particular line name, timetable interval, or the hand-written cases above.
    """
    n_stops = 4
    if n_trips:
        # The two cumulative maxima impose both GTFS-like in-trip time order and the FIFO column
        # invariant the optimized binary search relies on.  The raw perturbation is deliberately
        # irregular, rather than a fixed-frequency synthetic timetable.
        raw = rng.integers(60, 760, size=(n_trips, n_stops), dtype=np.int64)
        schedule = np.maximum.accumulate(raw, axis=1)
        schedule = np.maximum.accumulate(schedule, axis=0)
        flat_schedule = schedule.ravel()
    else:
        flat_schedule = np.empty(0, dtype=np.int64)

    # Egress -> upstream starts the first active pattern queue.  The two additional forward
    # footpaths can mark a later queue/round after a transit improvement.  All transfers are
    # non-negative and therefore valid under the same one-hop snapshot policy as production.
    transfer_from = [3, 3, 3, 0, 1]
    transfer_to = [0, 1, 2, 1, 2]
    transfer_time = rng.integers(0, 360, size=len(transfer_from), dtype=np.int64)
    tr_off = np.zeros(n_stops + 1, dtype=np.int32)
    for source in transfer_from:
        tr_off[source + 1] += 1
    tr_off = np.cumsum(tr_off, dtype=np.int32)
    # Sources were emitted in ascending groups except for the three egress edges.  Build a CSR
    # order explicitly so randomized cases do not accidentally test an invalid offset layout.
    order = np.argsort(np.asarray(transfer_from), kind="stable")
    return {
        "n_stops": n_stops,
        "pat_nstops": np.array([n_stops], dtype=np.int32),
        "pat_ntrips": np.array([n_trips], dtype=np.int32),
        "pat_stop_off": np.array([0], dtype=np.int32),
        "pat_mat_off": np.array([0], dtype=np.int32),
        "pat_stops": np.arange(n_stops, dtype=np.int32),
        "pat_dep": flat_schedule.copy(),
        "pat_arr": flat_schedule.copy(),
        "ras_off": np.arange(n_stops + 1, dtype=np.int32),
        "ras_pat": np.zeros(n_stops, dtype=np.int32),
        "ras_pos": np.arange(n_stops, dtype=np.int32),
        "tr_off": tr_off,
        "tr_to": np.asarray(transfer_to, dtype=np.int32)[order],
        "tr_time": transfer_time[order],
    }


def test_profile_trip_prune_fifo_property_differential():
    """Deterministic randomized FIFO matrices retain full-profile byte equality.

    This covers zero, singleton, and multi-trip patterns; varied deadlines and round caps; no
    boardable trip; equality; last-trip selection; and an upstream label too early to board the
    selected trip.  The pure sweep remains the independent full-range-search oracle.
    """
    from core import raptor as R
    from core import raptor_numba as RN

    if R._select_kernel() != "numba":
        pytest.skip("numba not installed")
    rng = np.random.default_rng(20260809)
    egress_g = np.array([3], dtype=np.int32)
    egress_w = np.array([0], dtype=np.int64)
    # Keep the matrix small enough that this is a bounded unit test while varying all search
    # states across a fixed, reproducible set of valid schedules.
    for n_trips in (0, 1, 2, 5):
        for sample in range(5):
            data = _random_fifo_data(rng, n_trips)
            if n_trips and sample == 0:
                # A valid but entirely post-horizon timetable: every profile board search must
                # return no trip, including its first `trip == -1` suffix probe.
                data["pat_dep"] += 5000
                data["pat_arr"] += 5000
            deadlines = np.sort(rng.integers(300, 1000, size=4, dtype=np.int64))
            # The profile contract requires strictly ascending deadline columns.
            deadlines = np.unique(deadlines)
            if deadlines.size < 2:
                deadlines = np.array([420, 780], dtype=np.int64)
            max_rounds = int(rng.integers(1, 5))
            reference = R.reverse_profile(
                data, egress_g, egress_w, deadlines, board_slack=60,
                max_rounds=max_rounds, kernel="python")
            actual = RN.reverse_profile(
                data, egress_g, egress_w, deadlines, 60, max_rounds)
            assert np.array_equal(actual, reference), (n_trips, max_rounds, deadlines.tolist())
