"""Exactness checks for the benchmark-only committed-MC stage helpers."""
import os
import sys

import numpy as np
import pytest


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "scripts"))


def _inputs():
    # Two joined patterns, matching the smallest fixture that exercises a committed transfer tail.
    data = {
        "n_stops": 3,
        "pat_nstops": np.array([2, 2], np.int32),
        "pat_ntrips": np.array([3, 3], np.int32),
        "pat_stop_off": np.array([0, 2], np.int32),
        "pat_mat_off": np.array([0, 6], np.int32),
        "pat_stops": np.array([0, 1, 1, 2], np.int32),
        "pat_dep": np.array([100, 150, 220, 270, 340, 390,
                             240, 300, 360, 420, 480, 540], np.int32),
        "pat_arr": np.array([100, 160, 220, 280, 340, 400,
                             240, 300, 360, 420, 480, 540], np.int32),
        "ras_off": np.array([0, 1, 3, 4], np.int32),
        "ras_pat": np.array([0, 0, 1, 1], np.int32),
        "ras_pos": np.array([0, 1, 0, 1], np.int32),
        "tr_off": np.array([0, 0, 0, 0], np.int32),
        "tr_to": np.empty(0, np.int32),
        "tr_time": np.empty(0, np.int32),
    }
    legs = {
        "commit_home": np.array([100, 220, 0, 0], np.int64),
        "commit_kind": np.array([2, 2, 1, 0], np.int64),
        "commit_walk0": np.array([0, 0, 0, 0], np.int64),
        "commit_pi": np.array([0, 0, 0, 0], np.int64),
        "commit_bpos": np.array([0, 0, 0, 0], np.int64),
        "commit_apos": np.array([1, 1, 0, 0], np.int64),
        "commit_as": np.array([1, 1, 0, 0], np.int64),
    }
    deadlines = np.arange(300, 661, 60, dtype=np.int64)
    perfect = np.array([5, 5, 17, -1], np.int64)
    d0 = np.array([[0.9, 20.2, 0.0, 1.1, 11.9, 0.0],
                   [200.75, 0.0, 0.0, 140.5, 0.0, 0.0]], np.float64)
    slope = np.array([[0.017, 0.003, 0.0, 0.011, 0.005, 0.0],
                      [0.019, 0.0, 0.0, 0.013, 0.0, 0.0]], np.float64)
    return data, legs, deadlines, perfect, d0, slope


def test_stage_helpers_assemble_exact_production_committed_arrays():
    from core import raptor as R
    from core import raptor_numba as RN

    if R._select_kernel() != "numba":
        pytest.skip("numba not installed")
    data, legs, deadlines, perfect, d0, slope = _inputs()
    flat = R._mc_flat_args(data)
    (n_stops, pat_nstops, pat_ntrips, pat_stop_off, pat_mat_off, pat_trip_off,
     pat_stops, pat_dep, pat_arr, ras_off, ras_pat, ras_pos, tr_off, tr_to, tr_time) = flat
    staged_commute = np.empty((len(perfect), d0.shape[0]), np.float64)
    staged_tail = np.empty((d0.shape[0], int(n_stops), deadlines.size), np.uint16)
    staged_valid = np.empty(d0.shape[0], np.uint8)
    for draw in range(d0.shape[0]):
        dep = np.empty(pat_dep.size, np.int64)
        arr = np.empty(pat_arr.size, np.int64)
        RN._perturb(pat_nstops, pat_ntrips, pat_mat_off, pat_trip_off,
                    pat_dep, pat_arr, d0[draw], slope[draw], dep, arr)
        latest = RN._profile(
            n_stops, pat_nstops, pat_ntrips, pat_stop_off, pat_mat_off, pat_stops,
            dep, arr, ras_off, ras_pat, ras_pos, tr_off, tr_to, tr_time,
            np.array([2], np.int32), np.array([0], np.int64), deadlines, 60, 4)
        RN._mc_stage_score_committed_draw(
            pat_nstops, pat_ntrips, pat_mat_off, deadlines, 60,
            legs["commit_home"], legs["commit_kind"], legs["commit_walk0"],
            legs["commit_pi"], legs["commit_bpos"], legs["commit_apos"], legs["commit_as"],
            perfect, 99, latest, dep, arr, staged_commute[:, draw])
        staged_valid[draw] = RN._mc_stage_encode_tail(latest, deadlines, staged_tail[draw])

    production = RN.montecarlo_committed(
        *flat, np.array([2], np.int32), np.array([0], np.int64), deadlines, 60, 4,
        legs["commit_home"], legs["commit_kind"], legs["commit_walk0"], legs["commit_pi"],
        legs["commit_bpos"], legs["commit_apos"], legs["commit_as"], perfect, 99,
        d0, slope, True)
    assert np.array_equal(staged_commute, production[0])
    assert np.array_equal(staged_tail, production[1])
    assert np.array_equal(staged_valid, production[2])
