"""Structural regression tests for the graph-native RAPTOR engine.

These checks deliberately use small in-memory fixtures.  Route quality and API
contracts are covered by the graph/route suites; this module focuses on pure
kernel dispatch and planned-departure boundary invariants.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "scripts"))


def test_tail_lag_encoding_is_lossless_or_rejected():
    """uint16 capture reserves 65535 for unreachable and rejects reachable overflow."""
    from core import raptor as R

    deadlines = np.array([100_000, 100_060, 100_120], np.int64)
    latest = np.array([
        [100_000, 99_999, R.NEG],
        [99_940, 34_526, 100_120],
    ], np.int64)
    encoded, valid = R._encode_tail_lag(latest, deadlines)
    assert valid
    assert encoded.tolist() == [[0, 61, 65535], [60, 65534, 0]]

    overflow = latest.copy()
    overflow[1, 1] -= 1
    encoded2, valid2 = R._encode_tail_lag(overflow, deadlines)
    assert not valid2
    assert encoded2[1, 1] == 65535


def test_committed_deadline_prefix_boundaries_and_reduction():
    """The production MC horizon includes the exact cap and ignores non-transit rows."""
    from core.raptor_engine import _committed_deadline_prefix

    def legs(kind, home):
        return {"commit_kind": np.asarray(kind, np.int8),
                "commit_home": np.asarray(home, np.int64)}

    grid = np.arange(0, 301, 60, dtype=np.int64)
    assert np.array_equal(
        _committed_deadline_prefix(grid, legs([2], [0]), 2),
        np.array([0, 60, 120], np.int64))
    assert np.array_equal(
        _committed_deadline_prefix(grid, legs([2, 0, 2], [0, -(1 << 60), 120]), 2),
        grid[:5])
    assert len(_committed_deadline_prefix(grid, legs([0, 1], [-(1 << 60), 0]), 2)) == 1
    assert np.array_equal(
        _committed_deadline_prefix(grid, legs([2], [-(1 << 60)]), 2), grid)
    assert np.array_equal(
        _committed_deadline_prefix(grid[::-1], legs([2], [0]), 2), grid[::-1])

    production = np.arange(8 * 3600 + 35 * 60,
                           8 * 3600 + 35 * 60 + 131 * 60, 60, dtype=np.int64)
    early = _committed_deadline_prefix(production, legs([2], [production[0]]), 75)
    late = _committed_deadline_prefix(production, legs([2], [production[30]]), 75)
    assert (len(production), len(early), len(late)) == (131, 76, 106)


def test_select_planned_nonuniform_anchors_on_grid_point():
    """A nonuniform planned grid anchors at the represented profile column."""
    from core import raptor as R

    inf = int(R.INF)
    access_off = np.array([0, 1], np.int64)
    access_to = np.array([0], np.int64)
    access_w = np.array([60], np.int64)
    purewalk = np.array([-1], np.int64)
    dep_grid = np.array([0, 100, 200], np.int64)
    cell_deps = np.array([0, 50], np.int64)
    arrival_w = np.array([[300, 310, inf]], np.int64)
    saved = R._NUMBA
    try:
        R._NUMBA = False
        painted, s_star, aw_sel, d_star, t_star, is_walk = R.select_planned_departafter(
            access_off, access_to, access_w, purewalk, arrival_w,
            dep_grid, cell_deps, 75)
    finally:
        R._NUMBA = saved
    assert int(painted[0]) == 5
    assert int(s_star[0]) == 0 and int(aw_sel[0]) == 60
    assert int(t_star[0]) == 310 and not bool(is_walk[0])
    assert int(d_star[0]) == 40
    assert int(d_star[0]) + int(aw_sel[0]) in dep_grid.tolist()


def test_select_planned_unpainted_walk_leaves_sentinels():
    """An over-cap walk-only cell remains unpainted and keeps its sentinels."""
    from core import raptor as R

    inf = int(R.INF)
    max_min = 75
    dep_grid = np.arange(0, 300, 60, dtype=np.int64)
    cell_deps = np.arange(0, 180, 60, dtype=np.int64)
    access_off = np.array([0, 0, 0], np.int64)
    access_to = np.zeros(0, np.int64)
    access_w = np.zeros(0, np.int64)
    purewalk = np.array([80 * 60, 10 * 60], np.int64)
    arrival_w = np.full((1, len(dep_grid)), inf, np.int64)
    saved = R._NUMBA
    try:
        R._NUMBA = False
        out = R.select_planned_departafter(
            access_off, access_to, access_w, purewalk, arrival_w,
            dep_grid, cell_deps, max_min)
    finally:
        R._NUMBA = saved
    painted, s_star, aw_sel, d_star, t_star, is_walk = out
    assert int(painted[0]) == -1 and not bool(is_walk[0])
    assert int(d_star[0]) == int(R.NEG) and int(t_star[0]) == -1 and int(s_star[0]) == -1
    assert int(painted[1]) == 10 and bool(is_walk[1])
    assert int(d_star[1]) == int(cell_deps[-1]) - 10 * 60
    assert int(aw_sel[1]) == 10 * 60
