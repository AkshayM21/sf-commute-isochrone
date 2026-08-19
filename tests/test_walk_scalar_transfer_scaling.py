"""Focused walk-speed regression tests independent of the baked transit data."""
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from core import raptor_engine as engine_module


def _synthetic_engine():
    """Minimal engine shell: enough to exercise public routing branches without a GTFS bake."""
    engine = object.__new__(engine_module.RaptorEngine)
    engine.data = {"tr_time": np.array([20, 0], dtype=np.int32)}
    engine._walk_scaled_data = {}
    engine.access_off = np.array([0, 1], dtype=np.int64)
    engine.access_to = np.array([7], dtype=np.int32)
    engine.access_w = np.array([10], dtype=np.int64)
    engine.Tgrid = np.array([100, 160], dtype=np.int64)
    engine.Tgrid_planned = np.array([100, 160], dtype=np.int64)
    engine.dep_grid = np.array([100], dtype=np.int64)
    engine.cell_deps = np.array([100], dtype=np.int64)
    engine.dep_sec = 100
    engine.win_sec = 60
    engine.access_cap_min = 1
    engine.target_sec = 200
    engine.max_min = 75
    return engine


def test_legacy_paths_scale_transfer_footpaths_with_other_walk_legs(monkeypatch):
    """All legacy public paths must use one pace-scaled graph; scalar 1 remains the raw graph."""
    engine = _synthetic_engine()
    raw = engine.data
    scaled = engine._data_for_walk_scalar(1.5)
    assert scaled is not raw
    assert scaled["tr_time"].tolist() == [30, 0]
    assert engine._data_for_walk_scalar(1.0) is raw
    slow_dep_grid, slow_deadlines = engine._departafter_grids(1.5)
    assert slow_dep_grid[-1] >= engine.cell_deps[-1] + 90
    assert slow_deadlines[-1] >= slow_dep_grid[-1] + engine.max_min * 60

    reverse_calls = []

    def fake_reverse(egress_g, egress_w, deadlines, max_rounds=engine_module.MAX_ROUNDS, data=None):
        reverse_calls.append((data, np.asarray(egress_w).copy()))
        return "latest"

    engine._reverse = fake_reverse
    monkeypatch.setattr(engine_module.R, "stop_arrival_profile", lambda *_args: "arrival-profile")
    assembled = []

    def fake_departafter(_off, _to, access_w, purewalk, *_args, **_kwargs):
        assembled.append((np.asarray(access_w).copy(), np.asarray(purewalk).copy()))
        return np.array([[11, 12]], dtype=np.int32)

    monkeypatch.setattr(engine_module.R, "assemble_departafter", fake_departafter)
    args = (np.array([0, 1]), np.array([7]), np.array([10]), np.array([2]), np.array([20]),
            np.array([30]))
    engine.commute_for_access(*args, semantic="departafter", walk_scalar=1.5)
    assert reverse_calls[-1][0] is scaled
    assert reverse_calls[-1][1].tolist() == [30]       # egress: 20 * 1.5
    assert assembled[-1][0].tolist() == [15]           # access: 10 * 1.5
    assert assembled[-1][1].tolist() == [45]           # pure walk: 30 * 1.5

    arriveby_assembled = []

    def fake_arriveby(_off, _to, access_w, purewalk, *_args, **_kwargs):
        arriveby_assembled.append((np.asarray(access_w).copy(), np.asarray(purewalk).copy()))
        return np.array([[13, 14]], dtype=np.int32)

    monkeypatch.setattr(engine_module, "_assemble_arriveby_window", fake_arriveby)
    engine.commute_for_access(*args, semantic="arriveby", walk_scalar=1.5)
    assert reverse_calls[-1][0] is scaled
    assert reverse_calls[-1][1].tolist() == [30]
    assert arriveby_assembled[-1][0].tolist() == [15]
    assert arriveby_assembled[-1][1].tolist() == [45]

    planned_tree_data = []

    class FakePlannedTree:
        def __init__(self, data, *_args, **_kwargs):
            planned_tree_data.append(data)

        def commute(self):
            return np.array([22], dtype=np.int32)

    monkeypatch.setattr(engine_module.raptor_journey, "DepartAfterJourneyTree", FakePlannedTree)
    engine.commute_for_access(*args, semantic="planned", walk_scalar=1.5)
    assert reverse_calls[-1][0] is scaled
    assert reverse_calls[-1][1].tolist() == [30]
    assert planned_tree_data[-1] is scaled

    engine.commute_for_access(*args, semantic="departafter", walk_scalar=1.0)
    assert reverse_calls[-1][0] is raw, "explicit reference scalar must preserve legacy graph identity"

    traced = []
    monkeypatch.setattr(
        engine_module.R, "reverse_raptor_traced_fast",
        lambda data, _eg, _target, _ew, **_kwargs: traced.append(data) or "trace")
    tree_data = []

    class FakeJourneyTree:
        def __init__(self, data, *_args, **_kwargs):
            tree_data.append(data)

    monkeypatch.setattr(engine_module.raptor_journey, "JourneyTree", FakeJourneyTree)
    engine.journey_tree(np.array([2]), np.array([20]), np.array([30]), walk_scalar=1.5)
    assert traced[-1] is scaled
    assert tree_data[-1] is scaled

    departafter_tree_data = []

    class FakeDepartAfterJourneyTree:
        def __init__(self, data, *_args, **_kwargs):
            departafter_tree_data.append(data)

    monkeypatch.setattr(engine_module.raptor_journey, "DepartAfterJourneyTree", FakeDepartAfterJourneyTree)
    engine.journey_tree_departafter(np.array([2]), np.array([20]), np.array([30]),
                                    walk_scalar=1.5, planned=False)
    assert reverse_calls[-1][0] is scaled
    assert departafter_tree_data[-1] is scaled

    engine.journey_tree_departafter(np.array([2]), np.array([20]), np.array([30]),
                                    walk_scalar=1.5, planned=True)
    assert reverse_calls[-1][0] is scaled
    assert departafter_tree_data[-1] is scaled
