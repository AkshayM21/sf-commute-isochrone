"""Focused checks for graph-baked transfer geometry served by the RAPTOR provider."""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from core import raptor as R, server_raptor as sr  # noqa: E402


class _Engine:
    def __init__(self, tr_off, tr_to, tr_time, path_off, path_points, fallback=None,
                 reverse_off=None, reverse_to=None, reverse_time=None):
        self.data = {
            "tr_forward_off": np.asarray(tr_off, dtype=np.int64),
            "tr_forward_to": np.asarray(tr_to, dtype=np.int32),
            "tr_off": np.asarray(reverse_off if reverse_off is not None else tr_off, dtype=np.int64),
            "tr_to": np.asarray(reverse_to if reverse_to is not None else tr_to, dtype=np.int32),
            "tr_time": np.asarray(reverse_time if reverse_time is not None else tr_time, dtype=np.int64),
        }
        self.transfer_path_off = np.asarray(path_off, dtype=np.int64)
        self.transfer_path_points = np.asarray(path_points, dtype=np.float64).reshape(-1, 2)
        self.transfer_path_fallback = np.asarray(
            fallback if fallback is not None else [False] * len(tr_to), dtype=bool)


def _engine():
    # 0 -> 1 and 2 -> 0 are present; 1 -> 0 is intentionally absent. The second path is an
    # explicit station pathway endpoint segment, not a street route.
    return _Engine(
        tr_off=[0, 1, 1, 2],
        tr_to=[1, 0],
        tr_time=[37, 91],
        path_off=[0, 3, 5],
        path_points=[[37.0, -122.0], [37.01, -122.01], [37.02, -122.02],
                     [38.0, -123.0], [38.1, -123.1]],
        fallback=[False, True],
        reverse_off=[0, 0, 1, 2], reverse_to=[0, 0], reverse_time=[37, 91],
    )


def test_directed_lookup_uses_timing_edge_and_preserves_path_order():
    engine = _engine()
    edge = sr._baked_transfer_edge(engine, 0, 1)
    assert edge == 0
    assert engine.data["tr_time"][0] == 37
    assert sr._baked_transfer_edge(engine, 1, 0) is None

    points, approx = sr._baked_transfer_geometry(engine, 0, 1)
    assert points == [[37.0, -122.0], [37.01, -122.01], [37.02, -122.02]]
    assert approx is False


def test_provider_returns_independent_copies_without_graph_routing(monkeypatch):
    engine = _engine()
    monkeypatch.setattr(sr, "_RAPTOR", engine)

    class _NoRoutingGraph:
        def path_tree(self, *args, **kwargs):  # pragma: no cover - failure proves no fallback
            raise AssertionError("transfer geometry must not route live")

    monkeypatch.setattr(sr, "_WG", _NoRoutingGraph())
    provider = sr._JourneyGeomProvider(37.7, -122.4)
    first, first_approx = provider.transfer(0, 1)
    second, second_approx = provider.transfer(0, 1)
    first[0][0] = -1.0
    assert second == [[37.0, -122.0], [37.01, -122.01], [37.02, -122.02]]
    assert first_approx is second_approx is False


def test_explicit_pathway_fallback_is_marked_approximate(monkeypatch):
    engine = _engine()
    monkeypatch.setattr(sr, "_RAPTOR", engine)
    points, approx = sr._JourneyGeomProvider(37.7, -122.4).transfer(2, 0)
    assert points == [[38.0, -123.0], [38.1, -123.1]]
    assert approx is True


def test_missing_duplicate_and_corrupt_edges_omit_geometry():
    engine = _engine()
    assert sr._baked_transfer_geometry(engine, 1, 0) is None

    duplicate = _Engine(
        tr_off=[0, 2, 2], tr_to=[1, 1], tr_time=[10, 11], path_off=[0, 2, 4],
        path_points=[[1, 2], [1, 3], [2, 2], [2, 3]])
    assert sr._baked_transfer_edge(duplicate, 0, 1) is None
    assert sr._baked_transfer_geometry(duplicate, 0, 1) is None

    corrupt = _Engine(
        tr_off=[0, 1], tr_to=[1], tr_time=[10], path_off=[0, 9],
        path_points=[[1, 2]])
    assert sr._baked_transfer_geometry(corrupt, 0, 1) is None


def test_one_way_forward_geometry_pairs_with_reverse_raptor_parent():
    engine = _engine()
    # Reverse RAPTOR starts at the egress target (1) and must discover source 0 through
    # the reverse CSR row, while geometry must still draw the stored 0 -> 1 forward path.
    data = {
        "n_stops": 2,
        "pat_nstops": np.zeros(0, np.int32), "pat_ntrips": np.zeros(0, np.int32),
        "pat_stop_off": np.zeros(1, np.int64), "pat_mat_off": np.zeros(1, np.int64),
        "pat_stops": np.zeros(0, np.int32), "pat_dep": np.zeros(0, np.int32),
        "pat_arr": np.zeros(0, np.int32), "ras_off": np.zeros(3, np.int64),
        "ras_pat": np.zeros(0, np.int32), "ras_pos": np.zeros(0, np.int32),
        "tr_off": engine.data["tr_off"], "tr_to": engine.data["tr_to"],
        "tr_time": engine.data["tr_time"],
    }
    traced = R.reverse_raptor_traced(
        data, np.asarray([1], np.int32), np.asarray([1000], np.int64),
        np.asarray([0], np.int64), max_rounds=1)
    assert traced["best"].tolist() == [963, 1000]
    assert traced["par_kind"].tolist() == [2, 0]
    assert traced["par_from"].tolist() == [1, -1]
    points, approx = sr._baked_transfer_geometry(engine, 0, 1)
    assert points[0] == [37.0, -122.0] and points[-1] == [37.02, -122.02]
    assert approx is False
