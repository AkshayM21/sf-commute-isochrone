"""Focused tests for the offline graph-backed transfer bake."""

import gc
from types import MappingProxyType, SimpleNamespace
import weakref

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from core.graph_transfers import MAX_CANDIDATE_RADIUS_M, bake_graph_transfers
from core.transfer_rules import (
    PathwayEdge,
    ScopedTransferRule,
    StopKey,
    TransferRuleSet,
)
from core.walk import WalkGraph


def _walk_graph(tmp_path, *, disconnected=False):
    n = 10
    lon = -122.5 + np.arange(n, dtype=np.float64) * (0.002 if disconnected else 0.001)
    lat = np.full(n, 37.75, dtype=np.float64)
    if disconnected:
        rows, cols, weights = [0, 1], [1, 0], [70.0, 130.0]
    else:
        rows, cols, weights = [], [], []
        for index in range(n - 1):
            rows.extend((index, index + 1))
            cols.extend((index + 1, index))
            weights.extend((70.0, 130.0))
    graph = csr_matrix((weights, (rows, cols)), shape=(n, n), dtype=np.float64)
    path = tmp_path / ("disconnected.npz" if disconnected else "line.npz")
    np.savez(
        path,
        node_lon=lon,
        node_lat=lat,
        node_elev=np.zeros(n, dtype=np.float32),
        indptr=graph.indptr.astype(np.int32),
        indices=graph.indices.astype(np.int32),
        w_ref=graph.data,
        w_flat=graph.data,
        walk_ref_kmh=np.array(4.8),
    )
    return WalkGraph.load(path)


def _stops(wg, indexes=(0, 1, 2, 3)):
    indexes = np.asarray(indexes, dtype=np.int64)
    return (
        tuple(StopKey("muni", str(int(index))) for index in indexes),
        wg.lon[indexes],
        wg.lat[indexes],
    )


def _rules(*, mins=None, prohibited=(), pathways=(), scoped=()):
    mins = mins or {}
    return TransferRuleSet(
        feed="muni",
        unconditional_rules=(),
        route_scoped_rules=tuple(scoped),
        pathway_edges=tuple(pathways),
        min_transfer_seconds=MappingProxyType(dict(mins)),
        prohibited_pairs=frozenset(prohibited),
    )


def _edge(bake, source, target):
    start, end = int(bake.tr_off[source]), int(bake.tr_off[source + 1])
    matches = np.flatnonzero(bake.tr_to[start:end] == target)
    assert len(matches) == 1
    return start + int(matches[0])


def test_ordinary_edges_use_directed_route_result_for_time_and_path(tmp_path):
    wg = _walk_graph(tmp_path)
    keys, lon, lat = _stops(wg, (0, 9))
    bake = bake_graph_transfers(keys, lon, lat, wg, radius_m=1000.0)

    forward = _edge(bake, 0, 1)
    reverse = _edge(bake, 1, 0)
    expected_forward = wg.path_tree((lon[0], lat[0]), 1800).route_result((lon[1], lat[1]))
    expected_reverse = wg.path_tree((lon[1], lat[1]), 1800).route_result((lon[0], lat[0]))

    assert bake.tr_walk_time[forward] == expected_forward.seconds
    assert bake.tr_walk_time[reverse] == expected_reverse.seconds
    np.testing.assert_array_equal(bake.path_points(forward), np.asarray(expected_forward.points))
    np.testing.assert_array_equal(bake.path_points(reverse), np.asarray(expected_reverse.points))
    assert bake.tr_walk_time[reverse] > bake.tr_walk_time[forward]


def test_unreachable_ordinary_candidates_are_dropped(tmp_path):
    wg = _walk_graph(tmp_path, disconnected=True)
    keys, lon, lat = _stops(wg, (0, 1, 9))
    # The target's K nearest nodes do not overlap the source's K nearest nodes;
    # the explicit row considers 0 -> 9, but the disconnected graph rejects it.
    pair = (keys[0], keys[2])
    bake = bake_graph_transfers(
        keys, lon, lat, wg, _rules(mins={pair: 20}), radius_m=1800.0)

    assert bake.edge_count == 2  # only the reachable 0 <-> 1 ordinary pair
    assert set(bake.tr_to.tolist()) == {1, 0}
    assert not np.any(bake.tr_to == 2)


def test_unconditional_prohibition_drops_only_its_direction(tmp_path):
    wg = _walk_graph(tmp_path)
    keys, lon, lat = _stops(wg, (0, 1))
    prohibited = (keys[0], keys[1])
    bake = bake_graph_transfers(
        keys, lon, lat, wg, _rules(prohibited={prohibited}), radius_m=250.0)

    assert bake.edge_count == 1
    assert bake.tr_to.tolist() == [0]
    assert int(bake.tr_off[0]) == int(bake.tr_off[1])


def test_route_scoped_prohibition_is_not_promoted_to_global_ban(tmp_path):
    wg = _walk_graph(tmp_path)
    keys, lon, lat = _stops(wg, (0, 1))
    scoped = ScopedTransferRule(
        keys[0], keys[1], from_route_id="R1", to_route_id="R2", prohibited=True)
    bake = bake_graph_transfers(keys, lon, lat, wg, _rules(scoped=(scoped,)), radius_m=250.0)

    assert bake.edge_count == 2


def test_scoped_only_pair_keeps_physical_path_metadata_without_global_edge(tmp_path):
    wg = _walk_graph(tmp_path)
    keys, lon, lat = _stops(wg, (0, 1))
    scoped = ScopedTransferRule(
        keys[0], keys[1], from_route_id="R1", to_route_id="R2", min_transfer_seconds=17)
    bake = bake_graph_transfers(keys, lon, lat, wg, _rules(scoped=(scoped,)), radius_m=0.0)

    assert bake.edge_count == 0
    assert len(bake.scoped_rules) == 1
    record = bake.scoped_rules[0]
    assert record.physical_seconds > 0.0
    assert record.path_points[0] == (lat[0], lon[0])
    assert record.path_points[-1] == (lat[1], lon[1])


def test_fixed_minimum_is_separate_from_physical_walk_time(tmp_path):
    wg = _walk_graph(tmp_path)
    keys, lon, lat = _stops(wg, (0, 1))
    pair = (keys[0], keys[1])
    bake = bake_graph_transfers(
        keys, lon, lat, wg, _rules(mins={pair: 500}), radius_m=250.0)
    edge = _edge(bake, 0, 1)

    assert bake.tr_walk_time[edge] < bake.tr_min_time[edge]
    assert bake.tr_min_time[edge] == 500.0
    assert bake.tr_time[edge] == 500.0
    assert np.asarray(bake.path_points(edge)).shape[0] >= 1


def test_directed_pathway_uses_authoritative_time_and_direction(tmp_path):
    wg = _walk_graph(tmp_path)
    keys, lon, lat = _stops(wg, (0, 9))
    pathway = PathwayEdge(
        "p1", keys[0], keys[1], 33, "1", None,
        reversed_from_bidirectional=False)
    # radius=0 proves that the pathway, not ordinary radius discovery, supplies
    # this edge.  Its directed reverse is intentionally absent.
    bake = bake_graph_transfers(
        keys, lon, lat, wg, _rules(pathways=(pathway,)), radius_m=0.0)

    assert bake.edge_count == 1
    edge = _edge(bake, 0, 1)
    assert bake.tr_walk_time[edge] == 33.0
    assert bake.tr_time[edge] == 33.0
    assert bake.tr_path_fallback[edge]
    np.testing.assert_array_equal(
        bake.path_points(edge),
        np.asarray([[lat[0], lon[0]], [lat[1], lon[1]]], dtype=np.float64),
    )
    assert bake.tr_pathway_metadata[edge] == (pathway,)
    assert int(bake.tr_off[1]) == int(bake.tr_off[2])


def test_unreachable_pathway_keeps_explicit_edge_with_safe_endpoint_geometry(tmp_path):
    wg = _walk_graph(tmp_path, disconnected=True)
    keys, lon, lat = _stops(wg, (0, 2))
    pathway = PathwayEdge(
        "elevator", keys[0], keys[1], 45, "elevator", None,
        reversed_from_bidirectional=False)
    bake = bake_graph_transfers(keys, lon, lat, wg, _rules(pathways=(pathway,)), radius_m=0.0)

    assert bake.edge_count == 1
    edge = _edge(bake, 0, 1)
    assert bake.tr_walk_time[edge] == 45.0
    assert bake.tr_path_fallback[edge]
    np.testing.assert_array_equal(
        bake.path_points(edge),
        np.asarray([[lat[0], lon[0]], [lat[1], lon[1]]], dtype=np.float64),
    )


def test_untimed_pathway_is_preserved_but_does_not_invent_a_costed_edge(tmp_path):
    wg = _walk_graph(tmp_path)
    keys, lon, lat = _stops(wg, (0, 9))
    untimed = PathwayEdge(
        "untimed", keys[0], keys[1], None, "stairs", 25.0,
        reversed_from_bidirectional=False)

    bake = bake_graph_transfers(
        keys, lon, lat, wg, _rules(pathways=(untimed,)), radius_m=0.0)

    assert bake.edge_count == 0
    assert len(bake.pathway_records) == 1
    assert bake.pathway_records[0].edge == untimed


def test_outputs_are_sorted_and_path_offsets_are_parallel(tmp_path):
    wg = _walk_graph(tmp_path)
    keys, lon, lat = _stops(wg, (0, 1, 2))
    bake = bake_graph_transfers(keys, lon, lat, wg, radius_m=250.0)

    assert np.all(bake.tr_off[1:] >= bake.tr_off[:-1])
    assert np.all(bake.tr_path_off[1:] >= bake.tr_path_off[:-1])
    for source in range(len(keys)):
        start, end = bake.tr_off[source:source + 2]
        assert np.all(bake.tr_to[start:end][:-1] <= bake.tr_to[start:end][1:])
    assert len(bake.tr_path_fallback) == len(bake.tr_to)
    assert int(bake.tr_path_off[-1]) == len(bake.tr_path_points)


def test_one_path_tree_is_live_at_a_time_and_sources_are_not_cached():
    class SpyTree:
        def __init__(self, owner, root):
            self.owner = owner
            self.root = root
            owner.live.add(self)
            owner.max_live = max(owner.max_live, len(owner.live))

        def route_result(self, target):
            return SimpleNamespace(
                seconds=12.0,
                points=[[self.root[1], self.root[0]], [target[1], target[0]]],
            )

    class SpyGraph:
        mlon = 88_000.0
        mlat = 111_320.0

        def __init__(self):
            self.live = weakref.WeakSet()
            self.max_live = 0
            self.built_sources = []

        def supports_point(self, _lon, _lat):
            return True

        def path_tree(self, root, _cap):
            gc.collect()
            # The previous source tree must have been released before this call.
            assert not self.live
            self.built_sources.append(root)
            return SpyTree(self, root)

    graph = SpyGraph()
    keys = tuple(StopKey("muni", str(index)) for index in range(3))
    lon = np.asarray([-122.500, -122.499, -122.498])
    lat = np.asarray([37.750, 37.750, 37.750])
    bake = bake_graph_transfers(keys, lon, lat, graph, radius_m=250.0)
    gc.collect()

    assert bake.edge_count == 6
    assert graph.max_live == 1
    assert len(graph.built_sources) == 3
    assert not graph.live


def test_scoped_rules_and_all_selected_pair_pathway_metadata_are_preserved_deterministically(
        tmp_path):
    wg = _walk_graph(tmp_path)
    keys, lon, lat = _stops(wg, (0, 9))
    slow = PathwayEdge(
        "slow", keys[0], keys[1], 40, "stairs", 20.0,
        reversed_from_bidirectional=False)
    fast = PathwayEdge(
        "fast", keys[0], keys[1], 25, "elevator", 10.0,
        reversed_from_bidirectional=False)
    first = ScopedTransferRule(
        keys[1], keys[0], from_route_id="R2", to_route_id="R1", prohibited=True)
    second = ScopedTransferRule(
        keys[0], keys[1], from_trip_id="T1", to_trip_id="T2",
        min_transfer_seconds=35)
    rule_a = _rules(pathways=(slow,), scoped=(first,))
    rule_b = _rules(pathways=(fast, slow), scoped=(second, first))

    forward = bake_graph_transfers(
        keys, lon, lat, wg, (rule_a, rule_b), radius_m=0.0)
    reversed_input = bake_graph_transfers(
        keys, lon, lat, wg, (rule_b, rule_a), radius_m=0.0)
    edge = _edge(forward, 0, 1)

    assert forward.tr_walk_time[edge] == 25.0
    assert forward.tr_pathway_metadata[edge] == (fast, slow)
    assert forward.tr_pathway_metadata == reversed_input.tr_pathway_metadata
    assert forward.scoped_rules == reversed_input.scoped_rules
    assert forward.pathway_records == reversed_input.pathway_records
    assert [record.edge for record in forward.pathway_records] == [fast, slow]
    assert [(record.source, record.target) for record in forward.scoped_rules] == [(0, 1), (1, 0)]
    assert [record.rule for record in forward.scoped_rules] == [second, first]


@pytest.mark.parametrize("radius", [-1.0, np.nan, np.inf, MAX_CANDIDATE_RADIUS_M + 0.1])
def test_candidate_radius_rejects_invalid_or_impractically_large_values(tmp_path, radius):
    wg = _walk_graph(tmp_path)
    keys, lon, lat = _stops(wg, (0, 1))

    with pytest.raises(ValueError, match="radius_m"):
        bake_graph_transfers(keys, lon, lat, wg, radius_m=radius)


def test_out_of_range_route_path_coordinates_are_rejected():
    class BadTree:
        def route_result(self, _target):
            return SimpleNamespace(seconds=5.0, points=[[91.0, -122.5]])

    class BadGraph:
        mlon = 88_000.0
        mlat = 111_320.0

        def supports_point(self, _lon, _lat):
            return True

        def path_tree(self, _root, _cap):
            return BadTree()

    keys = (StopKey("muni", "A"), StopKey("muni", "B"))
    with pytest.raises(ValueError, match="out-of-range"):
        bake_graph_transfers(
            keys,
            np.asarray([-122.500, -122.499]),
            np.asarray([37.750, 37.750]),
            BadGraph(),
            radius_m=250.0,
        )


def test_packed_arrays_are_read_only_but_returned_path_copy_is_mutable(tmp_path):
    wg = _walk_graph(tmp_path)
    keys, lon, lat = _stops(wg, (0, 1))
    bake = bake_graph_transfers(keys, lon, lat, wg, radius_m=250.0)
    arrays = (
        bake.tr_off,
        bake.tr_to,
        bake.tr_time,
        bake.tr_walk_time,
        bake.tr_min_time,
        bake.tr_path_off,
        bake.tr_path_points,
        bake.tr_path_fallback,
    )

    assert all(not array.flags.writeable for array in arrays)
    with pytest.raises(ValueError):
        bake.tr_to[0] = 99
    copy = bake.path_points(0)
    copy[0, 0] = 0.0
    assert bake.path_points(0)[0, 0] != 0.0


def test_explicit_non_pathway_transfer_outside_supported_graph_is_dropped(tmp_path):
    wg = _walk_graph(tmp_path)
    keys = (StopKey("muni", "outside-a"), StopKey("muni", "outside-b"))
    lon = np.asarray([-122.270, -122.269])
    lat = np.asarray([37.800, 37.800])
    pair = (keys[0], keys[1])

    assert not wg.supports_point(lon[0], lat[0])
    bake = bake_graph_transfers(
        keys, lon, lat, wg, _rules(mins={pair: 45}), radius_m=250.0)

    assert bake.edge_count == 0


def test_invalid_stop_coordinates_keep_gid_alignment_but_drop_edges(tmp_path):
    wg = _walk_graph(tmp_path)
    keys = tuple(StopKey("muni", str(index)) for index in range(2))
    lon = np.asarray([-122.500, np.nan])
    lat = np.asarray([37.750, 37.750])
    bake = bake_graph_transfers(keys, lon, lat, wg, radius_m=250.0)
    assert bake.tr_forward_off.shape == (3,)
    assert bake.edge_count == 0
    assert bake.tr_reverse_off.shape == (3,)
