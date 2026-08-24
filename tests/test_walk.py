"""Hill-aware graph-native walk-router tests.

The load-bearing physics are directional walking costs, graph snapping, and exact
path-tree/geometry reuse. Small fixtures cover these contracts without external data.
"""
import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
WALK_NPZ = os.path.join(_REPO, "data", "walk_graph.npz")


@pytest.fixture
def tiny_wg(tmp_path):
    """Small deterministic directed graph for unit tests that do not need the SF data bake."""
    n = 10
    lon = -122.5 + np.arange(n, dtype=np.float64) * 0.001
    lat = np.full(n, 37.75, dtype=np.float64)
    rows, cols, weights = [], [], []
    for i in range(n - 1):
        rows.extend((i, i + 1))
        cols.extend((i + 1, i))
        weights.extend((70.0, 130.0))             # intentionally directional
    graph = csr_matrix((weights, (rows, cols)), shape=(n, n))
    path = tmp_path / "tiny_walk_graph.npz"
    np.savez(
        path,
        node_lon=lon,
        node_lat=lat,
        node_elev=np.zeros(n, dtype=np.float32),
        indptr=graph.indptr.astype(np.int32),
        indices=graph.indices.astype(np.int32),
        w_ref=graph.data.astype(np.float64),
        w_flat=graph.data.astype(np.float64),
        walk_ref_kmh=np.array(4.8),
    )
    from core import walk
    return walk.WalkGraph.load(path)


@pytest.fixture
def disconnected_wg(tmp_path):
    """Four-node graph with one isolated target for unreachable PathTree results."""
    from core import walk

    n = 4
    lon = -122.5 + np.arange(n, dtype=np.float64) * 0.03
    lat = np.full(n, 37.75, dtype=np.float64)
    graph = csr_matrix(
        ([70.0, 130.0], ([0, 1], [1, 0])), shape=(n, n), dtype=np.float64)
    path = tmp_path / "disconnected_walk_graph.npz"
    np.savez(
        path,
        node_lon=lon,
        node_lat=lat,
        node_elev=np.zeros(n, dtype=np.float32),
        indptr=graph.indptr.astype(np.int32),
        indices=graph.indices.astype(np.int32),
        w_ref=graph.data.astype(np.float64),
        w_flat=graph.data.astype(np.float64),
        walk_ref_kmh=np.array(4.8),
    )
    return walk.WalkGraph.load(path)


@pytest.fixture
def route_wg(tmp_path):
    """Line graph with wide node spacing so graph paths beat direct snap connectors."""
    from core import walk

    n = 10
    lon = -122.5 + np.arange(n, dtype=np.float64) * 0.01
    lat = np.full(n, 37.75, dtype=np.float64)
    rows, cols, weights = [], [], []
    for i in range(n - 1):
        rows.extend((i, i + 1))
        cols.extend((i + 1, i))
        weights.extend((70.0, 130.0))
    graph = csr_matrix((weights, (rows, cols)), shape=(n, n))
    path = tmp_path / "route_walk_graph.npz"
    np.savez(
        path,
        node_lon=lon,
        node_lat=lat,
        node_elev=np.zeros(n, dtype=np.float32),
        indptr=graph.indptr.astype(np.int32),
        indices=graph.indices.astype(np.int32),
        w_ref=graph.data.astype(np.float64),
        w_flat=graph.data.astype(np.float64),
        walk_ref_kmh=np.array(4.8),
    )
    return walk.WalkGraph.load(path)


def test_walk_graph_support_policy_uses_nearest_connector_boundary(tiny_wg):
    lon = float(tiny_wg.lon[0])
    lat = float(tiny_wg.lat[0])
    max_m = tiny_wg.MAX_CONNECTOR_M

    assert max_m == 300.0
    assert tiny_wg.nearest_distance_m(lon, lat) == pytest.approx(0.0)
    assert tiny_wg.supports_point(lon, lat, max_connector_m=0.0)

    # Query west of the first node so no other tiny-fixture node can make the boundary nearer.
    inside_lon = lon - (max_m - 0.1) / tiny_wg.mlon
    outside_lon = lon - (max_m + 0.1) / tiny_wg.mlon
    assert tiny_wg.nearest_distance_m(inside_lon, lat) == pytest.approx(max_m - 0.1, abs=1e-6)
    assert tiny_wg.supports_point(inside_lon, lat)
    assert not tiny_wg.supports_point(outside_lon, lat)


@pytest.mark.parametrize(
    "lon, lat",
    [(None, 37.75), ("not-a-number", 37.75), (np.nan, 37.75),
     (-122.5, np.inf), (181.0, 37.75), (-122.5, -91.0)],
)
def test_walk_graph_support_policy_rejects_malformed_or_nonfinite_points(tiny_wg, lon, lat):
    assert not tiny_wg.supports_point(lon, lat)
    assert math.isinf(tiny_wg.nearest_distance_m(lon, lat))


def test_walk_graph_support_policy_rejects_invalid_thresholds(tiny_wg):
    lon = float(tiny_wg.lon[0])
    lat = float(tiny_wg.lat[0])
    for threshold in (-1.0, np.nan, np.inf, "bad"):
        assert not tiny_wg.supports_point(lon, lat, threshold)


@pytest.mark.parametrize("reverse", [False, True])
def test_path_tree_distances_exactly_match_one_to_many_at_caller_caps(tiny_wg, reverse):
    """One max-cap PathTree must reproduce the old separately-capped Dijkstras byte-for-byte."""
    root = (float(tiny_wg.lon[0]), float(tiny_wg.lat[0]))
    picks = np.array([0, 2, 4, 7, 9])
    nodes, conn = tiny_wg.snap(np.column_stack((tiny_wg.lon[picks], tiny_wg.lat[picks])))
    tree = tiny_wg.path_tree(root, 1200, reverse=reverse)

    for cap in (300, 650, 1200):
        expected = tiny_wg.one_to_many(root, nodes, conn, cap, reverse=reverse)
        actual = tree.distances_to(nodes, conn, cap)
        np.testing.assert_array_equal(actual, expected)

    with pytest.raises(ValueError, match="exceeds PathTree build cap"):
        tree.distances_to(nodes, conn, 1201)
    with pytest.raises(ValueError, match=r"matching \[n, k\] shapes"):
        tree.distances_to(nodes[:, 0], conn[:, 0], 300)


def test_path_tree_route_result_uses_one_argmin_for_duration_and_geometry(route_wg):
    root = (float(route_wg.lon[0]), float(route_wg.lat[0]))
    target = (float(route_wg.lon[3]), float(route_wg.lat[3]))
    nodes, conn = route_wg.snap([target])
    expected = route_wg.one_to_many(root, nodes, conn, 1200)[0]
    tree = route_wg.path_tree(root, 1200)

    result = tree.route_result(target)

    assert result is not None
    assert result.seconds == pytest.approx(expected)
    assert result.points == tree.path_points(target)
    assert result.points[0] == [round(float(route_wg.lat[0]), 6), round(float(route_wg.lon[0]), 6)]
    assert result.points[-1] == [round(float(route_wg.lat[3]), 6), round(float(route_wg.lon[3]), 6)]

    # Each call owns its point list; a rendered response cannot mutate the next result.
    again = tree.route_result(target)
    result.points.append([0.0, 0.0])
    assert again.points == tree.path_points(target)


def test_path_tree_route_result_preserves_directed_forward_and_reverse_semantics(route_wg):
    node0 = (float(route_wg.lon[0]), float(route_wg.lat[0]))
    node2 = (float(route_wg.lon[2]), float(route_wg.lat[2]))

    forward = route_wg.path_tree(node0, 1200).route_result(node2)
    reverse = route_wg.path_tree(node0, 1200, reverse=True).route_result(node2)

    assert forward is not None and reverse is not None
    assert forward.seconds == pytest.approx(140.0)
    assert reverse.seconds == pytest.approx(260.0)
    assert forward.points == [
        [round(float(route_wg.lat[i]), 6), round(float(route_wg.lon[i]), 6)]
        for i in (0, 1, 2)
    ]
    assert reverse.points == [
        [round(float(route_wg.lat[i]), 6), round(float(route_wg.lon[i]), 6)]
        for i in (2, 1, 0)
    ]


def test_path_tree_route_result_includes_exact_connectors_in_walking_order(route_wg):
    source = (float(route_wg.lon[0] - 0.00013), float(route_wg.lat[0] + 0.00011))
    target = (float(route_wg.lon[3] + 0.00017), float(route_wg.lat[3] - 0.00009))
    source_point = [round(source[1], 6), round(source[0], 6)]
    target_point = [round(target[1], 6), round(target[0], 6)]
    source_node = [round(float(route_wg.lat[0]), 6), round(float(route_wg.lon[0]), 6)]
    target_node = [round(float(route_wg.lat[3]), 6), round(float(route_wg.lon[3]), 6)]

    forward = route_wg.path_tree(source, 1200).route_result(target)
    reverse = route_wg.path_tree(target, 1200, reverse=True).route_result(source)

    assert forward is not None and reverse is not None
    assert forward.seconds == pytest.approx(reverse.seconds)
    assert forward.points == reverse.points
    assert forward.points[:2] == [source_point, source_node]
    assert forward.points[-2:] == [target_node, target_point]

    # Exact endpoints are added only when they differ from the chosen graph nodes.
    on_node = route_wg.path_tree(
        (float(route_wg.lon[0]), float(route_wg.lat[0])), 1200
    ).route_result((float(route_wg.lon[3]), float(route_wg.lat[3])))
    assert on_node.points[0] == source_node
    assert on_node.points[-1] == target_node
    assert on_node.points.count(source_node) == 1
    assert on_node.points.count(target_node) == 1


@pytest.mark.parametrize(
    "target",
    [(np.nan, 37.75), (-122.5, np.inf), (181.0, 37.75), (-122.5, -91.0), ("bad", 0)],
)
def test_path_tree_route_result_rejects_invalid_wgs84_target(route_wg, target):
    root = (float(route_wg.lon[0]), float(route_wg.lat[0]))
    assert route_wg.path_tree(root, 1200).route_result(target) is None


def test_path_tree_route_result_rejects_invalid_wgs84_root(route_wg):
    invalid_root = (181.0, float(route_wg.lat[0]))
    target = (float(route_wg.lon[2]), float(route_wg.lat[2]))
    assert route_wg.path_tree(invalid_root, 1e9).route_result(target) is None


def test_path_tree_route_result_returns_none_for_unreachable_target(disconnected_wg):
    root = (float(disconnected_wg.lon[0]), float(disconnected_wg.lat[0]))
    target = (float(disconnected_wg.lon[2]), float(disconnected_wg.lat[2]))
    tree = disconnected_wg.path_tree(root, 1200)

    assert tree.route_result(target) is None


def _install_tiny_server_walk(monkeypatch, wg, stop_nodes, stop_conn, cell_nodes, cell_conn):
    from core import server_raptor as sr

    monkeypatch.setattr(sr, "_WG", wg)
    monkeypatch.setattr(sr, "_WG_STOP_NODES", stop_nodes)
    monkeypatch.setattr(sr, "_WG_STOP_CONN", stop_conn)
    monkeypatch.setattr(sr, "_WG_CELL_NODES", cell_nodes)
    monkeypatch.setattr(sr, "_WG_CELL_CONN", cell_conn)
    monkeypatch.setattr(sr, "_WG_STOP_GIDS", np.arange(len(stop_nodes), dtype=np.int32) + 100)
    monkeypatch.setattr(sr, "_RAPTOR", SimpleNamespace(access_cap_min=6))
    # Isolate these tests from the live module caches and from every other server test.
    monkeypatch.setattr(sr, "_RAPTOR_EGRESS_CACHE", sr.BoundedLRU(4))
    monkeypatch.setattr(sr, "_WALKPATH_TREE_CACHE", sr.BoundedLRU(4))
    monkeypatch.setattr(sr, "_WALKPATH_INFLIGHT", {})
    monkeypatch.setattr(sr, "_CELL_WALKPATH_TREE_CACHE", sr.BoundedLRU(2))
    monkeypatch.setattr(sr, "_CELL_WALKPATH_INFLIGHT", {})
    return sr


def test_raptor_walk_arrays_and_geometry_reuse_one_reverse_tree(monkeypatch, tiny_wg):
    stop_ix = np.array([2, 5, 8, 9])
    cell_ix = np.array([0, 3, 6, 9])
    stop_nodes, stop_conn = tiny_wg.snap(
        np.column_stack((tiny_wg.lon[stop_ix], tiny_wg.lat[stop_ix])))
    cell_nodes, cell_conn = tiny_wg.snap(
        np.column_stack((tiny_wg.lon[cell_ix], tiny_wg.lat[cell_ix])))
    sr = _install_tiny_server_walk(
        monkeypatch, tiny_wg, stop_nodes, stop_conn, cell_nodes, cell_conn)
    lat, lon = float(tiny_wg.lat[0]), float(tiny_wg.lon[0])

    # Capture the old two-one_to_many semantics before counting PathTree construction.
    old_eg = tiny_wg.one_to_many((lon, lat), stop_nodes, stop_conn, 360, reverse=True)
    old_pw = tiny_wg.one_to_many(
        (lon, lat), cell_nodes, cell_conn, sr.config.MAX_MIN * 60, reverse=True)
    original_path_tree = tiny_wg.path_tree
    calls = []

    def counted_path_tree(*args, **kwargs):
        calls.append((args, kwargs))
        return original_path_tree(*args, **kwargs)

    monkeypatch.setattr(tiny_wg, "path_tree", counted_path_tree)
    got = sr.raptor_egress_purewalk(lat, lon)

    finite = np.isfinite(old_eg)
    np.testing.assert_array_equal(got[0], sr._WG_STOP_GIDS[finite].astype(np.int32))
    np.testing.assert_array_equal(got[1], np.rint(old_eg[finite]).astype(np.int64))
    np.testing.assert_array_equal(
        got[2], np.where(np.isfinite(old_pw), np.rint(old_pw), -1).astype(np.int64))
    assert len(calls) == 1
    cached_tree = sr._WALKPATH_TREE_CACHE.get(sr.coarse_key(lat, lon))
    assert cached_tree is not None
    assert sr._JourneyGeomProvider(lat, lon)._w_tree() is cached_tree
    assert sr.raptor_egress_purewalk(lat, lon) is got
    assert len(calls) == 1


def test_transfer_geometry_uses_baked_directed_path_and_returns_isolated_output(monkeypatch):
    """Transfer timing and drawing use one directed baked edge; no graph routing is available."""
    from core import server_raptor as sr

    source, target = 2, 5
    engine = SimpleNamespace(
        data={
            # Geometry uses the forward source -> target view; RAPTOR's timing
            # uses the reverse target -> source row for the same one-way edge.
            "tr_forward_off": np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
            "tr_forward_to": np.array([target], dtype=np.int32),
            "tr_off": np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.int64),
            "tr_to": np.array([source], dtype=np.int32),
            "tr_time": np.array([42], dtype=np.int64),
        },
        transfer_path_off=np.array([0, 3], dtype=np.int64),
        transfer_path_points=np.array([[37.0, -122.0], [37.01, -122.01],
                                       [37.02, -122.02]], dtype=np.float64),
        transfer_path_fallback=np.array([False]),
    )
    monkeypatch.setattr(sr, "_WG", object())
    monkeypatch.setattr(
        sr, "_RAPTOR", engine)
    provider = sr._JourneyGeomProvider(37.77, -122.42)
    first = provider.transfer(source, target)
    assert first == ([[37.0, -122.0], [37.01, -122.01], [37.02, -122.02]], False)
    assert engine.data["tr_time"][sr._baked_transfer_edge(engine, source, target)] == 42
    first[0].append([0.0, 0.0])
    second = provider.transfer(source, target)
    assert second == ([[37.0, -122.0], [37.01, -122.01], [37.02, -122.02]], False)
    assert provider.transfer(target, source) == ([], False)


def _install_cell_access_provider(monkeypatch, tiny_wg):
    """Install just enough server state to draw deterministic cell->stop access paths."""
    from core import server_raptor as sr

    stop_lat = np.asarray(tiny_wg.lat, dtype=float)
    stop_lon = np.asarray(tiny_wg.lon, dtype=float)
    monkeypatch.setattr(sr, "_WG", tiny_wg)
    monkeypatch.setattr(
        sr, "_RAPTOR", SimpleNamespace(
            access_cap_min=6,
            cell_ids=["cell-0", "cell-1"],
            data={"stop_lat": stop_lat, "stop_lon": stop_lon}))
    monkeypatch.setattr(sr, "ORIGIN_LL", {
        "cell-0": (float(tiny_wg.lat[0]), float(tiny_wg.lon[0])),
        "cell-1": (float(tiny_wg.lat[1]), float(tiny_wg.lon[1])),
    })
    monkeypatch.setattr(
        sr, "_CELL_WALKPATH_TREE_CACHE",
        sr.BoundedLRU(2, maxbytes=512 * 1024, weight_fn=sr._walkpath_tree_weight))
    monkeypatch.setattr(sr, "_CELL_WALKPATH_INFLIGHT", {})
    return sr


def test_cell_access_paths_reuse_hover_tree_across_providers_without_sharing_output(
        monkeypatch, tiny_wg):
    """A fresh pin provider must reuse the exact hover tree, not merely an equal route."""
    sr = _install_cell_access_provider(monkeypatch, tiny_wg)
    original_path_tree = tiny_wg.path_tree
    calls = []

    def counted_path_tree(*args, **kwargs):
        calls.append((args, kwargs))
        return original_path_tree(*args, **kwargs)

    monkeypatch.setattr(tiny_wg, "path_tree", counted_path_tree)
    first = sr._JourneyGeomProvider(37.77, -122.42).access(0, 5)
    expected_points = first[0][:]
    # The HTTP response owns its mutable lists.  Mutating one rendered result cannot poison the
    # cached predecessor tree or the next pin's exact response.
    first[0].append([0.0, 0.0])
    second = sr._JourneyGeomProvider(37.77, -122.42).access(0, 5)

    assert second == (expected_points, False)
    assert len(calls) == 1
    cache = sr._CELL_WALKPATH_TREE_CACHE
    assert len(cache) == 1
    assert 0 < cache.nbytes <= cache.maxbytes


def test_cell_access_tree_cache_is_graph_scoped_and_count_bounded(monkeypatch, tiny_wg):
    sr = _install_cell_access_provider(monkeypatch, tiny_wg)
    # One retained root makes eviction observable without relying on timing or a large graph.
    cache = sr.BoundedLRU(1, maxbytes=512 * 1024, weight_fn=sr._walkpath_tree_weight)
    monkeypatch.setattr(sr, "_CELL_WALKPATH_TREE_CACHE", cache)
    first = sr._cell_walkpath_tree(0, sr.ORIGIN_LL["cell-0"])
    second = sr._cell_walkpath_tree(1, sr.ORIGIN_LL["cell-1"])
    assert first is not second
    assert len(cache) == 1
    assert cache.get((id(tiny_wg), 0, 720)) is None
    assert cache.get((id(tiny_wg), 1, 720)) is second

    # A graph re-init may reuse cell index 0.  Its identity-qualified key must force a new tree
    # rather than hand out predecessors rooted in the old graph.
    fresh_tree = object()
    fresh_graph = SimpleNamespace(path_tree=lambda *_args, **_kwargs: fresh_tree)
    monkeypatch.setattr(sr, "_WG", fresh_graph)
    reinit = sr._cell_walkpath_tree(0, sr.ORIGIN_LL["cell-0"])
    assert reinit is fresh_tree
    assert reinit is not first
    assert cache.get((id(fresh_graph), 0, 720)) is fresh_tree


def test_cell_access_tree_singleflight_shares_owner_result_with_waiter(monkeypatch, tiny_wg):
    sr = _install_cell_access_provider(monkeypatch, tiny_wg)
    entered = threading.Event()
    release = threading.Event()
    calls = []
    original_path_tree = tiny_wg.path_tree

    def blocked_path_tree(*args, **kwargs):
        calls.append((args, kwargs))
        entered.set()
        assert release.wait(timeout=2), "test failed to release cell PathTree owner"
        return original_path_tree(*args, **kwargs)

    monkeypatch.setattr(tiny_wg, "path_tree", blocked_path_tree)
    ll = sr.ORIGIN_LL["cell-0"]
    key = (id(tiny_wg), 0, 720)
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(sr._cell_walkpath_tree, 0, ll)
        assert entered.wait(timeout=2)
        flight = sr._CELL_WALKPATH_INFLIGHT[key]
        waiter_entered = threading.Event()
        original_wait = flight["event"].wait

        def observed_wait(*args, **kwargs):
            waiter_entered.set()
            return original_wait(*args, **kwargs)

        flight["event"].wait = observed_wait
        waiter = pool.submit(sr._cell_walkpath_tree, 0, ll)
        assert waiter_entered.wait(timeout=2)
        release.set()
        first = owner.result(timeout=2)
        second = waiter.result(timeout=2)

    assert len(calls) == 1
    assert first is second
    assert first is sr._CELL_WALKPATH_TREE_CACHE.get(key)
    assert not sr._CELL_WALKPATH_INFLIGHT


def test_walkpath_singleflight_wakes_waiters_and_failure_does_not_poison(monkeypatch, tiny_wg):
    target_ix = np.array([2, 7])
    nodes, conn = tiny_wg.snap(
        np.column_stack((tiny_wg.lon[target_ix], tiny_wg.lat[target_ix])))
    sr = _install_tiny_server_walk(monkeypatch, tiny_wg, nodes, conn, nodes, conn)
    lat, lon = float(tiny_wg.lat[0]), float(tiny_wg.lon[0])
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def failing_path_tree(*args, **kwargs):
        calls.append(1)
        entered.set()
        assert release.wait(timeout=2), "test failed to release the PathTree builder"
        raise RuntimeError("synthetic dijkstra failure")

    original_path_tree = tiny_wg.path_tree
    monkeypatch.setattr(tiny_wg, "path_tree", failing_path_tree)
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(sr._walkpath_tree_and_egress, lat, lon)
        assert entered.wait(timeout=2)
        flight = sr._WALKPATH_INFLIGHT[sr.coarse_key(lat, lon)]
        waiter_entered = threading.Event()
        original_wait = flight["event"].wait

        def observed_wait(*args, **kwargs):
            waiter_entered.set()
            return original_wait(*args, **kwargs)

        flight["event"].wait = observed_wait
        waiter = pool.submit(sr._walkpath_tree_and_egress, lat, lon)
        assert waiter_entered.wait(timeout=2)
        release.set()
        for future in (owner, waiter):
            with pytest.raises(RuntimeError, match="synthetic dijkstra failure"):
                future.result(timeout=2)

    assert len(calls) == 1
    assert not sr._WALKPATH_INFLIGHT
    assert len(sr._WALKPATH_TREE_CACHE) == 0
    assert len(sr._RAPTOR_EGRESS_CACHE) == 0

    # A failed flight is not sticky: the next request becomes owner and publishes normally.
    monkeypatch.setattr(tiny_wg, "path_tree", original_path_tree)
    tree, egress = sr._walkpath_tree_and_egress(lat, lon)
    assert tree is sr._WALKPATH_TREE_CACHE.get(sr.coarse_key(lat, lon))
    assert egress is sr._RAPTOR_EGRESS_CACHE.get(sr.coarse_key(lat, lon))


@pytest.fixture(scope="module")
def wg():
    if not os.path.exists(WALK_NPZ):
        pytest.skip("data/walk_graph.npz not built; run scripts/build_walk_graph.py")
    from core import walk
    return walk.WalkGraph.load()


def test_walk_graph_sane(wg):
    assert len(wg.lon) > 100_000                      # SF pedestrian graph is large
    assert len(wg.indices) == len(wg.w_ref) == len(wg.w_flat)
    assert wg.w_ref.min() > 0 and np.isfinite(wg.w_ref).all()
    assert wg.ref_kmh == pytest.approx(4.8, abs=0.01)


def test_walk_is_directional_uphill_slower(wg):
    """The whole point of the hill model: for a directed edge, going UP costs more than the
    grade-agnostic time and going (gently) DOWN costs less — so uphill edges are slower than
    downhill edges. Checked in aggregate over every edge via per-edge slope."""
    # per-edge: source node (expand indptr), dest node (indices), horizontal length from w_flat
    n = len(wg.indptr) - 1
    src = np.repeat(np.arange(n), np.diff(wg.indptr))
    dst = wg.indices
    hlen = wg.w_flat * wg.speed_mps                    # w_flat = hlen / speed
    slope = (wg.elev[dst] - wg.elev[src]) / np.maximum(hlen, 1.0)
    ratio = wg.w_ref / np.maximum(wg.w_flat, 1e-6)     # time multiplier vs flat
    up = ratio[slope > 0.08]                           # climbing
    down = ratio[slope < -0.08]                        # descending
    assert up.mean() > 1.2, f"uphill edges not penalized (mean ratio {up.mean():.2f})"
    assert up.mean() > down.mean(), "uphill is not slower than downhill (model not directional)"
    # a directed edge and its reverse must differ on steep ground (asymmetry, not symmetric cost)
    assert ratio[slope > 0.12].mean() > ratio[(slope < -0.12) & (slope > -0.25)].mean()
