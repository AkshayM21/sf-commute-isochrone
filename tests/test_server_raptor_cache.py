"""Pure concurrency tests for the RAPTOR workplace-tree cache (no server boot or JIT)."""
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "scripts"))


class _FakeJourneyTree:
    def commute_and_dominant(self):
        return np.array([17], dtype=np.int32), np.array(["R"], dtype=object)


class _FakeEngine:
    cell_ids = ["cell"]

    def __init__(self, build):
        self._build = build

    def journey_tree(self, *args, **kwargs):
        return self._build()


def _install(monkeypatch, build, egress=None):
    from core import server_raptor as sr

    monkeypatch.setattr(sr, "RAPTOR_SEMANTIC", "arriveby")
    monkeypatch.setattr(sr, "_RAPTOR", _FakeEngine(build))
    monkeypatch.setattr(sr, "_RAPTOR_TREE_CACHE", sr.BoundedLRU(8, copy_mode="shallow"))
    monkeypatch.setattr(sr, "_RAPTOR_TREE_INFLIGHT", {})
    if egress is None:
        egress = lambda _lat, _lon: (
            np.array([1], dtype=np.int32),
            np.array([60], dtype=np.int64),
            np.array([120], dtype=np.int64),
        )
    monkeypatch.setattr(sr, "raptor_egress_purewalk", egress)
    return sr


def test_raptor_tree_same_key_has_one_owner_and_waiter_reads_shallow_cache(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def build():
        calls.append(1)
        entered.set()
        assert release.wait(timeout=2), "test failed to release RAPTOR tree owner"
        return _FakeJourneyTree()

    sr = _install(monkeypatch, build)
    lat, lon = 37.77, -122.42
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(sr.raptor_tree, lat, lon)
        assert entered.wait(timeout=2)
        flight = sr._RAPTOR_TREE_INFLIGHT[sr.coarse_key(lat, lon)]
        waiter_entered = threading.Event()
        original_wait = flight["event"].wait

        def observed_wait(*args, **kwargs):
            waiter_entered.set()
            return original_wait(*args, **kwargs)

        flight["event"].wait = observed_wait
        waiter = pool.submit(sr.raptor_tree, lat, lon)
        assert waiter_entered.wait(timeout=2)
        release.set()
        first = owner.result(timeout=2)
        second = waiter.result(timeout=2)

    assert len(calls) == 1
    assert not sr._RAPTOR_TREE_INFLIGHT
    assert first is not second                         # waiter's documented shallow cache copy
    assert first["tree"] is second["tree"]             # nested immutable/shared values preserved
    assert first["cells"] is second["cells"]
    assert sr.raptor_tree(lat, lon) is not first        # warm hits also read through the LRU
    assert len(calls) == 1


def test_raptor_tree_waiter_receives_owner_result_even_if_lru_immediately_evicts(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def build():
        entered.set()
        assert release.wait(timeout=2)
        return _FakeJourneyTree()

    sr = _install(monkeypatch, build)
    # A zero-entry cache deterministically models >capacity different-key completions evicting the
    # successful owner before its same-key waiter gets scheduled.
    monkeypatch.setattr(sr, "_RAPTOR_TREE_CACHE", sr.BoundedLRU(0, copy_mode="shallow"))
    lat, lon = 37.77, -122.42
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(sr.raptor_tree, lat, lon)
        assert entered.wait(timeout=2)
        flight = sr._RAPTOR_TREE_INFLIGHT[sr.coarse_key(lat, lon)]
        waiter_entered = threading.Event()
        original_wait = flight["event"].wait

        def observed_wait(*args, **kwargs):
            waiter_entered.set()
            return original_wait(*args, **kwargs)

        flight["event"].wait = observed_wait
        waiter = pool.submit(sr.raptor_tree, lat, lon)
        assert waiter_entered.wait(timeout=2)
        release.set()
        first = owner.result(timeout=2)
        second = waiter.result(timeout=2)

    assert first is not second
    assert first["tree"] is second["tree"]
    assert len(sr._RAPTOR_TREE_CACHE) == 0
    assert not sr._RAPTOR_TREE_INFLIGHT


def test_raptor_tree_different_keys_build_concurrently(monkeypatch):
    barrier = threading.Barrier(2)
    egress_calls = []

    def egress(lat, _lon):
        egress_calls.append(lat)
        barrier.wait(timeout=2)                         # fails if a global build lock serializes keys
        return (np.array([1], dtype=np.int32),
                np.array([60], dtype=np.int64),
                np.array([120], dtype=np.int64))

    sr = _install(monkeypatch, _FakeJourneyTree, egress=egress)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(sr.raptor_tree, 37.76, -122.42)
        b = pool.submit(sr.raptor_tree, 37.78, -122.42)
        assert a.result(timeout=2)["cells"] == {"cell": [17, 17]}
        assert b.result(timeout=2)["cells"] == {"cell": [17, 17]}

    assert sorted(egress_calls) == [37.76, 37.78]
    assert len(sr._RAPTOR_TREE_CACHE) == 2
    assert not sr._RAPTOR_TREE_INFLIGHT


def test_raptor_tree_failure_wakes_waiter_and_retry_is_clean(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    should_fail = [True]
    calls = []

    def build():
        calls.append(1)
        if should_fail[0]:
            entered.set()
            assert release.wait(timeout=2), "test failed to release failing RAPTOR tree owner"
            raise RuntimeError("synthetic tree failure")
        return _FakeJourneyTree()

    sr = _install(monkeypatch, build)
    lat, lon = 37.77, -122.42
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(sr.raptor_tree, lat, lon)
        assert entered.wait(timeout=2)
        flight = sr._RAPTOR_TREE_INFLIGHT[sr.coarse_key(lat, lon)]
        waiter_entered = threading.Event()
        original_wait = flight["event"].wait

        def observed_wait(*args, **kwargs):
            waiter_entered.set()
            return original_wait(*args, **kwargs)

        flight["event"].wait = observed_wait
        waiter = pool.submit(sr.raptor_tree, lat, lon)
        assert waiter_entered.wait(timeout=2)
        release.set()
        for future in (owner, waiter):
            with pytest.raises(RuntimeError, match="synthetic tree failure"):
                future.result(timeout=2)

    assert len(calls) == 1
    assert not sr._RAPTOR_TREE_INFLIGHT

    should_fail[0] = False
    recovered = sr.raptor_tree(lat, lon)
    assert recovered["cells"] == {"cell": [17, 17]}
    assert len(calls) == 2
    assert len(sr._RAPTOR_TREE_CACHE) == 1
    assert not sr._RAPTOR_TREE_INFLIGHT


def test_speed_only_python_entrypoints_derive_the_matching_product_scalar(monkeypatch):
    """A direct Fast/Slow call cannot route with Medium while caching under another speed."""
    tree_scalars = []

    class CapturingEngine:
        cell_ids = ["cell"]

        def journey_tree(self, *args, **kwargs):
            tree_scalars.append(kwargs["walk_scalar"])
            return _FakeJourneyTree()

    sr = _install(monkeypatch, _FakeJourneyTree)
    monkeypatch.setattr(sr, "_RAPTOR", CapturingEngine())
    fast_expected = sr.config.WALK_KMH / sr.WALK_SPEEDS["fast"]

    assert sr.compute_raptor(37.77, -122.42, speed="fast") == {"cell": [17, 17]}
    assert tree_scalars == [pytest.approx(fast_expected)]
    # Attribution reuses the same correctly-keyed tree instead of rebuilding with Medium.
    assert sr.raptor_attribution(37.77, -122.42, speed="fast") == {"cell": "R"}
    assert len(tree_scalars) == 1

    medium_expected = sr.config.WALK_KMH / sr.WALK_SPEEDS[sr.DEFAULT_SPEED]
    assert sr._resolve_walk_speed("not-a-preset") == (
        sr.DEFAULT_SPEED, pytest.approx(medium_expected))
    assert sr._resolve_walk_speed("fast", 1.0) == ("fast", 1.0)

    mc_scalars = []
    monkeypatch.setattr(sr, "_RAPTOR_MC_CACHE", sr.BoundedLRU(2, copy_mode="shallow"))
    monkeypatch.setattr(sr, "_RAPTOR_MC_INFLIGHT", {})

    def fake_mc_build(key, dlat, dlon, max_rides, speed, walk_scalar, perf=None):
        mc_scalars.append((speed, walk_scalar))
        return {"realistic": {}, "variance": {}}

    monkeypatch.setattr(sr, "_raptor_mc_build", fake_mc_build)
    assert sr.raptor_mc(37.77, -122.42, speed="slow") == {
        "realistic": {}, "variance": {}}
    assert mc_scalars == [(
        "slow", pytest.approx(sr.config.WALK_KMH / sr.WALK_SPEEDS["slow"]))]


def test_bounded_lru_enforces_byte_budget_and_replacement_accounting():
    from core import server_raptor as sr

    cache = sr.BoundedLRU(10, maxbytes=7, weight_fn=len)
    cache.put("a", b"1234")
    cache.put("b", b"567")
    assert len(cache) == 2
    assert cache.nbytes == 7

    # A read refreshes recency, so replacing b with a larger value evicts a first.
    assert cache.get("a") == b"1234"
    cache.put("b", b"56789")
    assert cache.get("a") is None
    assert cache.get("b") == b"56789"
    assert cache.nbytes == 5

    # An individual value over budget is deliberately not retained.
    cache.put("huge", b"12345678")
    assert cache.get("huge") is None
    assert cache.nbytes == 0

    cache.put("c", b"12")
    assert cache.pop("c") == b"12"
    assert cache.nbytes == 0


def _isolated_init_state(monkeypatch):
    """Fresh boot globals for lifecycle guards without perturbing the shared server fixture."""
    from core import server_raptor as sr
    for name in (
        "_RAPTOR", "_RAPTOR_STOPS", "_RAPTOR_CELL_POS", "_WG", "_WG_STOP_NODES",
        "_WG_STOP_CONN", "_WG_CELL_NODES", "_WG_CELL_CONN", "_WG_STOP_GIDS", "ORIGIN_LL",
        "_NEED_R5", "_NETWORK", "_NET", "_SNAPPED_GRID", "_DEP", "_MC_SCENARIO_ACTIVE",
        "_MC_SCENARIO_SEQ", "_RAPTOR_EGRESS_CACHE", "_WALKPATH_TREE_CACHE",
        "_CELL_WALKPATH_TREE_CACHE", "_TRANSFER_PATH_CACHE", "_RAPTOR_TREE_CACHE",
        "_RAPTOR_MC_CACHE", "_WALKPATH_INFLIGHT", "_CELL_WALKPATH_INFLIGHT",
        "_RAPTOR_TREE_INFLIGHT", "_RAPTOR_MC_INFLIGHT",
    ):
        monkeypatch.setattr(sr, name, getattr(sr, name))
    monkeypatch.setattr(sr, "_MC_SCENARIO_ACTIVE", None)
    monkeypatch.setattr(sr, "_MC_SCENARIO_SEQ", 0)
    monkeypatch.setattr(sr, "_RAPTOR_EGRESS_CACHE", sr.BoundedLRU(4))
    monkeypatch.setattr(sr, "_WALKPATH_TREE_CACHE", sr.BoundedLRU(4))
    monkeypatch.setattr(sr, "_CELL_WALKPATH_TREE_CACHE", sr.BoundedLRU(4))
    monkeypatch.setattr(sr, "_TRANSFER_PATH_CACHE", sr.BoundedLRU(4))
    monkeypatch.setattr(sr, "_RAPTOR_TREE_CACHE", sr.BoundedLRU(4, copy_mode="shallow"))
    monkeypatch.setattr(sr, "_RAPTOR_MC_CACHE", sr.BoundedLRU(4, copy_mode="shallow"))
    monkeypatch.setattr(sr, "_WALKPATH_INFLIGHT", {})
    monkeypatch.setattr(sr, "_CELL_WALKPATH_INFLIGHT", {})
    monkeypatch.setattr(sr, "_RAPTOR_TREE_INFLIGHT", {})
    monkeypatch.setattr(sr, "_RAPTOR_MC_INFLIGHT", {})
    return sr


def _init_for_test(sr):
    sr.init(raptor=object(), raptor_stops=None, raptor_cell_pos=None, wg=None,
            wg_stop_nodes=None, wg_stop_conn=None, wg_cell_nodes=None, wg_cell_conn=None,
            wg_stop_gids=None, origin_ll=None, need_r5=True)


def test_init_evicts_every_graph_derived_cache_and_retained_scenario(monkeypatch):
    """Boot replacement drops all stale graph roots, including tree-owned nested fragments."""
    sr = _isolated_init_state(monkeypatch)

    class Scenario:
        nbytes = 1024

    nested_geom = {"old-cell": object()}
    caches = (
        sr._RAPTOR_EGRESS_CACHE, sr._WALKPATH_TREE_CACHE, sr._CELL_WALKPATH_TREE_CACHE,
        sr._TRANSFER_PATH_CACHE, sr._RAPTOR_TREE_CACHE, sr._RAPTOR_MC_CACHE,
    )
    for number, cache in enumerate(caches):
        value = ({"geom": nested_geom, "branch_geom": {"old": object()}}
                 if cache is sr._RAPTOR_TREE_CACHE else {"old": number})
        cache.put(("old", number), value)
        assert len(cache) == 1

    key = ("workplace",)
    old = Scenario()
    old_token = sr._retain_mc_scenario(key, old)
    assert sr._mc_scenario_for(old_token, key) is old

    _init_for_test(sr)
    assert all(len(cache) == 0 for cache in caches)
    assert sr._mc_scenario_for(old_token, key) is None
    # The outer tree eviction drops the server-owned entry that contained these fragments.  This
    # local reference intentionally proves no nested cache is mutated independently during init.
    assert nested_geom


@pytest.mark.parametrize("registry", (
    "_WALKPATH_INFLIGHT", "_CELL_WALKPATH_INFLIGHT", "_RAPTOR_TREE_INFLIGHT",
    "_RAPTOR_MC_INFLIGHT",
))
def test_init_rejects_nonquiescent_inflight_registry(monkeypatch, registry):
    sr = _isolated_init_state(monkeypatch)
    getattr(sr, registry)["active"] = object()
    with pytest.raises(AssertionError, match="in-flight"):
        _init_for_test(sr)


def test_init_rejects_active_mc_kernel(monkeypatch):
    sr = _isolated_init_state(monkeypatch)
    assert sr._MC_BUSY.acquire(blocking=False)
    try:
        with pytest.raises(AssertionError, match="active Monte-Carlo kernel"):
            _init_for_test(sr)
    finally:
        sr._MC_BUSY.release()


def test_mc_scenario_reinit_invalidates_old_token_and_same_process_reuses_fresh(monkeypatch):
    """A quiescent boot/test replacement cannot replay an old profile but preserves normal reuse."""
    sr = _isolated_init_state(monkeypatch)

    class Scenario:
        def __init__(self, name):
            self.name = name
            self.nbytes = 1024

    key = ("workplace",)
    old = Scenario("old")
    old_token = sr._retain_mc_scenario(key, old)
    assert sr._mc_scenario_for(old_token, key) is old

    _init_for_test(sr)
    assert sr._mc_scenario_for(old_token, key) is None

    fresh = Scenario("fresh")
    fresh_token = sr._retain_mc_scenario(key, fresh)
    assert fresh_token != old_token
    assert sr._mc_scenario_for(fresh_token, key) is fresh


def test_mc_scenario_requires_the_exact_graph_object():
    """A strong graph reference closes the object-id-reuse hole in engine-level replay."""
    from core.raptor_engine import MonteCarloScenario

    data = {"n_stops": 1}
    scenario = MonteCarloScenario(
        tail_lag=np.zeros((1, 1, 1), np.uint16), deadlines=np.array([60], np.int64),
        delta0_all=np.zeros((1, 1), np.float64), slope_all=np.zeros((1, 1), np.float64),
        seed=7, max_rounds=8, board_slack=60, max_min=90, walk_scalar=1.0,
        data=data, egress_g=np.array([0], np.int32), egress_w=np.array([0], np.int64))
    args = dict(data=data, deadlines=np.array([60], np.int64), egress_g=np.array([0], np.int32),
                egress_w=np.array([0], np.int64), seed=7, n_draws=1, max_rounds=8,
                board_slack=60, max_min=90, walk_scalar=1.0)
    assert scenario.compatible(**args)
    args["data"] = dict(data)
    assert not scenario.compatible(**args)


def test_heavy_cache_key_does_not_alias_distinct_addresses_in_old_110m_bucket():
    from core import server_raptor as sr

    a = sr.coarse_key(37.77001, -122.42001)
    b = sr.coarse_key(37.77049, -122.42049)
    assert a != b


def test_bounded_cell_cache_caps_adversarial_distinct_cell_pokes():
    from core import server_raptor as sr

    cache = sr.BoundedCellCache(2)
    cache[1] = {"route": "one"}
    cache[2] = {"route": "two"}
    cache[2] = {"route": "two-updated"}       # replacement does not evict another entry
    cache[3] = {"route": "three"}

    assert list(cache) == [2, 3]
    assert cache[2] == {"route": "two-updated"}
