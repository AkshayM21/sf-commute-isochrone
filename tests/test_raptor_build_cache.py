"""Focused persistence tests for the RAPTOR structure cache.

These tests use a tiny structurally valid payload, so cache behavior is exercised without
requiring a local GTFS download or running the expensive parser.
"""
import datetime as dt
import multiprocessing
import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from core import raptor_build


def test_source_metadata_is_second_aligned_for_cross_host_transfers(tmp_path):
    feed = tmp_path / "feed.zip"
    feed.write_bytes(b"feed")
    os.utime(feed, ns=(1_777_777_777_987_654_321,) * 2)

    assert raptor_build._source_mtimes([feed]) == (
        ("feed.zip", 4, 1_777_777_777_000_000_000),
    )


def _payload(marker=None):
    """Return the smallest payload accepted by the production cache validator."""
    empty_i32 = np.zeros(0, dtype=np.int32)
    empty_i64 = np.zeros(1, dtype=np.int64)
    data = {
        "build_version": raptor_build.BUILD_VERSION,
        "n_stops": 1,
        "stop_feed": np.asarray(["test"], dtype="U"),
        "stop_id": np.asarray(["S1"], dtype="U"),
        "stop_lat": np.array([37.75]),
        "stop_lon": np.array([-122.45]),
        "stop_name": ["test stop"],
        "pat_nstops": empty_i32,
        "pat_ntrips": empty_i32,
        "pat_stop_off": empty_i64.copy(),
        "pat_mat_off": empty_i64.copy(),
        "pat_stops": empty_i32,
        "pat_dep": empty_i32,
        "pat_arr": empty_i32,
        "pat_feed": np.zeros(0, dtype=np.int16),
        "pat_line": empty_i32,
        "pat_mode": np.zeros(0, dtype=np.int8),
        "feeds": ["test"],
        "line_table": [],
        "ras_off": np.array([0, 0], dtype=np.int64),
        "ras_pat": empty_i32,
        "ras_pos": empty_i32,
        "tr_off": np.array([0, 0], dtype=np.int64),
        "tr_to": empty_i32,
        "tr_time": empty_i32,
        "feed_trip_counts": {"test": 1},
        "date": "20260101",
        "band": (18000, 36000),
        "footpath_m": 250.0,
        "source_mtimes": (("feed.zip", None, None),),
    }
    if marker is not None:
        data["marker"] = marker
    return data


def _loader(monkeypatch, tmp_path, builder):
    cache_dir = tmp_path / "raptor_cache"
    monkeypatch.setattr(raptor_build, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(
        raptor_build, "_cache_path",
        lambda *args: cache_dir / "raptor_test.pkl",
    )
    monkeypatch.setattr(raptor_build, "band_seconds", lambda: (18000, 36000))
    monkeypatch.setattr(raptor_build, "build", builder)
    return [tmp_path / "feed.zip"], dt.date(2026, 1, 1)


def test_corrupt_and_truncated_cache_are_rebuilt(monkeypatch, tmp_path):
    builds = []

    def builder(*args):
        builds.append(1)
        return _payload(marker=len(builds))

    paths, date = _loader(monkeypatch, tmp_path, builder)
    cache = tmp_path / "raptor_cache" / "raptor_test.pkl"
    cache.parent.mkdir()
    cache.write_bytes(b"not a pickle")

    first = raptor_build.load_or_build(paths, date, verbose=False)
    assert first["marker"] == 1
    assert len(builds) == 1

    # A valid published cache is a hit; truncate it and verify the next call self-heals.
    assert raptor_build._validate_cache_data(pickle.loads(cache.read_bytes()))
    cache.write_bytes(cache.read_bytes()[:7])
    second = raptor_build.load_or_build(paths, date, verbose=False)
    assert second["marker"] == 2
    assert len(builds) == 2
    assert raptor_build._validate_cache_data(pickle.loads(cache.read_bytes()))


def test_schema_incompatible_cache_is_a_miss(monkeypatch, tmp_path):
    builds = []

    def builder(*args):
        builds.append(1)
        return _payload(marker=1)

    paths, date = _loader(monkeypatch, tmp_path, builder)
    cache = tmp_path / "raptor_cache" / "raptor_test.pkl"
    cache.parent.mkdir()
    incompatible = _payload()
    incompatible.pop("tr_time")
    cache.write_bytes(pickle.dumps(incompatible))

    result = raptor_build.load_or_build(paths, date, verbose=False)
    assert result["marker"] == 1
    assert len(builds) == 1


def test_validation_failure_does_not_replace_existing_cache(monkeypatch, tmp_path):
    cache = tmp_path / "raptor.pkl"
    old = _payload(marker="old")
    cache.write_bytes(pickle.dumps(old))
    old_bytes = cache.read_bytes()

    assert not raptor_build._publish_cache(cache, {"build_version": raptor_build.BUILD_VERSION},
                                           verbose=False)
    assert cache.read_bytes() == old_bytes
    assert not list(cache.parent.glob(f".{cache.name}.*.tmp"))

    real_replace = raptor_build.os.replace

    def fail_replace(*args):
        raise OSError("simulated publish failure")

    monkeypatch.setattr(raptor_build.os, "replace", fail_replace)
    assert not raptor_build._publish_cache(cache, _payload(marker="new"), verbose=False)
    assert cache.read_bytes() == old_bytes
    monkeypatch.setattr(raptor_build.os, "replace", real_replace)


def test_same_cache_identity_builds_once_per_process(monkeypatch, tmp_path):
    builds = 0
    builds_lock = threading.Lock()

    def builder(*args):
        nonlocal builds
        with builds_lock:
            builds += 1
            marker = builds
        time.sleep(0.05)
        return _payload(marker=marker)

    paths, date = _loader(monkeypatch, tmp_path, builder)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(
            lambda _: raptor_build.load_or_build(paths, date, verbose=False), range(4)))

    assert builds == 1
    assert [r["marker"] for r in results] == [1, 1, 1, 1]


@pytest.mark.parametrize("field,broken", [
    ("pat_feed", np.array([0], dtype=np.int16)),
    ("pat_stop_off", np.array([1], dtype=np.int64)),
    ("pat_stops", np.array([0.5])),
    ("ras_off", np.array([0, 1], dtype=np.int64)),
    ("tr_to", np.array([0], dtype=np.int32)),
])
def test_structural_array_mismatches_are_rejected(field, broken):
    data = _payload()
    data[field] = broken
    assert not raptor_build._validate_cache_data(data)


def test_cached_parameters_must_match_current_request(monkeypatch, tmp_path):
    builds = []

    def builder(*args):
        builds.append(1)
        data = _payload(marker="rebuilt")
        data["footpath_m"] = args[4]
        return data

    paths, date = _loader(monkeypatch, tmp_path, builder)
    cache = tmp_path / "raptor_cache" / "raptor_test.pkl"
    cache.parent.mkdir()
    stale = _payload(marker="stale")
    stale["footpath_m"] = 100.0
    cache.write_bytes(pickle.dumps(stale))

    result = raptor_build.load_or_build(paths, date, footpath_m=250.0, verbose=False)
    assert result["marker"] == "rebuilt"
    assert len(builds) == 1
    assert pickle.loads(cache.read_bytes())["footpath_m"] == 250.0


def test_canonical_stop_identity_is_required_and_gid_aligned():
    data = _payload()
    assert raptor_build._validate_cache_data(data)
    data.pop("stop_id")
    assert not raptor_build._validate_cache_data(data)
    data = _payload()
    data["stop_id"] = np.asarray(["different"], dtype="U")
    # A single identity is still structurally valid; duplicate identities are not.
    assert raptor_build._validate_cache_data(data)
    data["stop_feed"] = np.asarray(["test"], dtype="U")
    data["stop_id"] = np.asarray([""], dtype="U")
    assert not raptor_build._validate_cache_data(data)


def test_feed_source_mtime_change_invalidates_canonical_cache(monkeypatch, tmp_path):
    builds = []
    feed = tmp_path / "feed.zip"
    feed.write_bytes(b"one")

    def builder(*args):
        builds.append(1)
        return _payload(marker=len(builds))

    paths, date = _loader(monkeypatch, tmp_path, builder)
    paths[:] = [feed]
    first = raptor_build.load_or_build(paths, date, verbose=False)
    assert first["marker"] == 1
    original_mtime_ns = feed.stat().st_mtime_ns
    feed.write_bytes(b"two")
    os.utime(feed, ns=(original_mtime_ns + 2_000_000_000,) * 2)
    second = raptor_build.load_or_build(paths, date, verbose=False)
    assert second["marker"] == 2
    assert len(builds) == 2


def _atomic_writer(cache_path, marker, start, failures):
    start.wait()
    for _ in range(30):
        if not raptor_build._publish_cache(
                Path(cache_path), _payload(marker=marker), verbose=False):
            with failures.get_lock():
                failures.value += 1


def _atomic_reader(cache_path, start, failures):
    start.wait()
    for _ in range(1500):
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            if not raptor_build._validate_cache_data(data):
                raise ValueError("reader observed structurally invalid cache")
        except Exception:
            with failures.get_lock():
                failures.value += 1
            return


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(),
                    reason="atomic cache process test requires fork")
def test_concurrent_process_publishers_never_expose_partial_cache(tmp_path):
    cache = tmp_path / "raptor.pkl"
    assert raptor_build._publish_cache(cache, _payload(marker="initial"), verbose=False)
    ctx = multiprocessing.get_context("fork")
    start = ctx.Event()
    failures = ctx.Value("i", 0)
    processes = [
        ctx.Process(target=_atomic_writer, args=(str(cache), "writer-a", start, failures)),
        ctx.Process(target=_atomic_writer, args=(str(cache), "writer-b", start, failures)),
        ctx.Process(target=_atomic_reader, args=(str(cache), start, failures)),
    ]
    for process in processes:
        process.start()
    start.set()
    try:
        for process in processes:
            process.join(timeout=15)
        assert all(not process.is_alive() for process in processes)
        assert all(process.exitcode == 0 for process in processes)
        assert failures.value == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    final = pickle.loads(cache.read_bytes())
    assert raptor_build._validate_cache_data(final)
    assert final["marker"] in {"writer-a", "writer-b"}
