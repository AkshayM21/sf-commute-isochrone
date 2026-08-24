"""Synthetic contract tests for the direct runtime-readiness validators."""

import datetime as dt
import io
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from core import config, readiness, raptor_build  # noqa: E402


TARGET = dt.date(2026, 8, 19)       # Wednesday


def _csv(rows):
    return "\n".join(",".join(row) for row in rows) + "\n"


def _feed(path, *, end="20260930", active=True, calendar=True,
          stop_ref="s", trip_ref="trip", stop_lat="37.7", stop_lon="-122.4",
          stop_times=None, calendar_dates=None, trips=None):
    if stop_times is None:
        stop_times = [
            [trip_ref, stop_ref, "1", "08:00:00", "08:00:00"],
            [trip_ref, "s2" if stop_ref == "s" else stop_ref, "2", "08:05:00", "08:05:00"],
        ]
    if trips is None:
        trips = [["r", "wk" if active else "other", "trip"]]
    files = {
        "routes.txt": _csv([["route_id", "route_short_name"], ["r", "R"]]),
        "stops.txt": _csv([["stop_id", "stop_lat", "stop_lon"],
                            ["s", stop_lat, stop_lon], ["s2", "37.71", "-122.41"]]),
        "trips.txt": _csv([["route_id", "service_id", "trip_id"]] + trips),
        "stop_times.txt": _csv([["trip_id", "stop_id", "stop_sequence", "arrival_time",
                                  "departure_time"]] + stop_times),
    }
    if calendar:
        files["calendar.txt"] = _csv([
            ["service_id", "monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday", "start_date", "end_date"],
            ["wk", "1", "1", "1", "1", "1", "1", "1", "20260801", end],
        ])
    else:
        files["calendar_dates.txt"] = _csv(calendar_dates or [
            ["service_id", "date", "exception_type"],
            ["wk", "20260819", "1"], ["wk", "20260826", "1"],
        ])
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in files.items():
            z.writestr(name, text)
    return path


def _raptor(date="20260819"):
    i32 = np.int32
    i64 = np.int64
    return {
        "build_version": raptor_build.BUILD_VERSION,
        "n_stops": 2,
        "stop_lat": np.array([37.7, 37.71]), "stop_lon": np.array([-122.4, -122.41]),
        "stop_name": np.array(["A", "B"]),
        "stop_feed": np.array(["muni", "muni"]),
        "stop_id": np.array(["s", "s2"]),
        "pat_nstops": np.array([2], i32), "pat_ntrips": np.array([1], i32),
        "pat_stop_off": np.array([0, 2], i64), "pat_mat_off": np.array([0, 2], i64),
        "pat_stops": np.array([0, 1], i32), "pat_dep": np.array([100, 200], i32),
        "pat_arr": np.array([90, 190], i32), "pat_feed": np.array([0], i32),
        "pat_line": np.array([0], i32), "pat_mode": np.array([0], np.int8),
        "feeds": ["muni"], "line_table": [("muni", "r", "R", "bus")],
        "ras_off": np.array([0, 1, 1], i64), "ras_pat": np.array([0], i32),
        "ras_pos": np.array([0], i32), "tr_off": np.array([0, 1, 1], i64),
        "tr_to": np.array([1], i32), "tr_time": np.array([30], i32),
        "feed_trip_counts": {"muni": 1}, "date": date, "band": (0, 1000),
        "footpath_m": 250, "source_mtimes": (),
    }


def _walk():
    return {
        "node_lon": np.array([-122.4, -122.41]), "node_lat": np.array([37.7, 37.71]),
        "node_elev": np.array([1., 2.]), "indptr": np.array([0, 1, 1]),
        "indices": np.array([1], np.int32), "w_ref": np.array([10.]),
    }


def _access():
    return {
        "cell_ids": np.array(["0", "1"]), "access_off": np.array([0, 1, 1]),
        "access_to": np.array([0], np.int32), "access_w": np.array([12], np.int32),
        "grid_m": np.array(200), "n_stops": np.array(2), "service_date": np.array("20260819"),
    }


def _static():
    return {"svc_date": "20260819", "grid_m": 200,
            "grid_source_name": "sf_neighborhoods.geojson",
            "grid_source_size": 123, "grid_source_mtime_ns": 456,
            "origin_ll": {"0": [37.7, -122.4]},
            "cells": {"type": "FeatureCollection", "features": []},
            "lines": {"type": "FeatureCollection", "features": []}}


def test_modeled_wednesday_before_after_departure_and_sunday():
    before = dt.datetime(2026, 8, 19, 8, 34, tzinfo=readiness.LA)
    after = dt.datetime(2026, 8, 19, 8, 35, tzinfo=readiness.LA)
    sunday = dt.datetime(2026, 8, 16, 12, 0, tzinfo=readiness.LA)
    assert readiness.modeled_wednesday(before) == TARGET
    assert readiness.modeled_wednesday(after) == dt.date(2026, 8, 26)
    assert readiness.modeled_wednesday(sunday) == TARGET


def test_gtfs_valid_and_calendar_dates_only(tmp_path):
    path = _feed(tmp_path / "muni.zip")
    assert readiness.validate_gtfs_feed(path, TARGET).as_dict() == {"ok": True, "reason_code": "ok"}
    dates_only = _feed(tmp_path / "bart.zip", calendar=False)
    assert readiness.validate_gtfs_feed(dates_only, TARGET).reason_code == "ok"


def test_gtfs_does_not_require_following_wednesday_coverage(tmp_path):
    # Official Muni coverage can end on Friday after the modeled Wednesday. That is enough for
    # this boot; a synthetic following-Wednesday requirement would incorrectly reject it.
    path = _feed(tmp_path / "muni-short-window.zip", end="20260828")
    modeled = dt.date(2026, 8, 26)
    assert readiness.validate_gtfs_feed(path, modeled).reason_code == "ok"


def test_gtfs_stop_times_is_streamed_and_references_are_usable(tmp_path, monkeypatch):
    path = _feed(tmp_path / "streamed.zip")
    original_read = zipfile.ZipFile.read

    def no_stop_times_materialization(self, name, *args, **kwargs):
        if name == "stop_times.txt":
            raise AssertionError("stop_times.txt must be streamed, not ZipFile.read()")
        return original_read(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", no_stop_times_materialization)
    assert readiness.validate_gtfs_feed(path, TARGET).reason_code == "ok"
    assert readiness.validate_gtfs_feed(
        _feed(tmp_path / "bad-ref.zip", trip_ref="missing"), TARGET
    ).reason_code == "invalid_feed"


def test_gtfs_stop_times_trip_invariants_and_interleaving(tmp_path):
    one = [["trip", "s", "1", "08:00:00", "08:00:00"]]
    duplicate_sequence = [one[0], ["trip", "s2", "1", "08:05:00", "08:05:00"]]
    time_reversal = [one[0], ["trip", "s2", "2", "07:59:00", "08:05:00"]]
    arrival_after_departure = [one[0], ["trip", "s2", "2", "08:06:00", "08:05:00"]]
    for name, rows in (("one", one), ("sequence", duplicate_sequence),
                       ("time", time_reversal), ("dwell", arrival_after_departure)):
        assert readiness.validate_gtfs_feed(
            _feed(tmp_path / f"{name}.zip", stop_times=rows), TARGET
        ).reason_code == "invalid_feed"

    interleaved = [
        ["a", "s", "1", "08:00:00", "08:00:00"],
        ["b", "s", "1", "08:01:00", "08:01:00"],
        ["a", "s2", "2", "08:05:00", "08:05:00"],
        ["b", "s2", "2", "08:06:00", "08:06:00"],
    ]
    assert readiness.validate_gtfs_feed(
        _feed(tmp_path / "interleaved.zip", trips=[["r", "wk", "a"], ["r", "wk", "b"]],
              stop_times=interleaved), TARGET
    ).reason_code == "ok"


def test_gtfs_malformed_dates_and_contradictory_exceptions_are_safe(tmp_path):
    feed = _feed(tmp_path / "dates.zip")
    assert readiness.validate_gtfs_feed(feed, "not-a-date").reason_code == "invalid_feed"
    assert readiness.validate_gtfs_feed(feed, "2026-99-99").reason_code == "invalid_feed"
    exceptions = [
        ["service_id", "date", "exception_type"],
        ["wk", "20260819", "1"], ["wk", "20260819", "2"],
        ["wk", "20260826", "1"],
    ]
    contradictory = _feed(tmp_path / "contradictory.zip", calendar=False,
                          calendar_dates=exceptions)
    assert readiness.validate_gtfs_feed(contradictory, TARGET).reason_code == "invalid_feed"
    assert readiness.validate_gtfs_feed(
        _feed(tmp_path / "bad-coord.zip", stop_lat="not-a-number"), TARGET
    ).reason_code == "invalid_feed"


def test_gtfs_safe_failure_categories(tmp_path):
    assert readiness.validate_gtfs_feed(tmp_path / "missing.zip", TARGET).reason_code == "missing_feed"
    expired = _feed(tmp_path / "expired.zip", end="20260818")
    assert readiness.validate_gtfs_feed(expired, TARGET).reason_code == "calendar_expired"
    empty = _feed(tmp_path / "empty.zip", active=False)
    assert readiness.validate_gtfs_feed(empty, TARGET).reason_code == "no_service"
    malformed = tmp_path / "malformed.zip"
    with zipfile.ZipFile(malformed, "w") as z:
        z.writestr("routes.txt", "not,a,valid,table\n")
    assert readiness.validate_gtfs_feed(malformed, TARGET).reason_code == "invalid_feed"
    assert all(c in readiness.REASON_CODES for c in (
        "missing_feed", "calendar_expired", "no_service", "invalid_feed"))


def test_required_feed_roles_are_all_required(tmp_path):
    muni = _feed(tmp_path / "muni.zip")
    feeds = {"muni": muni, "bart": muni}
    assert readiness.validate_required_feeds(feeds, TARGET).as_dict() == {
        "ok": False, "reason_code": "missing_feed", "detail": "caltrain"
    }
    feeds["caltrain"] = muni
    assert readiness.validate_required_feeds(feeds, TARGET).reason_code == "ok"


def test_runtime_structural_validators_and_corruption(tmp_path):
    raptor = _raptor()
    assert readiness.validate_raptor_state(raptor, TARGET).reason_code == "ok"
    broken = dict(raptor)
    broken["pat_dep"] = np.array([100])
    assert readiness.validate_raptor_state(broken, TARGET).reason_code == "invalid_cache"
    assert readiness.validate_raptor_state(raptor, "bad-date").reason_code == "runtime_load_failed"
    pkl = tmp_path / "raptor.pkl"
    pkl.write_bytes(pickle.dumps(raptor))
    assert readiness.validate_raptor_state(pkl, TARGET).reason_code == "ok"
    assert readiness.validate_raptor_state(tmp_path / "none.pkl").reason_code == "missing_cache"
    bad_pickle = tmp_path / "bad.pkl"
    bad_pickle.write_bytes(b"not a pickle")
    assert readiness.validate_raptor_state(bad_pickle).reason_code == "runtime_load_failed"

    assert readiness.validate_walk_graph(_walk()).reason_code == "ok"
    walk_bad = _walk(); walk_bad["indices"] = np.array([2])
    assert readiness.validate_walk_graph(walk_bad).reason_code == "runtime_load_failed"
    walk_float_offsets = _walk(); walk_float_offsets["indptr"] = np.array([0.0, 1.0, 1.0])
    assert readiness.validate_walk_graph(walk_float_offsets).reason_code == "runtime_load_failed"
    walk_bad_coord = _walk(); walk_bad_coord["node_lat"] = np.array([91.0, 37.71])
    assert readiness.validate_walk_graph(walk_bad_coord).reason_code == "runtime_load_failed"
    assert readiness.validate_access_bake(_access(), expected_n_stops=2, expected_grid_m=200,
                                         service_date=TARGET).reason_code == "ok"
    access_bad = _access(); access_bad["access_off"] = np.array([0, 2, 1])
    assert readiness.validate_access_bake(access_bad).reason_code == "runtime_load_failed"
    access_float_offsets = _access(); access_float_offsets["access_off"] = np.array([0.0, 1.0, 1.0])
    assert readiness.validate_access_bake(access_float_offsets).reason_code == "runtime_load_failed"
    access_bad_date = _access(); access_bad_date["service_date"] = "bad-date"
    assert readiness.validate_access_bake(access_bad_date).reason_code == "runtime_load_failed"
    assert readiness.validate_walk_graph(tmp_path / "none.npz").reason_code == "missing_walk_graph"
    assert readiness.validate_access_bake(tmp_path / "none-access.npz").reason_code == "missing_access_bake"


def test_static_bundle_and_aggregate_readiness(tmp_path):
    bundle = _static()
    assert readiness.validate_static_bundle(bundle, TARGET).reason_code == "ok"
    wrong = dict(bundle); wrong["svc_date"] = "20260826"
    assert readiness.validate_static_bundle(wrong, TARGET).reason_code == "service_date_stale"
    malformed_date = dict(bundle); malformed_date["svc_date"] = "bad-date"
    assert readiness.validate_static_bundle(malformed_date).reason_code == "runtime_load_failed"
    assert readiness.validate_static_bundle(tmp_path / "none.json").reason_code == "missing_static_bundle"
    bad_origin = dict(bundle); bad_origin["origin_ll"] = {"0": [-181, 37.7]}
    assert readiness.validate_static_bundle(bad_origin, TARGET).reason_code == "runtime_load_failed"
    feed = _feed(tmp_path / "feed.zip")
    feeds = {role: feed for role in readiness.DEFAULT_REQUIRED_FEEDS}
    stat = feed.stat()
    bundle["source_mtimes"] = [
        (feed.name, stat.st_size, config.portable_mtime_ns(stat))
    ] * 3
    result = readiness.check_readiness(feeds, _raptor(), _walk(), _access(), bundle,
                                       now=dt.datetime(2026, 8, 16, 12, tzinfo=readiness.LA),
                                       grid_m=200)
    assert result.as_dict() == {"ok": True, "reason_code": "ok",
                                "service_date": "20260819"}
    raptor_path = tmp_path / "runtime.pkl"
    raptor_path.write_bytes(pickle.dumps(_raptor()))
    access_wrong_stops = _access(); access_wrong_stops["n_stops"] = 999
    mismatch = readiness.check_readiness(
        feeds, raptor_path, _walk(), access_wrong_stops, bundle,
        now=dt.datetime(2026, 8, 16, 12, tzinfo=readiness.LA), grid_m=200)
    assert mismatch.reason_code == "runtime_load_failed"
    missing = readiness.check_readiness({}, _raptor(), _walk(), _access(), bundle,
                                        now=dt.datetime(2026, 8, 16, tzinfo=readiness.LA))
    assert missing.reason_code == "missing_feed"


def test_static_bundle_requires_current_grid_resolution_and_source_metadata(tmp_path):
    source = tmp_path / "sf_neighborhoods.geojson"
    source.write_text("{}")
    stat = source.stat()
    bundle = _static()
    bundle.update({"grid_source_name": source.name, "grid_source_size": stat.st_size,
                   "grid_source_mtime_ns": config.portable_mtime_ns(stat)})
    assert readiness.validate_static_bundle(
        bundle, TARGET, expected_grid_m=200, expected_grid_source=source
    ).reason_code == "ok"
    assert readiness.validate_static_bundle(bundle, TARGET, expected_grid_m=400).reason_code \
        == "runtime_load_failed"
    source.write_text("{\"changed\": true}")
    assert readiness.validate_static_bundle(
        bundle, TARGET, expected_grid_source=source
    ).reason_code == "runtime_load_failed"
    missing_meta = _static(); missing_meta.pop("grid_source_mtime_ns")
    assert readiness.validate_static_bundle(missing_meta, TARGET).reason_code == "runtime_load_failed"


def test_static_bundle_rejects_changed_direct_gtfs_metadata(tmp_path):
    feeds = []
    for name in ("muni_current.zip", "bart_gtfs.zip", "caltrain.zip"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        feeds.append(path)
    stat = [p.stat() for p in feeds]
    bundle = _static()
    bundle["source_mtimes"] = [
        (p.name, s.st_size, config.portable_mtime_ns(s)) for p, s in zip(feeds, stat)
    ]
    expected = tuple(
        (p.name, s.st_size, config.portable_mtime_ns(s)) for p, s in zip(feeds, stat)
    )
    assert readiness.validate_static_bundle(bundle, TARGET,
                                            expected_gtfs_sources=expected).reason_code == "ok"
    feeds[0].write_bytes(b"new current feed with a different size")
    changed = tuple(
        (p.name, p.stat().st_size, config.portable_mtime_ns(p.stat())) for p in feeds
    )
    assert readiness.validate_static_bundle(bundle, TARGET,
                                            expected_gtfs_sources=changed).reason_code \
        == "runtime_load_failed"


def test_runtime_state_reason_codes_and_contract_completeness():
    assert readiness.validate_runtime_state(
        engine_kind="legacy", semantic="arriveby", graph_backed=True,
        engine=object(), walk_graph=object(), initialized=True
    ).reason_code == "wrong_engine"
    assert readiness.validate_runtime_state(
        engine_kind="raptor", semantic="arriveby", graph_backed=True,
        engine=None, walk_graph=object()
    ).reason_code == "runtime_uninitialized"
    assert readiness.validate_runtime_state(
        engine_kind="raptor", semantic="arriveby", graph_backed=True,
        engine=object(), walk_graph=object(), initialized=None
    ).reason_code == "runtime_uninitialized"
    assert readiness.validate_runtime_state(
        engine_kind="raptor", semantic="arriveby", graph_backed=False,
        engine=object(), walk_graph=object(), initialized=True
    ).reason_code == "wrong_engine"
    assert readiness.validate_runtime_state(
        engine_kind="raptor", semantic="departafter", graph_backed=True,
        engine=object(), walk_graph=object(), initialized=True,
        service_date=dt.date(2026, 8, 26),
        now=dt.datetime(2026, 8, 16, tzinfo=readiness.LA)
    ).reason_code == "service_date_stale"
    assert readiness.validate_runtime_state(
        engine_kind="raptor", semantic="departafter", graph_backed=True,
        engine=object(), walk_graph=object(), initialized=True, service_date=TARGET,
        now=dt.datetime(2026, 8, 16, tzinfo=readiness.LA)
    ).reason_code == "ok"
    assert readiness.validate_runtime_state(
        engine_kind="raptor", semantic="departafter", graph_backed=True,
        engine=object(), walk_graph=object(), initialized=True, service_date="bad-date",
        now=dt.datetime(2026, 8, 16, tzinfo=readiness.LA)
    ).reason_code == "service_date_stale"
    approved = {
        "missing_feed", "invalid_feed", "no_service", "calendar_expired",
        "service_date_stale", "missing_cache", "invalid_cache", "missing_walk_graph",
        "missing_access_bake", "missing_static_bundle", "runtime_uninitialized",
        "runtime_load_failed", "wrong_engine", "ok",
    }
    assert readiness.REASON_CODES == approved


def test_source_metadata_accepts_legacy_nanoseconds_after_rsync_truncation():
    local = (("muni.zip", 1234, 1_777_777_777_987_654_321),)
    transferred = (("muni.zip", 1234, 1_777_777_777_000_000_000),)

    assert readiness._same_source_metadata(local, transferred)
    assert readiness._grid_source_metadata(
        {"name": "grid.geojson", "size": 10, "mtime_ns": local[0][2]}
    ) == ("grid.geojson", 10, transferred[0][2])
