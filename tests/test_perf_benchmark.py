"""Pure unit tests for scripts/perf_benchmark.py (no server import, JIT, or bound port)."""

import json

from types import SimpleNamespace

import perf_benchmark as bench
import pytest


def test_normalized_response_excludes_only_top_level_server_timing_for_direct_comparison():
    left = {"cells": {"1": [20, 21]}, "ms": 140, "nested": {"ms": "product-data"}}
    right = {"nested": {"ms": "product-data"}, "ms": 0, "cells": {"1": [20, 21]}}
    changed = {"cells": {"1": [20, 22]}, "ms": 0, "nested": {"ms": "product-data"}}
    assert bench.normalized_response(left) == bench.normalized_response(right)
    assert bench.normalized_response(left) != bench.normalized_response(changed)
    assert "sha" not in bench.__dict__


def test_optional_phase_headers_are_consumed_without_response_fields():
    assert bench.optional_phase_telemetry({
        "server-timing": 'kernel;dur=12.5, overlay;desc="trace";dur=3',
        "x-perf-phases": '{"geometry":4.25}',
    }) == {"kernel": 12.5, "overlay": 3.0, "geometry": 4.25}
    assert bench.optional_phase_telemetry({}) is None


def test_distribution_is_deterministic_and_reports_tail():
    got = bench.distribution([40, 10, 30, 20])
    assert got == {
        "n": 4,
        "min": 10.0,
        "p50": 25.0,
        "p90": 37.0,
        "p95": 38.5,
        "max": 40.0,
        "mean": 25.0,
        "stddev": 11.18,
    }
    assert bench.distribution([]) is None


def test_output_summaries_bound_large_responses():
    compute = {"cells": {"a": [12, 15], "b": [None, None], "c": [9, 9]}, "ms": 4}
    assert bench.output_summary("compute", compute) == {"cells": 3, "reachable": 2}

    route = {
        "total": 22,
        "xfers": 1,
        "family": {"key": "f1"},
        "geom": [{"pts": [[1, 2], [3, 4]]}],
        "alts": [
            {"total": 23, "family": {"key": "f1"}, "geom": [{"pts": [[5, 6]]}]},
            {"total": 24, "family": {"key": "f2"}, "geom": []},
        ],
    }
    summary = bench.output_summary("pin", route)
    assert summary["options"] == 3
    assert summary["families"] == 2
    assert summary["geometry_points"] == 3


class _FakeClient:
    def __init__(self):
        self.paths = []
        self.endpoint_calls = 0

    def request_json(self, path):
        self.paths.append(path)
        if path == "/healthz":
            return {
                "body": {
                    "ok": True,
                    "engine": "raptor",
                    "semantic": "departafter",
                    "walk": "graph",
                    "svc_date": "2026-07-17",
                    "uptime_s": 9,
                    "benchmark": {
                        "pid": 123,
                        "rss_bytes": 1000,
                        "cache_counts": {"raptor_tree_workplaces": self.endpoint_calls},
                    },
                }
            }
        self.endpoint_calls += 1
        if path.startswith("/compute?"):
            body = {"cells": {"1916": [22, 22]}, "ms": self.endpoint_calls}
        elif path.startswith("/variance?"):
            body = {"variance": {"1916": {"frag": 3}}, "ms": self.endpoint_calls}
        else:
            body = {
                "total": 22,
                "xfers": 0,
                "legs": [],
                "geom": [{"mode": "walk", "pts": [[37.7, -122.4], [37.8, -122.3]]}],
                "alts": [],
                "family": {"key": "family:test"},
                "branch": {"key": "branch:test"},
            }
            if "pin=1" in path:
                body["frag"] = 3
        return {
            "status": 200,
            "client_ms": float(self.endpoint_calls),
            "response_bytes": 100 + self.endpoint_calls,
            "headers": {},
            "body": body,
            "retries": 0,
        }


def test_run_case_uses_progressive_ui_order_and_names_first_in_flow_honestly():
    fake = _FakeClient()
    case = bench.PUBLIC_CASES["townsend"]
    result = bench.run_case(fake, case, "med", warm_repeats=2)
    endpoint_paths = [path.split("?", 1)[0] for path in fake.paths if path != "/healthz"]
    assert endpoint_paths == [
        "/compute", "/compute", "/compute",
        "/itinerary", "/itinerary", "/itinerary",
        "/variance", "/variance", "/variance",
        "/itinerary", "/itinerary", "/itinerary",
    ]
    assert all(ep["output_equivalent"] for ep in result["endpoints"].values())
    assert "pin=1" in result["endpoints"]["pin"]["path"]
    assert result["endpoints"]["compute"]["resources"]["first_in_flow_cache_delta"] == {
        "raptor_tree_workplaces": 1
    }
    assert result["endpoints"]["compute"]["first_in_flow"][0]["cache_state"] == \
        "first_in_flow"
    compute = result["endpoints"]["compute"]
    sample = compute["first_in_flow"][0]
    assert sample["response_ms"] == 1.0
    assert sample["transport_remainder_ms"] == 0.0
    assert "signature" not in sample
    assert compute["output_comparison"]["method"] == "direct_normalized_object_equality"
    assert "cold" not in json.dumps(result)


def _snapshot(*, engine="raptor", semantic="departafter", walk="graph", uptime=9, pid=123):
    benchmark = None if pid is None else {"pid": pid, "rss_bytes": 1000, "cache_counts": {}}
    return {
        "server": {
            "ok": True, "engine": engine, "semantic": semantic, "walk": walk,
            "svc_date": "2026-07-17",
        },
        "uptime_s": uptime,
        "health_benchmark": benchmark,
        "local_server_pid": None,
        "local_rss_bytes": None,
    }


def test_health_guard_requires_default_stack_but_allows_explicit_alternate():
    with pytest.raises(RuntimeError, match="requires"):
        bench.HealthGuard().observe(_snapshot(semantic="arriveby"))

    guard = bench.HealthGuard(allow_alternate_stack=True)
    guard.observe(_snapshot(semantic="arriveby", uptime=10))
    guard.observe(_snapshot(semantic="arriveby", uptime=11))
    with pytest.raises(RuntimeError, match="configuration changed"):
        guard.observe(_snapshot(semantic="departafter", uptime=12))

    with pytest.raises(RuntimeError, match="service date"):
        bench.HealthGuard(expected_svc_date="2026-07-18").observe(_snapshot())


def test_health_guard_detects_uptime_and_pid_resets():
    guard = bench.HealthGuard()
    guard.observe(_snapshot(uptime=20, pid=123))
    with pytest.raises(RuntimeError, match="uptime reset"):
        guard.observe(_snapshot(uptime=2, pid=123))

    guard = bench.HealthGuard()
    guard.observe(_snapshot(uptime=20, pid=123))
    with pytest.raises(RuntimeError, match="PID changed"):
        guard.observe(_snapshot(uptime=21, pid=456))


def test_endpoint_contracts_reject_stable_empty_or_unenriched_payloads():
    case = bench.PUBLIC_CASES["townsend"]
    with pytest.raises(RuntimeError, match="cells is empty"):
        bench.validate_endpoint_contract("compute", {"cells": {}}, case)
    with pytest.raises(RuntimeError, match="selected cell"):
        bench.validate_endpoint_contract("variance", {"variance": {"other": {}}}, case)

    route = {
        "total": 22,
        "geom": [{"pts": [[37.7, -122.4], [37.8, -122.3]]}],
        "family": {"key": "f"},
        "branch": {"key": "b"},
        "alts": [],
    }
    bench.validate_endpoint_contract("itinerary", route, case)
    with pytest.raises(RuntimeError, match="fragility enrichment"):
        bench.validate_endpoint_contract("pin", route, case)
    bench.validate_endpoint_contract("pin", {**route, "frag": 2}, case)

    no_geometry = {**route, "geom": []}
    with pytest.raises(RuntimeError, match="geometry is empty"):
        bench.validate_endpoint_contract("itinerary", no_geometry, case)


def test_browser_storage_init_is_an_executed_script_body():
    assert "localStorage.clear()" in bench.CLEAR_STORAGE_SCRIPT
    assert not bench.CLEAR_STORAGE_SCRIPT.lstrip().startswith("() =>")


def test_matrix_validation_is_ordered_and_rejects_unknown_values():
    args = SimpleNamespace(
        destinations="market,townsend,market", speeds="fast,med", warm_repeats=1,
        browser=False, browser_repeats=1,
    )
    matrix = bench._validated_matrix(args)
    assert [(case["slug"], speed) for case, speed in matrix] == [
        ("market", "fast"), ("market", "med"),
        ("townsend", "fast"), ("townsend", "med"),
    ]

    args.destinations = "market,townsend,city_hall"
    args.speeds = "slow,med,fast"
    args.warm_repeats = 3
    with pytest.raises(ValueError, match="/variance calls"):
        bench._validated_matrix(args)

    args.destinations = "townsend"
    args.speeds = "slow,med,fast"
    args.browser = True
    args.browser_repeats = 1
    bench._validated_matrix(args)  # 3 * (1 first + 3 warm + 2 browser pages) == 18
    args.browser_repeats = 2
    with pytest.raises(ValueError, match="/variance calls"):
        bench._validated_matrix(args)


def test_bounded_public_five_by_three_matrix_uses_first_in_flow_distribution_only():
    args = SimpleNamespace(
        destinations="public5", speeds="slow,med,fast", warm_repeats=0,
        browser=False, browser_repeats=1,
    )
    matrix = bench._validated_matrix(args)
    assert len(matrix) == 15
    assert {case["slug"] for case, _ in matrix} == set(bench.PUBLIC_CASES)
    assert {speed for _, speed in matrix} == {"slow", "med", "fast"}


def test_controlled_paths_send_the_ride_cap_to_every_endpoint():
    paths = dict(bench.endpoint_paths(bench.PUBLIC_CASES["townsend"], "med", max_rides=3))
    assert "maxrides=3" in paths["compute"]
    assert "maxrides=3" in paths["itinerary"]
    assert "maxrides=3" in paths["variance"]
    assert "maxrides=3" in paths["pin"]
    assert bench._frontend_transfer_mode(1) == "0"
    assert bench._frontend_transfer_mode(2) == "1"
    assert bench._frontend_transfer_mode(3) == "2"
    assert bench._frontend_transfer_mode(8) == "any"
    with pytest.raises(ValueError, match="supports max rides"):
        bench._frontend_transfer_mode(4)


def test_controlled_matrix_validates_explicit_routing_and_thread_settings():
    args = SimpleNamespace(
        destinations="townsend", speeds="med", warm_repeats=0, max_rides=8,
        numba_threads=1, browser=False, browser_repeats=1,
    )
    assert len(bench._validated_matrix(args)) == 1
    args.max_rides = 0
    with pytest.raises(ValueError, match="max-rides"):
        bench._validated_matrix(args)
    args.max_rides = 8
    args.numba_threads = 0
    with pytest.raises(ValueError, match="numba-threads"):
        bench._validated_matrix(args)


def test_named_baseline_uses_actual_normalized_objects_not_a_signature():
    fake = _FakeClient()
    case = bench.PUBLIC_CASES["townsend"]
    current = [bench.run_case(
        fake, case, "med", warm_repeats=0, max_rides=8, capture_response_references=True,
    )]
    identity = bench.workload_identity(current, _snapshot())
    baseline = {
        "workload_identity": identity,
        "cases": current,
    }
    assert bench.baseline_equivalence_report(current, baseline, identity) == {
        "ok": True, "mismatches": [],
    }
    changed = json.loads(json.dumps(baseline))
    changed["cases"][0]["endpoints"]["variance"]["direct_normalized_reference"]["variance"] = {}
    comparison = bench.baseline_equivalence_report(current, changed, identity)
    assert comparison["ok"] is False
    assert comparison["mismatches"][0]["endpoint"] == "variance"

    def keys(value):
        if isinstance(value, dict):
            yield from value
            for child in value.values():
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    # Inspect structured field names, never arbitrary values such as the local hostname.
    assert not {"sha", "signature", "integrity"} & {key.lower() for key in keys(current)}


def test_baseline_comparison_refuses_unlike_workloads_or_missing_references():
    fake = _FakeClient()
    case = bench.PUBLIC_CASES["townsend"]
    current = [bench.run_case(
        fake, case, "med", warm_repeats=0, max_rides=8, capture_response_references=True,
    )]
    identity = bench.workload_identity(current, _snapshot())
    with pytest.raises(ValueError, match="workload identity differs"):
        bench.baseline_equivalence_report(current, {"workload_identity": {}}, identity)
    stripped = json.loads(json.dumps(current))
    for endpoint in stripped[0]["endpoints"].values():
        endpoint.pop("direct_normalized_reference")
    with pytest.raises(ValueError, match="direct response references"):
        bench.baseline_equivalence_report(current, {"workload_identity": identity, "cases": stripped}, identity)


def test_phase_telemetry_report_requires_variance_and_pin_only():
    fake = _FakeClient()
    result = bench.run_case(fake, bench.PUBLIC_CASES["townsend"], "med", warm_repeats=0)
    assert bench.phase_telemetry_report([result])["ok"] is False
    for endpoint in ("variance", "pin"):
        result["endpoints"][endpoint]["first_in_flow"][0]["phase_ms"] = {"kernel": 1.0}
    assert bench.phase_telemetry_report([result]) == {"ok": True, "missing": []}
