#!/usr/bin/env python3
"""Deterministic end-to-end latency benchmark for the running commute server.

This runner NEVER imports or boots ``scripts/server.py``.  It drives one already-running server
sequentially, which avoids the repository's cross-process numba-cache corruption hazard.  The
default public Townsend/cell-1916 matrix follows the real progressive UI request order:

    /compute -> lightweight /itinerary -> /variance -> /itinerary?pin=1

For each endpoint the first observation is recorded as ``first_in_flow`` and immediate repeats as
``warm``.  ``first_in_flow`` is deliberately not called cold: itinerary intentionally reuses
compute state, pin intentionally reuses itinerary/variance state, the target process may have been
warm before the runner connected, and later speed cases reuse speed-independent destination walking
state.  For a genuinely process-cold measurement of one case, restart the server and run only that
destination/speed; an HTTP client cannot safely clear a live application's caches. Booting with
``PERF_BENCHMARK_STATS=1`` adds aggregate RSS/cache counts to /healthz for correlation.  Without
that opt-in, latency and direct output-equivalence measurements still work and telemetry is absent.

For a controlled current-vs-baseline comparison, run the fixed single-case command below once per
build/state/thread combination.  The first build opts into saving its actual normalized responses;
the second names that artifact and compares objects directly (never by response hash):

    .venv/bin/python scripts/perf_benchmark.py --destinations townsend --speeds med \\
      --warm-repeats 3 --max-rides 8 --expected-svc-date YYYY-MM-DD \\
      --process-state jit-warm-cache-cold --numba-threads 1 --require-phase-telemetry \\
      --run-label baseline-thread-1 --save-response-references --output out/baseline-t1.json
    .venv/bin/python scripts/perf_benchmark.py --destinations townsend --speeds med \\
      --warm-repeats 3 --max-rides 8 --expected-svc-date YYYY-MM-DD \\
      --process-state jit-warm-cache-cold --numba-threads 1 --require-phase-telemetry \\
      --run-label current-thread-1 --baseline-artifact out/baseline-t1.json --output out/current-t1.json

Examples:

    PERF_BENCHMARK_STATS=1 .venv/bin/python scripts/server.py
    .venv/bin/python scripts/perf_benchmark.py
    .venv/bin/python scripts/perf_benchmark.py --destinations townsend,market --speeds med
    .venv/bin/python scripts/perf_benchmark.py --destinations public5 --speeds slow,med,fast --warm-repeats 0
    .venv/bin/python scripts/perf_benchmark.py --browser
    .venv/bin/python scripts/perf_benchmark.py --speeds med --browser --browser-repeats 3

The artifact contains no personal addresses: every destination and cell is a fixed public fixture.
"""

from __future__ import annotations

import argparse
import datetime as dt
import http.client
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit


SCHEMA_VERSION = 4
DEFAULT_BASE_URL = os.environ.get("PERF_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_ARTIFACT = Path(os.environ.get("PERF_ARTIFACT", "out/perf_latency.json"))
DEFAULT_SEED = 20260717
SPEEDS = ("slow", "med", "fast")
DEFAULT_MAX_RIDES = 8
PROCESS_STATES = (
    "unspecified",
    "process-cold",
    "jit-warm-cache-cold",
    "serving-cache-cold",
    "serving-warm",
)
PUBLIC_MATRIX_TOKEN = "public5"
EXPECTED_STACK = {"engine": "raptor", "semantic": "departafter", "walk": "graph"}
CLEAR_STORAGE_SCRIPT = "try { localStorage.clear(); } catch (e) {}"

# Neutral, public destinations only. Cell 1916 is the committed Mission Dolores public hotspot
# used by the route-family E2E suite; keeping one cell fixed makes cross-speed output comparisons
# interpretable and avoids a discovery pass contaminating the first-in-flow endpoint samples.
PUBLIC_CASES = {
    "townsend": {
        "slug": "townsend",
        "label": "650 Townsend St",
        "lat": 37.7714154,
        "lon": -122.4030885,
        "cell_id": "1916",
    },
    "market": {
        "slug": "market",
        "label": "1 Market St",
        "lat": 37.79360,
        "lon": -122.39580,
        "cell_id": "1916",
    },
    "city_hall": {
        "slug": "city_hall",
        "label": "1 Dr Carlton B Goodlett Pl",
        "lat": 37.77930,
        "lon": -122.41920,
        "cell_id": "1916",
    },
    "ferry_building": {
        "slug": "ferry_building",
        "label": "Ferry Building",
        "lat": 37.79549,
        "lon": -122.39366,
        "cell_id": "1916",
    },
    "mission_bay": {
        "slug": "mission_bay",
        "label": "Mission Bay Commons",
        "lat": 37.77027,
        "lon": -122.38794,
        "cell_id": "1916",
    },
}


def _utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def normalized_response(value):
    """Return a directly comparable response without top-level response timing.

    The runner deliberately compares normal Python response objects rather than hashes.  ``ms`` is
    instrumentation, not product data; nested fields named ``ms`` stay intact because they may be
    meaningful response content.
    """
    if not isinstance(value, dict):
        return value
    return {key: child for key, child in value.items() if key != "ms"}


def _parse_server_timing(value):
    """Read standard Server-Timing durations into a compact phase map.

    Production does not emit these headers.  Supporting them here makes a profiling proxy or a
    future opt-in server phase timer useful without adding benchmark fields to product responses.
    Unknown/non-duration metrics are ignored conservatively.
    """
    if not isinstance(value, str):
        return {}
    phases = {}
    for metric in value.split(","):
        parts = [part.strip() for part in metric.split(";") if part.strip()]
        if not parts:
            continue
        name = parts[0]
        for part in parts[1:]:
            key, sep, raw = part.partition("=")
            if key.strip().lower() != "dur" or not sep:
                continue
            try:
                duration = float(raw.strip().strip('"'))
            except ValueError:
                continue
            if math.isfinite(duration) and duration >= 0:
                phases[name] = round(duration, 3)
            break
    return phases


def optional_phase_telemetry(headers):
    """Consume optional timing headers without requiring any production response changes."""
    headers = headers or {}
    phases = _parse_server_timing(headers.get("server-timing"))
    raw_json = headers.get("x-perf-phases")
    if isinstance(raw_json, str):
        try:
            encoded = json.loads(raw_json)
        except json.JSONDecodeError:
            encoded = None
        if isinstance(encoded, dict):
            for key, value in encoded.items():
                if isinstance(key, str) and isinstance(value, (int, float)) and not isinstance(value, bool):
                    if math.isfinite(float(value)) and float(value) >= 0:
                        phases[key] = round(float(value), 3)
    return phases or None


def _count_geometry_points(value):
    if isinstance(value, dict):
        points = value.get("pts")
        own = len(points) if isinstance(points, list) else 0
        return own + sum(_count_geometry_points(v) for k, v in value.items() if k != "pts")
    if isinstance(value, list):
        return sum(_count_geometry_points(v) for v in value)
    return 0


def output_summary(endpoint, body):
    """Small, stable description of a response; full multi-megabyte bodies stay out of artifacts."""
    if not isinstance(body, dict):
        return {"json_type": type(body).__name__}
    if endpoint == "compute":
        cells = body.get("cells") or {}
        return {
            "cells": len(cells),
            "reachable": sum(
                isinstance(v, list) and any(x is not None for x in v) for v in cells.values()
            ),
        }
    if endpoint == "variance":
        variance = body.get("variance") or {}
        return {
            "variance_cells": len(variance),
            "realistic_cells": len(body.get("realistic") or {}),
            "cells_with_alts": sum(bool((v or {}).get("alt")) for v in variance.values()),
        }
    if endpoint in ("itinerary", "pin"):
        options = [body, *(body.get("alts") or [])]
        return {
            "error": body.get("error"),
            "total": body.get("total"),
            "transfers": body.get("xfers"),
            "options": len(options),
            "families": len({
                str((option.get("family") or {}).get("key"))
                for option in options if (option.get("family") or {}).get("key") is not None
            }),
            "geometry_points": _count_geometry_points(body),
        }
    return {"keys": sorted(body)}


def distribution(values):
    values = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not values:
        return None

    def percentile(q):
        if len(values) == 1:
            return values[0]
        pos = (len(values) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        if lo == hi:
            return values[lo]
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)

    def rounded(value):
        return round(float(value), 3)

    return {
        "n": len(values),
        "min": rounded(values[0]),
        "p50": rounded(percentile(0.50)),
        "p90": rounded(percentile(0.90)),
        "p95": rounded(percentile(0.95)),
        "max": rounded(values[-1]),
        "mean": rounded(statistics.fmean(values)),
        "stddev": rounded(statistics.pstdev(values)),
    }


class HttpProbe:
    """One persistent HTTP/1.1 connection so warm samples do not repeatedly pay TCP setup."""

    def __init__(self, base_url, *, timeout_s=120.0, attempts=3):
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(f"base URL must be http(s), got {base_url!r}")
        self.scheme = parsed.scheme
        self.host = parsed.hostname
        self.port = parsed.port
        self.prefix = parsed.path.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.attempts = max(1, int(attempts))
        self._connection = None

    def _new_connection(self):
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=self.timeout_s)

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def request_json(self, path):
        if not path.startswith("/"):
            raise ValueError(f"request path must start with /, got {path!r}")
        full_path = self.prefix + path
        started = time.perf_counter()
        retries = 0
        last = None
        for attempt in range(self.attempts):
            try:
                if self._connection is None:
                    self._connection = self._new_connection()
                self._connection.request(
                    "GET", full_path,
                    headers={"Accept": "application/json", "User-Agent": "sfci-perf-benchmark/1"},
                )
                response = self._connection.getresponse()
                raw = response.read()
                headers = {k.lower(): v for k, v in response.getheaders()}
                status = int(response.status)
                last = (status, headers, raw)
                if status not in (429, 503) or attempt + 1 >= self.attempts:
                    break
                retries += 1
                try:
                    wait_s = float(headers.get("retry-after", "4"))
                except ValueError:
                    wait_s = 4.0
                time.sleep(max(0.05, min(wait_s, 10.0)))
            except (OSError, http.client.HTTPException):
                self.close()
                if attempt + 1 >= self.attempts:
                    raise
                retries += 1
                time.sleep(0.1 * (attempt + 1))
        if last is None:
            raise RuntimeError(f"GET {path} produced no response")
        status, headers, raw = last
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(
                f"GET {path} returned non-JSON status={status}, bytes={len(raw)}"
            ) from exc
        if not 200 <= status < 300:
            raise RuntimeError(f"GET {path} failed status={status}: {str(body)[:500]}")
        return {
            "status": status,
            "client_ms": round(elapsed_ms, 3),
            "response_bytes": len(raw),
            "headers": headers,
            "body": body,
            "retries": retries,
        }


def _local_rss_bytes(pid):
    if not pid:
        return None
    try:
        statm = Path(f"/proc/{int(pid)}/statm").read_text().split()
        return int(statm[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError, AttributeError):
        pass
    try:
        import subprocess
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(int(pid))],
            check=True, capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        return int(out) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _runner_host_snapshot():
    """Best-effort load/CPU context for a run executed on the server host.

    A remote server PID cannot reveal the remote machine's load to this process.  The controlled
    production procedure therefore runs this runner on the serving host; the artifact makes that
    provenance visible instead of silently treating a laptop's load average as server telemetry.
    """
    try:
        load = os.getloadavg()
        loadavg = {name: round(float(value), 3) for name, value in zip(
            ("load1", "load5", "load15"), load
        )}
    except (AttributeError, OSError):
        loadavg = None
    return {
        "hostname": platform.node() or None,
        "logical_cpus": os.cpu_count(),
        "loadavg": loadavg,
    }


def _resource_snapshot(client, server_pid=None):
    health = client.request_json("/healthz")["body"]
    if not isinstance(health, dict) or not health.get("ok"):
        raise RuntimeError(f"target is not ready: {health!r}")
    benchmark = health.get("benchmark") if isinstance(health, dict) else None
    pid = server_pid or ((benchmark or {}).get("pid") if benchmark else None)
    # Only probe a PID explicitly supplied by the user. A PID returned by a remote server has no
    # relationship to a process with the same number on this machine.
    local_rss = _local_rss_bytes(server_pid) if server_pid else None
    return {
        "server": {
            key: health.get(key) for key in ("ok", "engine", "semantic", "walk", "svc_date")
        },
        "uptime_s": health.get("uptime_s"),
        "health_benchmark": benchmark,
        "local_server_pid": int(pid) if server_pid and pid else None,
        "local_rss_bytes": local_rss,
        "runner_host": _runner_host_snapshot(),
    }


def _counts(snapshot):
    return (((snapshot or {}).get("health_benchmark") or {}).get("cache_counts") or {})


def _count_delta(before, after):
    b, a = _counts(before), _counts(after)
    return {key: int(a.get(key, 0)) - int(b.get(key, 0)) for key in sorted(set(b) | set(a))}


def _rss_value(snapshot):
    local = (snapshot or {}).get("local_rss_bytes")
    if local is not None:
        return local
    return ((snapshot or {}).get("health_benchmark") or {}).get("rss_bytes")


def _health_pid(snapshot):
    return ((snapshot or {}).get("health_benchmark") or {}).get("pid")


class HealthGuard:
    """Reject wrong-stack or restarted targets before their timings enter an artifact.

    PID is available when PERF_BENCHMARK_STATS is enabled. Without it, monotonically nondecreasing
    uptime is the best remote restart signal /healthz exposes. Configuration stability is enforced
    even when ``allow_alternate_stack`` waives the default RAPTOR/depart-after/graph requirement.
    """

    def __init__(self, *, allow_alternate_stack=False, expected_svc_date=None):
        self.allow_alternate_stack = bool(allow_alternate_stack)
        self.expected_svc_date = str(expected_svc_date) if expected_svc_date else None
        self.server = None
        self.pid = None
        self.last_uptime_s = None

    def observe(self, snapshot):
        server = dict((snapshot or {}).get("server") or {})
        actual_stack = {key: server.get(key) for key in EXPECTED_STACK}
        if not self.allow_alternate_stack and actual_stack != EXPECTED_STACK:
            raise RuntimeError(
                f"benchmark requires {EXPECTED_STACK}, got {actual_stack}; "
                "use --allow-alternate-stack only for an intentional comparison"
            )
        if self.expected_svc_date and server.get("svc_date") != self.expected_svc_date:
            raise RuntimeError(
                f"benchmark requires service date {self.expected_svc_date}, "
                f"got {server.get('svc_date')!r}"
            )
        if self.server is None:
            self.server = server
        elif server != self.server:
            raise RuntimeError(
                f"server configuration changed during benchmark: {self.server} -> {server}"
            )

        uptime = (snapshot or {}).get("uptime_s")
        if not isinstance(uptime, (int, float)) or isinstance(uptime, bool) or uptime < 0:
            raise RuntimeError(f"healthz returned invalid uptime_s: {uptime!r}")
        if self.last_uptime_s is not None and uptime < self.last_uptime_s:
            raise RuntimeError(
                f"server uptime reset during benchmark: {self.last_uptime_s} -> {uptime}"
            )
        self.last_uptime_s = uptime

        pid = _health_pid(snapshot)
        if pid is not None:
            try:
                pid = int(pid)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"healthz returned invalid benchmark pid: {pid!r}") from exc
            if self.pid is None:
                self.pid = pid
            elif pid != self.pid:
                raise RuntimeError(f"server PID changed during benchmark: {self.pid} -> {pid}")
        return snapshot


def validate_endpoint_contract(endpoint, body, case):
    """Fail stable-but-empty/error responses that direct equivalence cannot detect."""
    cell_id = str(case["cell_id"])
    if not isinstance(body, dict):
        raise RuntimeError(f"{endpoint} returned {type(body).__name__}, expected object")

    if endpoint == "compute":
        cells = body.get("cells")
        value = cells.get(cell_id) if isinstance(cells, dict) else None
        if not isinstance(cells, dict) or not cells:
            raise RuntimeError("compute contract failed: cells is empty or absent")
        if not isinstance(value, list) or not any(item is not None for item in value):
            raise RuntimeError(f"compute contract failed: selected cell {cell_id} is not reachable")
        return

    if endpoint == "variance":
        variance = body.get("variance")
        if not isinstance(variance, dict) or not variance:
            raise RuntimeError("variance contract failed: variance is empty or absent")
        if not isinstance(variance.get(cell_id), dict):
            raise RuntimeError(f"variance contract failed: selected cell {cell_id} is absent")
        return

    if endpoint in ("itinerary", "pin"):
        if body.get("error"):
            raise RuntimeError(f"{endpoint} contract failed: {body.get('error')}")
        if not isinstance(body.get("total"), (int, float)):
            raise RuntimeError(f"{endpoint} contract failed: numeric total is absent")
        if _count_geometry_points(body.get("geom")) <= 0:
            raise RuntimeError(f"{endpoint} contract failed: primary route geometry is empty")
        for key in ("family", "branch"):
            if not isinstance(body.get(key), dict) or not body[key].get("key"):
                raise RuntimeError(f"{endpoint} contract failed: {key} identity is absent")
        if endpoint == "pin":
            if not isinstance(body.get("alts"), list):
                raise RuntimeError("pin contract failed: alternatives collection is absent")
            if not isinstance(body.get("frag"), (int, float)):
                raise RuntimeError("pin contract failed: pin-only fragility enrichment is absent")
        return

    raise RuntimeError(f"unknown benchmark endpoint contract: {endpoint}")


def endpoint_paths(case, speed, *, max_rides=DEFAULT_MAX_RIDES):
    """Return the explicit, fixed settings used for one controlled workload.

    The live application defaults an omitted ``maxrides`` to the current "Any" setting.  Sending
    it explicitly prevents a later server-default change from making two artifacts look comparable
    when they exercised different routing constraints.
    """
    common = {
        "dlat": case["lat"], "dlon": case["lon"], "speed": speed,
        "maxrides": int(max_rides),
    }
    itinerary = {"id": case["cell_id"], **common}
    return (
        ("compute", "/compute?" + urlencode({
            "lat": case["lat"], "lon": case["lon"], "speed": speed,
            "maxrides": int(max_rides),
        })),
        ("itinerary", "/itinerary?" + urlencode(itinerary)),
        ("variance", "/variance?" + urlencode(common)),
        ("pin", "/itinerary?" + urlencode({**itinerary, "pin": 1})),
    )


def _sample(client, endpoint, path, state, ordinal, case):
    result = client.request_json(path)
    body = result.pop("body")
    headers = result.pop("headers", None)
    validate_endpoint_contract(endpoint, body, case)
    response_ms = body.get("ms") if isinstance(body, dict) else None
    response_ms = float(response_ms) if isinstance(response_ms, (int, float)) else None
    client_ms = float(result["client_ms"])
    return {
        "cache_state": state,
        "ordinal": ordinal,
        **result,
        # ``response_ms`` is the server-reported top-level timing when present.  The remainder
        # intentionally remains signed: a negative value tells us the two timers cover different
        # scopes rather than silently pretending the transport was free.
        "response_ms": response_ms,
        "transport_remainder_ms": (
            round(client_ms - response_ms, 3) if response_ms is not None else None
        ),
        "phase_ms": optional_phase_telemetry(headers),
        "output": output_summary(endpoint, body),
        "_normalized_response": normalized_response(body),
    }


def run_case(client, case, speed, *, warm_repeats, max_rides=DEFAULT_MAX_RIDES,
             capture_response_references=False, server_pid=None, health_guard=None,
             flow_ordinal=0):
    health_guard = health_guard or HealthGuard()
    endpoints = {}
    for endpoint, path in endpoint_paths(case, speed, max_rides=max_rides):
        before = health_guard.observe(_resource_snapshot(client, server_pid))
        first_in_flow = [_sample(client, endpoint, path, "first_in_flow", 0, case)]
        after_first = health_guard.observe(_resource_snapshot(client, server_pid))
        warm = [
            _sample(client, endpoint, path, "warm", i + 1, case)
            for i in range(int(warm_repeats))
        ]
        after_warm = health_guard.observe(_resource_snapshot(client, server_pid))
        samples = first_in_flow + warm
        normalized = [sample.pop("_normalized_response") for sample in samples]
        baseline = normalized[0]
        unequal_ordinals = [
            sample["ordinal"] for sample, candidate in zip(samples, normalized)
            if candidate != baseline
        ]
        endpoint_result = {
            "path": path,
            "first_in_flow": first_in_flow,
            "warm": warm,
            "output_equivalent": not unequal_ordinals,
            # Bodies are compared directly above.  Summaries make a drift actionable without
            # persisting multi-megabyte isochrone payloads unless the caller explicitly opts in
            # below to retain a named baseline reference.
            "output_comparison": {
                "method": "direct_normalized_object_equality",
                "ignored_top_level_fields": ["ms"],
                "unequal_ordinals": unequal_ordinals,
                "sample_outputs": [sample["output"] for sample in samples],
            },
            "resources": {
                "before_first_in_flow": before,
                "after_first_in_flow": after_first,
                "after_warm": after_warm,
                "first_in_flow_cache_delta": _count_delta(before, after_first),
                "warm_cache_delta": _count_delta(after_first, after_warm),
                "first_in_flow_rss_delta_bytes": (
                    _rss_value(after_first) - _rss_value(before)
                    if _rss_value(after_first) is not None and _rss_value(before) is not None else None
                ),
                "warm_rss_delta_bytes": (
                    _rss_value(after_warm) - _rss_value(after_first)
                    if _rss_value(after_warm) is not None and _rss_value(after_first) is not None else None
                ),
            },
        }
        # This is deliberately opt-in: a normalized variance body is large.  When explicitly
        # requested, persist the real object (not a hash/signature) so a later named build can be
        # compared with ordinary Python equality on the exact same workload.
        if capture_response_references:
            endpoint_result["direct_normalized_reference"] = baseline
        endpoints[endpoint] = endpoint_result
    return {
        "destination": {k: case[k] for k in ("slug", "label", "lat", "lon")},
        "cell_id": case["cell_id"],
        "speed": speed,
        "max_rides": int(max_rides),
        "flow_ordinal": int(flow_ordinal),
        "first_in_flow_dependency": (
            "target preexisting cache state; endpoint order reuses all preceding endpoints"
            if not flow_ordinal else
            "target preexisting cache state plus prior matrix cases; endpoint order reuses all preceding endpoints"
        ),
        "endpoints": endpoints,
    }


def aggregate_cases(cases):
    aggregate = {}
    for endpoint in ("compute", "itinerary", "variance", "pin"):
        aggregate[endpoint] = {}
        for state in ("first_in_flow", "warm"):
            samples = [
                sample
                for case in cases
                for sample in case["endpoints"][endpoint][state]
            ]
            aggregate[endpoint][state] = {
                "client_ms": distribution(s["client_ms"] for s in samples),
                "response_ms": distribution(s["response_ms"] for s in samples),
                "transport_remainder_ms": distribution(
                    s["transport_remainder_ms"] for s in samples
                ),
                "response_bytes": distribution(s["response_bytes"] for s in samples),
                "retries": sum(s["retries"] for s in samples),
                "phase_ms": {
                    phase: distribution(
                        (sample.get("phase_ms") or {}).get(phase) for sample in samples
                    )
                    for phase in sorted({
                        phase for sample in samples for phase in (sample.get("phase_ms") or {})
                    })
                } or None,
            }
    return aggregate


def equivalence_report(cases):
    mismatches = []
    for case in cases:
        for endpoint, result in case["endpoints"].items():
            if not result["output_equivalent"]:
                mismatches.append({
                    "destination": case["destination"]["slug"],
                    "speed": case["speed"],
                    "cell_id": case["cell_id"],
                    "endpoint": endpoint,
                    "comparison": result["output_comparison"],
                })
    return {"ok": not mismatches, "mismatches": mismatches}


def workload_identity(cases, health_before):
    """The fields which must match before two build artifacts are comparable.

    Process state and Numba thread count deliberately do *not* belong here: the Wave 0 matrix
    compares those intentionally.  Destination, cell, speed, ride limit and GTFS service date do.
    """
    return {
        "service_date": (health_before.get("server") or {}).get("svc_date"),
        "cases": [
            {
                "destination": case["destination"],
                "cell_id": case["cell_id"],
                "speed": case["speed"],
                "max_rides": case["max_rides"],
            }
            for case in cases
        ],
    }


def phase_telemetry_report(cases):
    """Report coverage for the endpoints whose server code emits opt-in phase telemetry."""
    missing = []
    for case in cases:
        for endpoint in ("variance", "pin"):
            samples = case["endpoints"][endpoint]["first_in_flow"]
            if not any(sample.get("phase_ms") for sample in samples):
                missing.append({
                    "destination": case["destination"]["slug"],
                    "speed": case["speed"],
                    "cell_id": case["cell_id"],
                    "endpoint": endpoint,
                })
    return {"ok": not missing, "missing": missing}


def baseline_equivalence_report(cases, baseline_artifact, current_identity):
    """Directly compare current normalized responses with an opted-in named baseline artifact.

    No digest is involved: the baseline artifact carries the actual normalized JSON objects only
    when its author explicitly requested them.  Refuse ambiguous cross-workload comparisons
    instead of producing deceptively precise timing deltas.
    """
    if not isinstance(baseline_artifact, dict):
        raise ValueError("baseline artifact must be a JSON object")
    baseline_identity = baseline_artifact.get("workload_identity")
    if baseline_identity != current_identity:
        raise ValueError(
            "baseline workload identity differs (destination/cell/speed/max-rides/service date); "
            "do not compare unlike runs"
        )
    baseline_cases = baseline_artifact.get("cases")
    if not isinstance(baseline_cases, list):
        raise ValueError("baseline artifact has no cases; re-run it with --save-response-references")
    baseline_by_key = {
        (str(case.get("destination", {}).get("slug")), str(case.get("cell_id")),
         str(case.get("speed")), case.get("max_rides")): case
        for case in baseline_cases
    }
    mismatches = []
    for case in cases:
        key = (
            str(case["destination"]["slug"]), str(case["cell_id"]), str(case["speed"]),
            case["max_rides"],
        )
        baseline_case = baseline_by_key.get(key)
        if baseline_case is None:
            raise ValueError(f"baseline artifact is missing controlled case {key}")
        for endpoint in ("compute", "itinerary", "variance", "pin"):
            current_reference = case["endpoints"][endpoint].get("direct_normalized_reference")
            baseline_reference = (
                (baseline_case.get("endpoints") or {}).get(endpoint, {})
                .get("direct_normalized_reference")
            )
            if current_reference is None or baseline_reference is None:
                raise ValueError(
                    "baseline comparison requires direct response references; re-run the baseline "
                    "with --save-response-references"
                )
            if current_reference != baseline_reference:
                mismatches.append({
                    "destination": key[0], "cell_id": key[1], "speed": key[2],
                    "max_rides": key[3], "endpoint": endpoint,
                    "comparison": "direct_normalized_object_equality",
                })
    return {"ok": not mismatches, "mismatches": mismatches}


def _frontend_transfer_mode(max_rides):
    """Map the server's explicit ride cap to the page's stable transfer-mode URL setting."""
    mapping = {1: "0", 2: "1", 3: "2", DEFAULT_MAX_RIDES: "any"}
    try:
        return mapping[int(max_rides)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "browser mode supports max rides 1, 2, 3, or 8 (the page's 0/1/2/Any controls)"
        ) from exc


def run_browser_tuc(base_url, matrix, *, max_rides=DEFAULT_MAX_RIDES, repeats=1):
    """Optional desktop+mobile product-ready timings; Playwright is imported lazily.

    Each repeat gets fresh browser storage but intentionally hits the server after the API matrix,
    so these are warm-server user-visible timings. Markers follow the actual interaction contract:
    visible hover route, completed touch preview, immediately useful inspector, then committed pin
    enrichment. The fixed public cell is centered before input so coordinate hit-testing is stable.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "--browser requires playwright; install the existing E2E dependencies first"
        ) from exc

    def _url(case, speed):
        label = quote(case["label"], safe="")
        return (
            f"{base_url.rstrip('/')}/#wp={case['lat']:.7f},{case['lon']:.7f},{label}"
            f"&metric=r&cmode=time&colors=on&mt={_frontend_transfer_mode(max_rides)}&sp={speed}&th=auto"
        )

    def _open(context, case, speed):
        page = context.new_page()
        # A string init script is executed as a script body. The previous arrow-function expression
        # merely constructed a function and never invoked it.
        page.add_init_script(CLEAR_STORAGE_SCRIPT)
        wall_start = time.perf_counter()
        page.goto(_url(case, speed), wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_function(
            "() => document.querySelector('#dest').textContent.trim().length > 0 && "
            "document.querySelectorAll('#list .nb').length > 0",
            timeout=60_000,
        )
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        metrics = page.evaluate(
            """() => {
              const entries=performance.getEntriesByType('resource');
              const compute=[...entries].reverse().find(e=>e.name.includes('/compute?'));
              return {navigation_to_useful_ms:performance.now(),
                      compute_resource_ms:compute?compute.duration:null,
                      neighborhoods:document.querySelectorAll('#list .nb').length};
            }"""
        )
        return page, {
            "wall_to_useful_ms": round(wall_ms, 3),
            "navigation_to_useful_ms": round(float(metrics["navigation_to_useful_ms"]), 3),
            "compute_resource_ms": (
                round(float(metrics["compute_resource_ms"]), 3)
                if metrics["compute_resource_ms"] is not None else None
            ),
            "neighborhoods": metrics["neighborhoods"],
        }

    def _cell_point(page, cell_id):
        found = page.evaluate(
            """id => {
              let target=null;
              layer.eachLayer(candidate=>{
                if(String(candidate.feature?.properties?.id)===String(id))target=candidate;
              });
              if(!target)return false;
              map.setView(target.getBounds().getCenter(),13,{animate:false});
              return true;
            }""",
            cell_id,
        )
        if not found:
            raise RuntimeError(f"browser could not find public cell {cell_id}")
        page.wait_for_timeout(100)
        return page.evaluate(
            """id => {
              let target=null;
              layer.eachLayer(candidate=>{
                if(String(candidate.feature?.properties?.id)===String(id))target=candidate;
              });
              const point=map.latLngToContainerPoint(target.getBounds().getCenter());
              const rect=document.getElementById('map').getBoundingClientRect();
              return {x:rect.left+point.x,y:rect.top+point.y};
            }""",
            cell_id,
        )

    rows = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for case, speed in matrix:
                for ordinal in range(int(repeats)):
                    desktop = browser.new_context(viewport={"width": 1280, "height": 800})
                    try:
                        page, desktop_nav = _open(desktop, case, speed)
                        point = _cell_point(page, case["cell_id"])
                        page.evaluate("() => { window.__perfHoverStart=performance.now(); }")
                        page.mouse.move(point["x"], point["y"])
                        page.wait_for_function(
                            """id => String(DRAWN?.id)===String(id) &&
                              !!document.querySelector('.leaflet-tooltip.tt:not(.tt-pending) .bd')""",
                            arg=case["cell_id"], timeout=60_000,
                        )
                        hover_ms = page.evaluate(
                            "() => performance.now()-window.__perfHoverStart"
                        )
                    finally:
                        desktop.close()

                    mobile = browser.new_context(
                        viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True,
                    )
                    try:
                        page, mobile_nav = _open(mobile, case, speed)
                        point = _cell_point(page, case["cell_id"])
                        page.evaluate("() => { window.__perfTapStart=performance.now(); }")
                        page.touchscreen.tap(point["x"], point["y"])
                        page.wait_for_function(
                            """id => document.getElementById('touchpeek').classList.contains('open') &&
                              !document.getElementById('peekbody').textContent.includes('Loading commute') &&
                              String(DRAWN?.id)===String(id)""",
                            arg=case["cell_id"], timeout=60_000,
                        )
                        tap_preview_ms = page.evaluate(
                            "() => performance.now()-window.__perfTapStart"
                        )
                        page.evaluate("() => { window.__perfInspectStart=performance.now(); }")
                        page.locator("#peekinspect").tap()
                        page.wait_for_function(
                            """id => document.getElementById('pincard').classList.contains('open') &&
                              document.querySelectorAll('#pincard .route-choice').length>0 &&
                              !document.getElementById('pinbody').textContent.toLowerCase().includes('loading route') &&
                              String(DRAWN?.id)===String(id)""",
                            arg=case["cell_id"], timeout=60_000,
                        )
                        inspect_light_ms = page.evaluate(
                            "() => performance.now()-window.__perfInspectStart"
                        )
                        page.wait_for_function(
                            """() => routePin!=null && BDCACHE.get(routePin)?._pin===true &&
                              !BDCACHE.get(routePin)?._pinPending""",
                            timeout=60_000,
                        )
                        inspect_enriched_ms = page.evaluate(
                            "() => performance.now()-window.__perfInspectStart"
                        )
                    finally:
                        mobile.close()

                    rows.append({
                        "destination": case["slug"], "speed": speed, "ordinal": ordinal,
                        "desktop_wall_to_useful_ms": desktop_nav["wall_to_useful_ms"],
                        "desktop_navigation_to_useful_ms": desktop_nav["navigation_to_useful_ms"],
                        "desktop_compute_resource_ms": desktop_nav["compute_resource_ms"],
                        "desktop_hover_to_route_ms": round(float(hover_ms), 3),
                        "desktop_neighborhoods": desktop_nav["neighborhoods"],
                        "mobile_wall_to_useful_ms": mobile_nav["wall_to_useful_ms"],
                        "mobile_navigation_to_useful_ms": mobile_nav["navigation_to_useful_ms"],
                        "mobile_compute_resource_ms": mobile_nav["compute_resource_ms"],
                        "mobile_tap_to_preview_route_ms": round(float(tap_preview_ms), 3),
                        "mobile_inspect_to_light_route_ms": round(float(inspect_light_ms), 3),
                        "mobile_inspect_to_pin_enriched_ms": round(float(inspect_enriched_ms), 3),
                        "mobile_neighborhoods": mobile_nav["neighborhoods"],
                    })
        finally:
            browser.close()
    metrics = (
        "desktop_wall_to_useful_ms", "desktop_navigation_to_useful_ms",
        "desktop_compute_resource_ms", "desktop_hover_to_route_ms",
        "mobile_wall_to_useful_ms", "mobile_navigation_to_useful_ms",
        "mobile_compute_resource_ms", "mobile_tap_to_preview_route_ms",
        "mobile_inspect_to_light_route_ms", "mobile_inspect_to_pin_enriched_ms",
    )
    return {
        "server_cache_state": "warm_after_api_matrix",
        "ready_markers": {
            "desktop_hover": "visible non-pending tooltip plus DRAWN route for target cell",
            "mobile_preview": "touch preview loaded plus DRAWN route for target cell",
            "mobile_inspector": "route choice row plus DRAWN route without loading shell",
            "mobile_enriched": "BDCACHE selected response has _pin=true and _pinPending=false",
        },
        "samples": rows,
        "distributions": {
            metric: distribution(row[metric] for row in rows) for metric in metrics
        },
    }


def _csv(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument(
        "--destinations", default="townsend",
        help=(f"comma-separated public fixtures: {','.join(PUBLIC_CASES)}; use {PUBLIC_MATRIX_TOKEN} "
              f"for the bounded five-destination matrix (default: townsend)"),
    )
    parser.add_argument(
        "--speeds", default="slow,med,fast",
        help="comma-separated walk speeds slow,med,fast (default: all three)",
    )
    parser.add_argument(
        "--max-rides", type=int, default=DEFAULT_MAX_RIDES,
        help=("explicit server maxrides setting (1-8; default: 8/Any). It is sent to every "
              "endpoint so controlled artifacts cannot inherit a changed server default."),
    )
    parser.add_argument(
        "--warm-repeats", type=int, default=3,
        help=("immediate repeats per endpoint (default: 3). Use 0 for the full public5 x "
              "three-speed matrix, which has enough first-in-flow observations for p50/p90."),
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="per-request timeout seconds")
    parser.add_argument("--attempts", type=int, default=3, help="HTTP retry attempts for 429/503")
    parser.add_argument(
        "--server-pid", type=int,
        help="optional LOCAL server PID for current RSS when PERF_BENCHMARK_STATS is unavailable",
    )
    parser.add_argument("--browser", action="store_true", help="also measure browser time-to-useful")
    parser.add_argument("--browser-repeats", type=int, default=1)
    parser.add_argument(
        "--process-state", choices=PROCESS_STATES, default="unspecified",
        help=("operator-attested server state before the first request. The runner cannot clear a "
              "live application's caches; use a new server process for process-cold and record "
              "jit-warm-cache-cold only after the documented warmup has completed."),
    )
    parser.add_argument(
        "--numba-threads", type=int,
        help=("operator-attested NUMBA_NUM_THREADS used to boot the target. This records the "
              "1-vs-2 controlled matrix; it does not mutate a running server."),
    )
    parser.add_argument("--run-label", default="unnamed",
                        help="human-readable build/configuration label retained in the artifact")
    parser.add_argument(
        "--expected-svc-date",
        help="fail unless /healthz reports this exact YYYY-MM-DD service date",
    )
    parser.add_argument(
        "--require-phase-telemetry", action="store_true",
        help=("fail unless first-in-flow /variance and pin responses carry opt-in phase headers; "
              "boot the server with PERF_BENCHMARK_STATS=1"),
    )
    parser.add_argument(
        "--save-response-references", action="store_true",
        help=("persist full normalized responses for direct equality against a later named build. "
              "This can make artifacts large; it never writes response hashes/signatures."),
    )
    parser.add_argument(
        "--baseline-artifact", type=Path,
        help=("named artifact created with --save-response-references. Its fixed workload must "
              "match exactly; current responses are compared by direct object equality."),
    )
    parser.add_argument(
        "--allow-alternate-stack", action="store_true",
        help=("allow a non-RAPTOR/depart-after/graph target for intentional comparisons; "
              "configuration and process stability plus endpoint contracts are still enforced"),
    )
    parser.add_argument(
        "--allow-output-drift", action="store_true",
        help="write the artifact but exit zero if direct normalized response comparison differs",
    )
    return parser.parse_args(argv)


def _validated_matrix(args):
    destination_names, speeds = _csv(args.destinations), _csv(args.speeds)
    max_rides = getattr(args, "max_rides", DEFAULT_MAX_RIDES)
    numba_threads = getattr(args, "numba_threads", None)
    if destination_names == [PUBLIC_MATRIX_TOKEN]:
        destination_names = list(PUBLIC_CASES)
    unknown_dest = [name for name in destination_names if name not in PUBLIC_CASES]
    unknown_speed = [speed for speed in speeds if speed not in SPEEDS]
    if not destination_names or unknown_dest:
        raise ValueError(f"unknown/empty destinations: {unknown_dest or destination_names}")
    if not speeds or unknown_speed:
        raise ValueError(f"unknown/empty speeds: {unknown_speed or speeds}")
    if args.warm_repeats < 0:
        raise ValueError("--warm-repeats must be >= 0")
    if not 1 <= int(max_rides) <= DEFAULT_MAX_RIDES:
        raise ValueError(f"--max-rides must be in [1, {DEFAULT_MAX_RIDES}]")
    if numba_threads is not None and numba_threads < 1:
        raise ValueError("--numba-threads must be >= 1")
    if args.browser_repeats < 1:
        raise ValueError("--browser-repeats must be >= 1")
    # Preserve CLI order but remove duplicates deterministically.
    destination_names = list(dict.fromkeys(destination_names))
    speeds = list(dict.fromkeys(speeds))
    if len(destination_names) > len(PUBLIC_CASES):
        raise ValueError(f"at most {len(PUBLIC_CASES)} public destinations are supported")
    matrix = [(PUBLIC_CASES[name], speed) for name in destination_names for speed in speeds]
    # /variance is intentionally limited to 20/minute. Keep two calls of headroom for a monitor or
    # a recently-open browser instead of turning an oversized matrix into misleading retry latency.
    # Browser mode opens one desktop and one mobile page per repeat; each page starts /variance.
    # A full five-destination/three-speed baseline therefore uses no warm repeats (15 calls) and
    # remains inside this conservative rate-limit budget while still yielding aggregate p50/p90.
    browser_variance_calls = 2 * args.browser_repeats if args.browser else 0
    variance_calls = len(matrix) * (1 + args.warm_repeats + browser_variance_calls)
    if variance_calls > 18:
        raise ValueError(
            f"matrix would issue {variance_calls} /variance calls (>18 safe budget); "
            "reduce destinations, speeds, warm repeats, or browser repeats"
        )
    return matrix


def main(argv=None):
    args = parse_args(argv)
    try:
        matrix = _validated_matrix(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    baseline_artifact = None
    if args.baseline_artifact is not None:
        try:
            baseline_artifact = json.loads(args.baseline_artifact.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read --baseline-artifact {args.baseline_artifact}: {exc}") from exc

    client = HttpProbe(args.base_url, timeout_s=args.timeout, attempts=args.attempts)
    started = time.perf_counter()
    health_guard = HealthGuard(
        allow_alternate_stack=args.allow_alternate_stack,
        expected_svc_date=args.expected_svc_date,
    )
    capture_response_references = bool(args.save_response_references or baseline_artifact is not None)
    try:
        health_before = health_guard.observe(_resource_snapshot(client, args.server_pid))
        cases = [
            run_case(
                client, case, speed, warm_repeats=args.warm_repeats,
                max_rides=args.max_rides,
                capture_response_references=capture_response_references,
                server_pid=args.server_pid, health_guard=health_guard, flow_ordinal=ordinal,
            )
            for ordinal, (case, speed) in enumerate(matrix)
        ]
        health_after = health_guard.observe(_resource_snapshot(client, args.server_pid))
    finally:
        client.close()

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "seed": DEFAULT_SEED,
        "target": args.base_url.rstrip("/"),
        "runner": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
        "method": {
            "request_order": ["compute", "itinerary", "variance", "pin"],
            "first_in_flow_definition": (
                "first endpoint observation at its progressive-flow position; not process-cold"
            ),
            "dependency_note": (
                "target may start warm; itinerary reuses compute; pin reuses itinerary and variance; "
                "later matrix cases may reuse speed-independent destination walking state"
            ),
            "warm_definition": (
                "repeat on one persistent HTTP connection after a non-mutating health snapshot"
            ),
            "warm_repeats": args.warm_repeats,
            "planned_variance_calls": len(matrix) * (
                1 + args.warm_repeats + (2 * args.browser_repeats if args.browser else 0)
            ),
            "sequential": True,
            "production_default_behavior_changed": False,
            "stats_opt_in": bool((health_before.get("health_benchmark") or {})),
            "alternate_stack_allowed": bool(args.allow_alternate_stack),
            "phase_telemetry_required": bool(args.require_phase_telemetry),
            "response_references_saved": capture_response_references,
        },
        "measurement": {
            "run_label": args.run_label,
            "process_state": args.process_state,
            "numba_threads": args.numba_threads,
            "thread_count_source": "operator_attested" if args.numba_threads is not None else None,
            "expected_service_date": args.expected_svc_date,
            "server_boot_note": (
                "The runner does not clear live caches or mutate server threads; state/thread labels "
                "describe how the target was booted before this sequential run."
            ),
        },
        "resources": {
            "before": health_before,
            "after": health_after,
            "cache_delta": _count_delta(health_before, health_after),
            "rss_delta_bytes": (
                _rss_value(health_after) - _rss_value(health_before)
                if _rss_value(health_after) is not None and _rss_value(health_before) is not None
                else None
            ),
        },
        "cases": cases,
        "workload_identity": workload_identity(cases, health_before),
        "aggregates": aggregate_cases(cases),
        "equivalence": equivalence_report(cases),
        "phase_telemetry": phase_telemetry_report(cases),
        "baseline_equivalence": None,
        "browser": None,
    }
    if baseline_artifact is not None:
        try:
            artifact["baseline_equivalence"] = baseline_equivalence_report(
                cases, baseline_artifact, artifact["workload_identity"],
            )
        except ValueError as exc:
            raise SystemExit(f"baseline comparison refused: {exc}") from exc
    if args.browser:
        artifact["browser"] = run_browser_tuc(
            args.base_url, matrix, max_rides=args.max_rides, repeats=args.browser_repeats,
        )
        # Browser interactions are part of the run: validate the same process/config survived them
        # and make the final resource delta include the browser-created cache entries.
        verifier = HttpProbe(args.base_url, timeout_s=args.timeout, attempts=args.attempts)
        try:
            health_after = health_guard.observe(_resource_snapshot(verifier, args.server_pid))
        finally:
            verifier.close()
        artifact["resources"].update({
            "after": health_after,
            "cache_delta": _count_delta(health_before, health_after),
            "rss_delta_bytes": (
                _rss_value(health_after) - _rss_value(health_before)
                if _rss_value(health_after) is not None and _rss_value(health_before) is not None
                else None
            ),
        })
        artifact["runner"]["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    print(f"wrote {args.output}")
    for endpoint, states in artifact["aggregates"].items():
        first = states["first_in_flow"]
        warm = states["warm"]
        first_client = first["client_ms"]
        first_response = first["response_ms"]
        first_remainder = first["transport_remainder_ms"]
        warm_client = warm["client_ms"]
        print(
            f"{endpoint:10s} first client p50/p90="
            f"{first_client['p50'] if first_client else 'n/a'}/"
            f"{first_client['p90'] if first_client else 'n/a'}ms "
            f"response p50={first_response['p50'] if first_response else 'n/a'}ms "
            f"transport p50={first_remainder['p50'] if first_remainder else 'n/a'}ms "
            f"warm client p50={warm_client['p50'] if warm_client else 'n/a'}ms"
        )
    output_drift = not artifact["equivalence"]["ok"]
    baseline_drift = bool(
        artifact["baseline_equivalence"] is not None
        and not artifact["baseline_equivalence"]["ok"]
    )
    phase_missing = args.require_phase_telemetry and not artifact["phase_telemetry"]["ok"]
    if output_drift:
        print("ERROR: direct normalized responses differ within the run; see artifact", file=sys.stderr)
    if baseline_drift:
        print("ERROR: current normalized responses differ from named baseline; see artifact", file=sys.stderr)
    if phase_missing:
        print("ERROR: required phase telemetry is absent; see artifact", file=sys.stderr)
    if output_drift or baseline_drift or phase_missing:
        return 0 if args.allow_output_drift else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
