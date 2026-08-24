"""Focused contract tests for the direct server readiness probes.

The production server fixture performs the canonical RAPTOR boot.  These tests deliberately
replace only its immutable cached result after import, so endpoint assertions remain independent
of the current feed window while still exercising the real Flask routes.
"""

from __future__ import annotations

import pytest

from core import readiness


HEALTH_PATHS = ("/livez", "/readyz", "/healthz")


@pytest.fixture(scope="session")
def server():
    """Use the shared runtime module when its heavyweight canonical boot is available.

    CI/deployment tests provide the access bake and optional Flask-Limiter dependency.  A source
    checkout without those runtime inputs should still collect this contract suite cleanly; the
    endpoint tests become runnable as soon as the canonical bake is rebuilt.
    """
    try:
        import server as runtime  # noqa: WPS433
    except (ImportError, FileNotFoundError) as exc:
        pytest.skip(f"canonical server boot unavailable: {exc}")
    return runtime


def _result(ready: bool = True, reason: str = "ok", detail: str | None = None):
    return readiness.ReadinessResult(
        ready=ready,
        reason_code=reason,
        service_date="20260819",
        detail=detail,
    )


def test_livez_readyz_healthz_healthy_get_head_and_no_store(client, server, monkeypatch):
    monkeypatch.delenv("PERF_BENCHMARK_STATS", raising=False)
    monkeypatch.setattr(server, "_READINESS", _result())

    live = client.get("/livez")
    assert live.status_code == 200
    assert live.get_json() == {"ok": True, "reason_code": "process_alive"}
    assert "process_alive" not in readiness.REASON_CODES
    assert live.headers["Cache-Control"] == "no-store"

    ready = client.get("/readyz")
    health = client.get("/healthz")
    assert ready.status_code == health.status_code == 200
    ready_body = ready.get_json()
    health_body = health.get_json()
    assert isinstance(ready_body.pop("uptime_s"), (int, float))
    assert isinstance(health_body.pop("uptime_s"), (int, float))
    assert ready_body == health_body == {
        "engine": "raptor",
        "ok": True,
        "reason_code": "ok",
        "semantic": server.RAPTOR_SEMANTIC,
        "service_date": "20260819",
        "svc_date": "2026-08-19",
        "walk": "graph",
    }
    for path in HEALTH_PATHS:
        response = client.head(path)
        assert response.status_code == 200
        assert response.data == b""
        assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("reason", sorted(readiness.REASON_CODES))
def test_readyz_exposes_each_safe_reason_code_and_status(client, server, monkeypatch, reason):
    monkeypatch.delenv("PERF_BENCHMARK_STATS", raising=False)
    healthy = reason == readiness.OK
    monkeypatch.setattr(server, "_READINESS", _result(healthy, reason))

    response = client.get("/readyz")
    assert response.status_code == (200 if healthy else 503)
    assert response.get_json()["reason_code"] == reason
    assert response.headers["Cache-Control"] == "no-store"


def test_healthz_is_exact_readyz_compatibility_alias_when_not_ready(client, server, monkeypatch):
    monkeypatch.delenv("PERF_BENCHMARK_STATS", raising=False)
    cached = _result(False, "missing_feed", "bart")
    monkeypatch.setattr(server, "_READINESS", cached)

    ready = client.get("/readyz")
    health = client.get("/healthz")
    assert ready.status_code == health.status_code == 503
    ready_body, health_body = ready.get_json(), health.get_json()
    assert isinstance(ready_body.pop("uptime_s"), (int, float))
    assert isinstance(health_body.pop("uptime_s"), (int, float))
    assert ready_body == health_body
    assert ready.get_json()["detail"] == "bart"  # only approved role labels are serializable


def test_unknown_reason_and_unsafe_detail_are_redacted(client, server, monkeypatch):
    monkeypatch.delenv("PERF_BENCHMARK_STATS", raising=False)
    cached = _result(False, "secret parser error /private/feed.zip", "/private/feed.zip")
    monkeypatch.setattr(server, "_READINESS", cached)

    for path in ("/readyz", "/healthz"):
        response = client.get(path)
        assert response.status_code == 503
        body = response.get_json()
        assert isinstance(body.pop("uptime_s"), (int, float))
        assert body == {
            "engine": "raptor",
            "ok": False,
            "reason_code": "runtime_load_failed",
            "semantic": server.RAPTOR_SEMANTIC,
            "service_date": "20260819",
            "svc_date": "2026-08-19",
            "walk": "graph",
        }
        assert "/private" not in response.get_data(as_text=True)


@pytest.mark.parametrize("path", HEALTH_PATHS)
@pytest.mark.parametrize("method", ("post", "put", "patch", "delete", "options"))
def test_health_probes_reject_non_get_head_with_405(client, path, method):
    response = getattr(client, method)(path)
    assert response.status_code == 405
    assert response.headers["Cache-Control"] == "no-store"
    assert response.get_json() == {"ok": False, "reason_code": "method_not_allowed"}


def test_repeated_probes_never_revalidate_cached_result(client, server, monkeypatch):
    monkeypatch.delenv("PERF_BENCHMARK_STATS", raising=False)
    monkeypatch.setattr(server, "_READINESS", _result())
    calls = {"feeds": 0, "runtime": 0}

    def fail_feeds(*args, **kwargs):
        calls["feeds"] += 1
        raise AssertionError("feed readiness must run during boot, never during a probe")

    def fail_runtime(*args, **kwargs):
        calls["runtime"] += 1
        raise AssertionError("runtime readiness must run during boot, never during a probe")

    monkeypatch.setattr(server.readiness, "check_readiness", fail_feeds)
    monkeypatch.setattr(server.readiness, "validate_runtime_state", fail_runtime)
    for _ in range(5):
        assert client.get("/readyz").status_code == 200
        assert client.head("/healthz").status_code == 200
    assert calls == {"feeds": 0, "runtime": 0}
