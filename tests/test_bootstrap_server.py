"""Focused tests for the liveness-first server bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bootstrap_server import BootstrapWSGI, RuntimeState, serve_app  # noqa: E402


def call(app, path, method="GET"):
    captured = {}

    def start_response(status, headers, exc_info=None):
        captured["status"] = status
        captured["headers"] = dict(headers)

    body = b"".join(
        app(
            {"PATH_INFO": path, "REQUEST_METHOD": method, "wsgi.url_scheme": "http"},
            start_response,
        )
    )
    return captured["status"], captured["headers"], body


class Target:
    def __call__(self, environ, start_response):
        path = environ["PATH_INFO"]
        if path in {"/readyz", "/healthz"}:
            if environ.get("target_not_ready"):
                start_response("503 Service Unavailable", [("Content-Type", "application/json")])
                return [b'{"ok":false,"reason":"target_not_ready"}']
            start_response("200 OK", [("Content-Type", "application/json")])
            return [b'{"target":"healthy"}']
        start_response("200 OK", [("Content-Type", "text/plain")])
        return [f"target:{path}".encode()]


def test_liveness_and_gated_health_before_loading():
    state = RuntimeState(lambda: Target())
    app = BootstrapWSGI(state)

    status, headers, body = call(app, "/livez")
    assert status == "200 OK"
    assert b'"reason_code":"process_alive"' in body
    assert headers["Cache-Control"] == "no-store"

    for path in ("/readyz", "/healthz", "/normal"):
        status, headers, body = call(app, path)
        assert status == "503 Service Unavailable"
        assert b'"reason_code":"runtime_uninitialized"' in body
        assert headers["Cache-Control"] == "no-store"


def test_success_load_delegates_normal_and_health_requests():
    state = RuntimeState(lambda: Target())
    app = BootstrapWSGI(state)
    state.start()
    assert state.wait(1) == "ready"

    status, headers, body = call(app, "/readyz")
    assert status == "200 OK" and body == b'{"target":"healthy"}'
    assert headers["Cache-Control"] == "no-store"
    status, headers, body = call(app, "/normal")
    assert status == "200 OK" and body == b"target:/normal"
    assert "Cache-Control" not in headers
    status, headers, body = call(app, "/healthz")
    assert status == "200 OK" and body == b'{"target":"healthy"}'
    assert headers["Cache-Control"] == "no-store"


def test_readyz_preserves_target_not_ready_response():
    state = RuntimeState(lambda: Target())
    app = BootstrapWSGI(state)
    state.start()
    assert state.wait(1) == "ready"

    # The synthetic target can reject readiness independently of import success.  The bootstrap
    # must not replace that status with an unconditional 200.
    environ = {"PATH_INFO": "/readyz", "REQUEST_METHOD": "GET", "target_not_ready": True}
    captured = {}

    def start_response(status, headers, exc_info=None):
        captured["status"] = status
        captured["headers"] = dict(headers)

    body = b"".join(app(environ, start_response))
    assert captured["status"] == "503 Service Unavailable"
    assert captured["headers"]["Cache-Control"] == "no-store"
    assert body == b'{"ok":false,"reason":"target_not_ready"}'


def test_server_is_bound_before_runtime_loader_starts():
    events = []

    class BoundServer:
        def run(self):
            state.wait(1)
            events.append("run")

    def factory():
        events.append("bind")
        assert state.status == "uninitialized"
        return BoundServer()

    def loader():
        events.append("load")
        return Target()

    state = RuntimeState(loader)
    app = BootstrapWSGI(state)
    result = serve_app(
        app,
        "127.0.0.1",
        0,
        server_factory=factory,
        server_runner=lambda server: server.run(),
    )
    assert result is None
    assert events == ["bind", "load", "run"]


def test_failed_load_has_generic_reason_and_keeps_liveness():
    def fail():
        raise RuntimeError("secret path /private/dependency.py")

    state = RuntimeState(fail)
    app = BootstrapWSGI(state)
    state.start()
    assert state.wait(1) == "failed"
    for path in ("/readyz", "/healthz", "/normal"):
        status, _, body = call(app, path)
        assert status == "503 Service Unavailable"
        assert body == b'{"ok":false,"reason_code":"runtime_load_failed"}'
    assert call(app, "/livez")[0] == "200 OK"
