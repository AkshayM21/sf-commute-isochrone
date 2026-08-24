#!/usr/bin/env python
"""Small liveness-first WSGI bootstrap for the commute server.

Importing :mod:`server` is intentionally kept out of this module's import path.  A
``RuntimeState`` loads it on a daemon thread while the bootstrap WSGI app can answer
probes immediately.  Keeping the state and WSGI adapter as ordinary objects also
makes this module straightforward to exercise without importing the production app.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, MutableSequence, Optional, Protocol


class WSGIApplication(Protocol):
    def __call__(self, environ: dict, start_response: Callable[..., object]) -> Iterable[bytes]:
        ...


UNINITIALIZED = "uninitialized"
LOADING = "loading"
READY = "ready"
FAILED = "failed"


@dataclass(frozen=True)
class RuntimeSnapshot:
    """A safe view of runtime state; it deliberately contains no load exception."""

    status: str
    target: Optional[WSGIApplication]

    @property
    def reason(self) -> str:
        if self.status == FAILED:
            return "runtime_load_failed"
        if self.status == READY:
            return "runtime_ready"
        return "runtime_uninitialized"


class RuntimeState:
    """Thread-safe target-app holder with a resettable, test-friendly lifecycle."""

    def __init__(self, loader: Callable[[], WSGIApplication]):
        self._loader = loader
        self._lock = threading.RLock()
        self._loaded = threading.Event()
        self._status = UNINITIALIZED
        self._target: Optional[WSGIApplication] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """Begin loading once and return whether this call started the worker."""
        with self._lock:
            if self._thread is not None:
                return False
            self._status = LOADING
            self._thread = threading.Thread(
                target=self._load, name="runtime-loader", daemon=True
            )
            self._thread.start()
            return True

    def _load(self) -> None:
        try:
            loaded = self._loader()
            # A test or alternate loader may naturally return the imported module rather than
            # its WSGI object; accept both forms without importing the production module here.
            target = loaded if callable(loaded) else getattr(loaded, "app", None)
            if not callable(target):
                raise TypeError("loaded runtime is not callable")
        except BaseException:
            # Never retain or expose import exception text: paths and dependency details can
            # contain secrets.  The generic state is all probes and callers need.
            with self._lock:
                self._target = None
                self._status = FAILED
                self._loaded.set()
            return
        with self._lock:
            self._target = target
            self._status = READY
            self._loaded.set()

    def snapshot(self) -> RuntimeSnapshot:
        with self._lock:
            return RuntimeSnapshot(self._status, self._target)

    @property
    def status(self) -> str:
        return self.snapshot().status

    @property
    def target(self) -> Optional[WSGIApplication]:
        return self.snapshot().target

    def wait(self, timeout: Optional[float] = None) -> str:
        """Wait for success/failure (or timeout) and return the current status."""
        self._loaded.wait(timeout)
        return self.status


def load_target(module_name: str = "server") -> WSGIApplication:
    """Import the heavy server module and return its Flask WSGI application."""
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    module = importlib.import_module(module_name)
    return getattr(module, "app")


def _json_response(
    start_response: Callable[..., object],
    status: str,
    payload: dict,
    method: str,
    *,
    no_store: bool = True,
) -> Iterable[bytes]:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    headers: MutableSequence[tuple[str, str]] = [
        ("Content-Type", "application/json"),
        ("Content-Length", str(len(body))),
    ]
    if no_store:
        headers.append(("Cache-Control", "no-store"))
    start_response(status, headers)
    return [b""] if method == "HEAD" else [body]


class BootstrapWSGI:
    """WSGI adapter that owns probes and gates the target app until it is ready."""

    def __init__(self, state: RuntimeState):
        self.state = state

    def __call__(self, environ: dict, start_response: Callable[..., object]) -> Iterable[bytes]:
        path = environ.get("PATH_INFO", "") or "/"
        method = (environ.get("REQUEST_METHOD", "GET") or "GET").upper()

        if path in {"/livez", "/readyz", "/healthz"} and method not in {"GET", "HEAD"}:
            return _json_response(
                start_response,
                "405 Method Not Allowed",
                {"ok": False, "reason_code": "method_not_allowed"},
                method,
            )

        if path == "/livez":
            return _json_response(
                start_response,
                "200 OK",
                {"ok": True, "reason_code": "process_alive"},
                method,
            )

        snapshot = self.state.snapshot()
        if snapshot.status != READY or snapshot.target is None:
            # /readyz, /healthz, and ordinary requests all use the same deliberately generic
            # failure vocabulary.  No exception object, module name, or filesystem path leaves
            # this process.
            return _json_response(
                start_response,
                "503 Service Unavailable",
                {"ok": False, "reason_code": snapshot.reason},
                method,
            )

        target = snapshot.target
        if path in {"/readyz", "/healthz"}:
            # Readiness belongs to the loaded application: it can validate feeds, baked
            # artifacts, and engine state rather than merely proving that import succeeded.
            return self._delegate(target, environ, start_response, no_store=True)
        return self._delegate(target, environ, start_response, no_store=False)

    @staticmethod
    def _delegate(
        target: WSGIApplication,
        environ: dict,
        start_response: Callable[..., object],
        *,
        no_store: bool,
    ) -> Iterable[bytes]:
        if not no_store:
            return target(environ, start_response)

        def health_start(status: str, headers: list[tuple[str, str]], exc_info=None):
            filtered = [(key, value) for key, value in headers if key.lower() != "cache-control"]
            filtered.append(("Cache-Control", "no-store"))
            return start_response(status, filtered, exc_info)

        return target(environ, health_start)


def create_app(
    state: Optional[RuntimeState] = None,
    *,
    loader: Optional[Callable[[], WSGIApplication]] = None,
    start: bool = False,
) -> BootstrapWSGI:
    """Build a bootstrap app; dependency injection keeps tests independent of ``server``."""
    if state is None:
        state = RuntimeState(loader or load_target)
    elif loader is not None:
        raise ValueError("pass either state or loader, not both")
    app = BootstrapWSGI(state)
    if start:
        state.start()
    return app


def serve_app(
    app: BootstrapWSGI,
    host: str,
    port: int,
    *,
    server_factory: Optional[Callable[[], object]] = None,
    server_runner: Optional[Callable[[object], object]] = None,
) -> object:
    """Bind first, start the runtime loader second, then enter the server loop.

    ``server_factory`` and ``server_runner`` are an intentionally small seam for deterministic
    tests.  Production defaults use waitress' bound ``create_server`` or Werkzeug's threaded
    ``make_server`` fallback; neither imports the target app.
    """
    if server_factory is None:
        try:
            from waitress import create_server
        except ImportError:
            from werkzeug.serving import make_server

            print(
                f"[boot] waitress not installed — falling back to threaded local server on "
                f"{host}:{port}"
            )
            server_factory = lambda: make_server(host, port, app, threaded=True)
            if server_runner is None:
                server_runner = lambda server: server.serve_forever()
        else:
            print(f"[boot] waitress serving on {host}:{port}")
            server_factory = lambda: create_server(
                app,
                host=host,
                port=port,
                threads=8,
                _quiet=False,
                channel_timeout=120,
            )
            if server_runner is None:
                server_runner = lambda server: server.run()

    server = server_factory()
    # create_server/make_server bind and listen during construction.  Starting the worker only
    # after that point ensures the socket exists even if import holds the GIL for a while.
    app.state.start()
    if server_runner is None:
        server_runner = lambda bound_server: (
            bound_server.run() if hasattr(bound_server, "run") else bound_server.serve_forever()
        )
    return server_runner(server)


def main() -> None:
    app = create_app(start=False)
    serve_app(app, "127.0.0.1", int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
