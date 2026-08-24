"""Shared fixtures for the Flask API integration suite (tests/test_api.py).

The app's scripts/server.py boots the RAPTOR engine plus graph-backed hill-aware walk router
and lean static bundle — import takes ~1s. We import the
module exactly ONCE per test session via a session-scoped fixture and expose both the module
(so tests can poke its locks/caches/globals directly) and a Flask `app.test_client()`. We
never bind a port, so the suite is independent of any live server on :8000 — but do NOT run
it beside a server that is still BOOTING (concurrent numba JIT corrupts the shared .nbc
cache; see CLAUDE.md).

ENGINE PINNING (post 2026-06-17 default flip): the served DEFAULT semantic is now
DEPART-AFTER (RAPTOR_SEMANTIC=departafter), but this in-process fixture is DELIBERATELY
pinned to ARRIVE-BY (the OPT-IN path) — that pin is what keeps the arrive-by engine, its
MC/variance/per-route-typical/geom tests, and its golden under test after the default
moved. The depart-after SERVED DEFAULT is covered by CHILD-process tests that boot their
own env with RAPTOR_SEMANTIC=departafter (test_itinerary_equals_map_departafter,
test_compute_exact_matches_golden_departafter), so the pin never leaks into them.

Two tiers of env setup below (they behave differently — don't conflate them):
  * HARD-PINNED (unconditional assignment; ambient/exported env can NOT override):
    RAPTOR_SEMANTIC=arriveby. The pin is load-bearing — an exported/leaked
    RAPTOR_SEMANTIC=departafter would silently skip the whole arrive-by suite.
  * DEFAULTED (setdefault; an exported env var DOES override — by design):
    RAPTOR_MC. The production runtime has no alternate routing engine or fallback flags.

PRIVACY: tests use a neutral public SF coordinate (the Ferry Building) as the workplace.
The user's real saved address/coords are NEVER imported or hardcoded here.
"""
import os
import sys

import pytest

# Make `scripts/` importable (server.py does `from core import ...`).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_REPO_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

# Engine env for the in-process suite (see the two-tier note in the module docstring).
# Must be set BEFORE server.py is imported (it reads these at import time).
# RAPTOR_SEMANTIC is HARD-PINNED to ARRIVE-BY on PURPOSE (unconditional, not setdefault —
# an exported RAPTOR_SEMANTIC must not silently swap/skip the arrive-by suite): the server
# default is now depart-after, so without this pin the arrive-by tests would all either run
# depart-after or skip. The depart-after default is covered by child-process tests that set
# their own env explicitly.
os.environ["RAPTOR_SEMANTIC"] = "arriveby"   # OPT-IN path, hard-pinned for the in-process suite
os.environ.setdefault("RAPTOR_MC", "1")

# The Playwright browser suite (tests/e2e/) drives an ALREADY-RUNNING server on :8000 and
# has its own conftest.py/pytest.ini — run it via tests/e2e/run.sh. Collecting it from a
# plain `pytest tests/` both shadows this conftest (two top-level modules named
# `conftest` -> test_api's `from conftest import FERRY_LAT` resolves to e2e's) and would
# hammer a server this suite assumes is NOT booting (numba .nbc corruption gotcha).
collect_ignore = ["e2e"]


def pytest_configure(config):
    """Register the `slow` marker so it isn't an unknown-mark warning. The marker survives
    because these are still the heaviest full-grid exact/itinerary passes. Deselect with
    `-m 'not slow'` for a shape-only smoke run."""
    config.addinivalue_line(
        "markers", "slow: full-grid exact/itinerary pass"
    )


# Neutral public SF workplace for all tests — NOT the user's saved address. The Ferry
# Building is a well-known public landmark; using it keeps the user's private workplace
# out of the test code, the golden file, and any CI logs.
FERRY_LAT = 37.7955
FERRY_LON = -122.3937
# A second, clearly-distinct public coordinate (Twin Peaks) for the cache-clearing test.
TWIN_PEAKS_LAT = 37.7544
TWIN_PEAKS_LON = -122.4477


@pytest.fixture(scope="session")
def server():
    """Import scripts/server.py once and return the module.

    Tests use this to reach into the bounded RAPTOR caches and other test-only internals.
    """
    import server  # noqa: E402
    return server


@pytest.fixture(scope="session")
def client(server):
    """A single Flask test client shared by every test (do NOT re-import per test)."""
    return server.app.test_client()


@pytest.fixture(autouse=True)
def _isolate_rate_limits(request):
    """Reset limits only for tests that actually depend on the Flask server.

    Keeping the dependency lazy lets pure routing/helper tests run in a clean checkout without
    ignored transit data while preserving isolation for API tests that request ``server`` or
    ``client`` directly or transitively.
    """
    if not {"server", "client"}.intersection(request.fixturenames):
        yield
        return
    server_module = request.getfixturevalue("server")
    server_module.limiter.reset()
    yield
    server_module.limiter.reset()
