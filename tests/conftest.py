"""Shared fixtures for the Flask API integration suite (tests/test_api.py).

The app's scripts/server.py boots the JVM-FREE RAPTOR stack (since 2026-05-25): RAPTOR engine
+ hill-aware walk router + lean static bundle — import takes ~1s, no R5/JVM. We import the
module exactly ONCE per test session via a session-scoped fixture and expose both the module
(so tests can poke its locks/caches/globals directly) and a Flask `app.test_client()`. We
never bind a port, so the suite is independent of any live server on :8000 — but do NOT run
it beside a server that is still BOOTING (concurrent numba JIT corrupts the shared .nbc
cache; see CLAUDE.md).

ENGINE PINNING (post 2026-06-17 default flip): the served DEFAULT semantic is now
DEPART-AFTER (RAPTOR_SEMANTIC=departafter), but this in-process fixture is DELIBERATELY
pinned to ARRIVE-BY (the OPT-IN path) via the setdefault below — that pin is what keeps the
arrive-by engine, its MC/variance/per-route-typical/geom tests, and its golden under test
after the default moved. The depart-after SERVED DEFAULT is covered by CHILD-process tests
that boot RAPTOR_SEMANTIC=departafter (test_itinerary_equals_map_departafter,
test_compute_exact_matches_golden_departafter). The setdefaults also shield the suite from
ambient/.env leakage (e.g. an exported USE_RAPTOR=0 silently swapping the engine). The
legacy R5 path (USE_RAPTOR=0, ~30s JVM boot) is only exercised if you export USE_RAPTOR=0.

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

# Pin the in-process engine under test (see module docstring). Must be set BEFORE server.py is
# imported (it reads these at import time). RAPTOR_SEMANTIC is pinned to ARRIVE-BY on PURPOSE:
# the server default is now depart-after, so without this pin the arrive-by tests would all
# either run depart-after or skip. The depart-after default is covered by child-process tests.
os.environ.setdefault("USE_RAPTOR", "1")
os.environ.setdefault("USE_WALK_GRAPH", "1")
os.environ.setdefault("RAPTOR_SEMANTIC", "arriveby")  # OPT-IN path, pinned for the in-process suite
os.environ.setdefault("RAPTOR_MC", "1")
# Only read on the legacy _NEED_R5 path (USE_RAPTOR=0 / missing walk bakes): keep that
# JVM modest so it coexists with another agent's live server / pytest JVM.
os.environ.setdefault("R5_MAX_MEMORY", "1200M")

# The Playwright browser suite (tests/e2e/) drives an ALREADY-RUNNING server on :8000 and
# has its own conftest.py/pytest.ini — run it via tests/e2e/run.sh. Collecting it from a
# plain `pytest tests/` both shadows this conftest (two top-level modules named
# `conftest` -> test_api's `from conftest import FERRY_LAT` resolves to e2e's) and would
# hammer a server this suite assumes is NOT booting (numba .nbc corruption gotcha).
collect_ignore = ["e2e"]


def pytest_configure(config):
    """Register the `slow` marker so it isn't an unknown-mark warning. Under the default
    RAPTOR boot these tests are seconds, not minutes — the marker survives because they
    are still the heaviest (full-grid exact/itinerary passes; ~30s+ only on the legacy
    USE_RAPTOR=0 R5 path). Deselect with `-m 'not slow'` for a shape-only smoke run."""
    config.addinivalue_line(
        "markers", "slow: full-grid exact/itinerary pass (seconds on RAPTOR; ~30s+ on legacy R5)"
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
    """Import scripts/server.py ONCE (JVM-free RAPTOR boot, ~1s) and return the module.

    Session-scoped so the boot (and the ~30s JVM boot on the legacy USE_RAPTOR=0 path)
    happens a single time for the whole run. Tests use this to reach into server
    internals: `_HEAVY_LOCK`, `_CELL_CACHE`, `_LAST_DEST_KEY`, the result caches, etc.
    """
    import server  # noqa: E402 — import here so the JVM boot is attributed to this fixture
    return server


@pytest.fixture(scope="session")
def client(server):
    """A single Flask test client shared by every test (do NOT re-import per test)."""
    return server.app.test_client()
