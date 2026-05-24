"""Shared fixtures for the Flask + R5 API integration suite (tests/test_api.py).

The app's scripts/server.py boots a real ~1.6 GB R5 JVM at *import* time (~30s). We do
that exactly ONCE per test session: a session-scoped fixture imports the module a single
time and exposes both the module (so tests can poke its locks/caches/globals directly) and
a Flask `app.test_client()`. Every test shares that one in-process R5 — independent of any
live server another agent may be running on :8000 (we never bind a port), just CPU-bound.

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

# Keep the test JVM modest so it coexists with another agent's live server / pytest JVM.
# Must be set BEFORE server.py is imported (it reads R5_MAX_MEMORY to cap -Xmx at boot).
os.environ.setdefault("R5_MAX_MEMORY", "1200M")


def pytest_configure(config):
    """Register the `slow` marker (heavy ~30s R5 passes) so it isn't an unknown-mark
    warning. Deselect with `-m 'not slow'` for a quick shape-only smoke run."""
    config.addinivalue_line(
        "markers", "slow: heavy test that drives a full R5 exact/itinerary pass (~30s+)"
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
    """Import scripts/server.py ONCE (boots R5) and return the module.

    Session-scoped so the ~30s JVM boot happens a single time for the whole run. Tests use
    this to reach into server internals: `_HEAVY_LOCK`, `_CELL_CACHE`, `_LAST_DEST_KEY`,
    the result caches, etc.
    """
    import server  # noqa: E402 — import here so the JVM boot is attributed to this fixture
    return server


@pytest.fixture(scope="session")
def client(server):
    """A single Flask test client shared by every test (do NOT re-import per test)."""
    return server.app.test_client()
