#!/usr/bin/env python
"""Regenerate the /compute_exact golden snapshot for the test workplace.

The /compute_exact endpoint is deterministic for a fixed network + service date + engine,
so we snapshot its full {cellId: [best, real]} output once and assert future runs match it
(tests/test_api.py::test_compute_exact_matches_golden). This is a REGRESSION guard: any
unintended change to routing params, grid, or refine logic flips the snapshot.

ENGINE: by default this boots the JVM-FREE RAPTOR stack (USE_RAPTOR=1, arrive-by — the
production default since 2026-05-25; ~1s boot, full grid in ~ms), and the snapshot records
the engine identity ("engine": {use_raptor, raptor_semantic, use_walk_graph, gtfs_fp}).
The golden test SKIPS (not fails) when the booted engine differs from the recorded one, so
running the suite with USE_RAPTOR=0 (legacy R5; R5_MAX_MEMORY only matters there) never
compares apples to oranges.

WHEN TO REGENERATE:
  * a GTFS repull — the auto-picked service date (feeds.pick_service_date) and the gtfs
    fingerprint shift, and exact times legitimately change;
  * an engine-DEFAULT change (e.g. the 2026-05-25 R5 -> RAPTOR flip, or a RAPTOR_SEMANTIC
    default change) — the recorded engine identity goes stale.
Run (with NO server booting on :8000 — concurrent numba JIT corrupts the .nbc cache):

    .venv/bin/python tests/make_golden.py

It overwrites tests/golden_exact_ferry.json. The workplace is the public Ferry Building
coordinate (privacy: never the user's address).
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
# Pin the engine to the production defaults (mirrors tests/conftest.py) so a regen on a
# clean shell snapshots exactly what the served product computes.
os.environ.setdefault("USE_RAPTOR", "1")
os.environ.setdefault("USE_WALK_GRAPH", "1")
os.environ.setdefault("RAPTOR_SEMANTIC", "arriveby")
os.environ.setdefault("RAPTOR_MC", "1")
os.environ.setdefault("R5_MAX_MEMORY", "1200M")   # only read on the legacy USE_RAPTOR=0 path

# Public, neutral workplace — kept identical to tests/conftest.py.
FERRY_LAT = 37.7955
FERRY_LON = -122.3937
GOLDEN_PATH = os.path.join(_HERE, "golden_exact_ferry.json")


def engine_identity(server):
    """The engine config that determines /compute_exact's output, recorded in the golden
    and compared by test_compute_exact_matches_golden (skip-on-mismatch, like service_date).
    gtfs_fp is the boot-time GTFS fingerprint (name:size:mtime of every feed) that already
    keys the baked structures — the right staleness signal alongside service_date."""
    return {
        "use_raptor": bool(server.USE_RAPTOR),
        "raptor_semantic": str(server.RAPTOR_SEMANTIC),
        "use_walk_graph": bool(server.USE_WALK_GRAPH),
        "gtfs_fp": server._gtfs_fp(),
    }


def main():
    import server  # boots the engine once (JVM-free RAPTOR by default)

    client = server.app.test_client()
    # Clear the coarse result cache so we capture a fresh exact compute, not a cached one.
    with server._RESULT_CACHE_LOCK:
        server._EXACT_RESULT_CACHE.clear()
    resp = client.get(f"/compute_exact?lat={FERRY_LAT}&lon={FERRY_LON}")
    assert resp.status_code == 200, f"compute_exact returned {resp.status_code}"
    cells = resp.get_json()["cells"]
    eng = engine_identity(server)
    payload = {
        "_comment": (
            "Golden snapshot of GET /compute_exact for the Ferry Building "
            f"({FERRY_LAT},{FERRY_LON}). REGENERATE on a GTFS repull (pick_service_date "
            "and gtfs_fp shift) or an engine-default change. See tests/make_golden.py."
        ),
        "dest": [FERRY_LAT, FERRY_LON],
        "service_date": str(server._SVC_DATE),
        "engine": eng,
        "cells": cells,
    }
    with open(GOLDEN_PATH, "w") as f:
        json.dump(payload, f, separators=(",", ":"), sort_keys=True)
    print(f"wrote {GOLDEN_PATH}: {len(cells)} cells, service_date={server._SVC_DATE}, "
          f"engine={eng}")


if __name__ == "__main__":
    main()
