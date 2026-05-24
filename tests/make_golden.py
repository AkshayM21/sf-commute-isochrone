#!/usr/bin/env python
"""Regenerate the /compute_exact golden snapshot for the test workplace.

The /compute_exact endpoint is deterministic for a fixed network + service date, so we
snapshot its full {cellId: [best, real]} output once and assert future runs match it
(tests/test_api.py::test_compute_exact_matches_golden). This is a REGRESSION guard: any
unintended change to routing params, grid, or refine logic flips the snapshot.

WHEN TO REGENERATE: the golden depends on the GTFS feeds and on the auto-picked service
date (server picks a weekday with trips in ALL feeds via feeds.pick_service_date). When the
GTFS feeds are refreshed (a 511 repull), pick_service_date can shift to a different weekday
and the exact times legitimately change — at that point this golden is STALE and must be
regenerated. Run:

    .venv/bin/python tests/make_golden.py

It boots its own R5 (~30s + one ~30s exact pass) and overwrites tests/golden_exact_ferry.json.
The workplace is the public Ferry Building coordinate (privacy: never the user's address).
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))
os.environ.setdefault("R5_MAX_MEMORY", "1200M")

# Public, neutral workplace — kept identical to tests/conftest.py.
FERRY_LAT = 37.7955
FERRY_LON = -122.3937
GOLDEN_PATH = os.path.join(_HERE, "golden_exact_ferry.json")


def main():
    import server  # boots R5 once

    client = server.app.test_client()
    # Clear the coarse result cache so we capture a fresh exact compute, not a cached one.
    with server._RESULT_CACHE_LOCK:
        server._EXACT_RESULT_CACHE.clear()
    resp = client.get(f"/compute_exact?lat={FERRY_LAT}&lon={FERRY_LON}")
    assert resp.status_code == 200, f"compute_exact returned {resp.status_code}"
    cells = resp.get_json()["cells"]
    payload = {
        "_comment": (
            "Golden snapshot of GET /compute_exact for the Ferry Building "
            f"({FERRY_LAT},{FERRY_LON}). REGENERATE when the GTFS feeds refresh "
            "(pick_service_date may shift). See tests/make_golden.py."
        ),
        "dest": [FERRY_LAT, FERRY_LON],
        "service_date": str(server._SVC_DATE),
        "cells": cells,
    }
    with open(GOLDEN_PATH, "w") as f:
        json.dump(payload, f, separators=(",", ":"), sort_keys=True)
    print(f"wrote {GOLDEN_PATH}: {len(cells)} cells, service_date={server._SVC_DATE}")


if __name__ == "__main__":
    main()
