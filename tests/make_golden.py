#!/usr/bin/env python
"""Regenerate the /compute_exact golden snapshot(s) for the test workplace.

The /compute_exact endpoint is deterministic for a fixed network + service date + engine,
so we snapshot its full {cellId: [best, real]} output once and assert future runs match it
(tests/test_api.py::test_compute_exact_matches_golden[_departafter]). This is a REGRESSION
guard: any unintended change to routing params, grid, or refine logic flips the snapshot.

TWO GOLDENS (one per RAPTOR semantic) since the 2026-06-17 default flip to depart-after:
  * tests/golden_exact_ferry.json            — ARRIVE-BY (RAPTOR_SEMANTIC=arriveby), the
    in-process suite's pinned engine (conftest.py setdefault); the opt-in path.
  * tests/golden_exact_ferry_departafter.json — DEPART-AFTER (RAPTOR_SEMANTIC=departafter),
    the SERVED DEFAULT since 2026-06-17 (planned, first-boarding-anchored schedule metric).
    Its test boots a depart-after server in a subprocess so the served-default map has real
    golden coverage (NOT a skip).

Each snapshot records the engine identity ("engine": {use_raptor, raptor_semantic,
use_walk_graph, gtfs_fp}). The golden tests SKIP (not fail) when the booted engine differs
from the recorded one, so running the suite with USE_RAPTOR=0 (legacy R5; R5_MAX_MEMORY only
matters there) never compares apples to oranges.

WHEN TO REGENERATE:
  * a GTFS repull — the auto-picked service date (feeds.pick_service_date) and the gtfs
    fingerprint shift, and exact times legitimately change;
  * an engine-DEFAULT change (e.g. the 2026-05-25 R5 -> RAPTOR flip, the 2026-06-17
    arrive-by -> depart-after default flip, or another RAPTOR_SEMANTIC change).
Run (with NO server booting on :8000 — concurrent numba JIT corrupts the .nbc cache):

    .venv/bin/python tests/make_golden.py                # BOTH goldens (default)
    .venv/bin/python tests/make_golden.py arriveby       # just the arrive-by golden
    .venv/bin/python tests/make_golden.py departafter    # just the depart-after golden

It overwrites the golden file(s). The workplace is the public Ferry Building coordinate
(privacy: never the user's address).
"""
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

# Public, neutral workplace — kept identical to tests/conftest.py.
FERRY_LAT = 37.7955
FERRY_LON = -122.3937

# One golden file per semantic. arrive-by keeps the original filename (no churn for the
# in-process arrive-by suite); depart-after gets its own file.
GOLDEN_PATHS = {
    "arriveby": os.path.join(_HERE, "golden_exact_ferry.json"),
    "departafter": os.path.join(_HERE, "golden_exact_ferry_departafter.json"),
}


def engine_identity(server):
    """The engine config that determines /compute_exact's output, recorded in the golden
    and compared by test_compute_exact_matches_golden[_departafter] (skip-on-mismatch, like
    service_date). gtfs_fp is the boot-time GTFS fingerprint (name:size:mtime of every feed)
    that already keys the baked structures. Walking configuration is part of the identity too:
    changing the product's default pace legitimately changes almost every cell."""
    return {
        "use_raptor": bool(server.USE_RAPTOR),
        "raptor_semantic": str(server.RAPTOR_SEMANTIC),
        "use_walk_graph": bool(server.USE_WALK_GRAPH),
        "gtfs_fp": server._gtfs_fp(),
        "walk_reference_kmh": float(server.config.WALK_KMH),
        "walk_speeds": {key: float(value) for key, value in sorted(server.WALK_SPEEDS.items())},
        "default_walk_speed": str(server.DEFAULT_SPEED),
    }


# The child process imports server.py with the semantic pinned, captures /compute_exact, and
# prints the snapshot JSON on the LAST stdout line. A subprocess (not two in-process boots)
# keeps the two semantics from sharing a numba JIT / module-global state and proves each boot
# is JVM-free. Identical engine pins to conftest.py for arrive-by; depart-after overrides only
# the semantic (still USE_RAPTOR=1 + USE_WALK_GRAPH=1 -> JVM-free).
_CHILD = r'''
import json, sys, os
sys.path.insert(0, %(scripts)r)
sys.path.insert(0, %(tests)r)
import server
from make_golden import engine_identity

assert server.USE_RAPTOR, "golden child booted without USE_RAPTOR"
assert server.RAPTOR_SEMANTIC == %(semantic)r, server.RAPTOR_SEMANTIC
# Prove JVM-free (the depart-after default must not load R5 either).
jvm = sorted(m for m in sys.modules
             if m == "r5py" or m.startswith("r5py.") or m.startswith("com.conveyal")
             or m == "jpype")
assert not jvm, "golden child is NOT JVM-free: %%s" %% jvm

client = server.app.test_client()
with server._RESULT_CACHE_LOCK:
    server._EXACT_RESULT_CACHE.clear()
resp = client.get("/compute_exact?lat=%(lat)s&lon=%(lon)s")
assert resp.status_code == 200, "compute_exact returned %%s" %% resp.status_code
cells = resp.get_json()["cells"]
payload = {
    "_comment": (
        "Golden snapshot of GET /compute_exact for the Ferry Building "
        "(%(lat)s,%(lon)s) under RAPTOR_SEMANTIC=%(semantic)s. REGENERATE on a GTFS repull "
        "(pick_service_date and gtfs_fp shift) or an engine-default change. See tests/make_golden.py."
    ),
    "dest": [%(lat)s, %(lon)s],
    "service_date": str(server._SVC_DATE),
    "engine": engine_identity(server),
    "cells": cells,
}
print("===GOLDEN-JSON===")
print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
'''


def _build_one(semantic):
    """Boot a child server pinned to ``semantic``, capture /compute_exact, write its golden."""
    scripts = os.path.join(_REPO_ROOT, "scripts")
    child = _CHILD % {"scripts": scripts, "tests": _HERE, "semantic": semantic,
                      "lat": FERRY_LAT, "lon": FERRY_LON}
    env = dict(os.environ)
    env.update(USE_RAPTOR="1", USE_WALK_GRAPH="1", RAPTOR_SEMANTIC=semantic, RAPTOR_MC="1")
    env.setdefault("R5_MAX_MEMORY", "1200M")        # only read on the legacy USE_RAPTOR=0 path
    proc = subprocess.run([sys.executable, "-c", child], env=env,
                          capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise SystemExit(
            f"golden child for {semantic} failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    marker = "===GOLDEN-JSON==="
    assert marker in proc.stdout, f"no golden marker in child stdout:\n{proc.stdout}"
    payload_line = proc.stdout.split(marker, 1)[1].strip().splitlines()[0]
    payload = json.loads(payload_line)
    path = GOLDEN_PATHS[semantic]
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"), sort_keys=True)
    print(f"wrote {path}: {len(payload['cells'])} cells, "
          f"service_date={payload['service_date']}, engine={payload['engine']}")


def main():
    which = sys.argv[1:] or ["arriveby", "departafter"]
    for semantic in which:
        if semantic not in GOLDEN_PATHS:
            raise SystemExit(f"unknown semantic {semantic!r}; choose from {sorted(GOLDEN_PATHS)}")
        _build_one(semantic)


if __name__ == "__main__":
    main()
