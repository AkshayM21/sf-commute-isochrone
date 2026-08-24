#!/usr/bin/env python3
"""Controlled synthetic stage benchmark for the committed Monte-Carlo kernel.

This is deliberately an *offline diagnostic*, not a server benchmark and not a new routing
path. It prepares a small deterministic synthetic workload, warms
the exact Numba specializations, then measures four sequential per-draw stages:

* schedule perturbation;
* reverse-profile sweep;
* committed-route scoring; and
* optional tail snapshot encoding.

The stage run is checked by direct ``numpy.array_equal`` against the production parallel
``montecarlo_committed(..., capture_tail=True)`` result.  No response or array hashes are
created.  Its stage totals are intentionally *not* an endpoint latency: Python dispatch between
the isolated stages and serial draw execution make them useful for attribution, not for deciding
the served wall-clock time.  Use ``scripts/perf_benchmark.py`` for controlled endpoint timing.

Examples (run one process at a time: Numba's cache is shared):

    NUMBA_NUM_THREADS=1 .venv/bin/python scripts/mc_kernel_stage_benchmark.py \\
      --repeats 5 --output out/mc-stages.json
    NUMBA_NUM_THREADS=2 .venv/bin/python scripts/mc_kernel_stage_benchmark.py \\
      --tree planned --walk-scalar 1.0 --repeats 5

The fixture exercises two joined patterns, a transfer tail, committed transit/walk rows, and
unreachable cells. It is intentionally small enough to run in a data-less checkout and is not a
performance claim for production geography.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _rss_bytes():
    """Best available process maximum RSS, normalized to bytes."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _distribution(samples):
    values = [round(float(v), 3) for v in samples]
    if not values:
        return {"samples_ms": [], "min_ms": None, "median_ms": None,
                "p90_ms": None, "max_ms": None}
    ordered = sorted(values)
    return {
        "samples_ms": values,
        "min_ms": ordered[0],
        "median_ms": round(float(statistics.median(ordered)), 3),
        "p90_ms": round(float(np.percentile(np.asarray(ordered), 90.0, method="linear")), 3),
        "max_ms": ordered[-1],
    }


class _SyntheticEngine:
    max_min = 99
    service_date = "synthetic"

    def _mc_draw_arrays(self, draws, seed):
        rng = np.random.default_rng(seed)
        return (rng.gamma(2.0, 15.0, size=(draws, 6)).astype(np.float64),
                rng.gamma(2.0, 0.02, size=(draws, 6)).astype(np.float64))


def _load_workload(tree_kind, walk_scalar, draws, seed, max_rounds):
    """Build a tiny workload with two joined patterns and a transfer tail."""
    from core import raptor as R
    engine = _SyntheticEngine()
    data = {
        "n_stops": 3,
        "pat_nstops": np.array([2, 2], np.int32),
        "pat_ntrips": np.array([3, 3], np.int32),
        "pat_stop_off": np.array([0, 2], np.int32),
        "pat_mat_off": np.array([0, 6], np.int32),
        "pat_stops": np.array([0, 1, 1, 2], np.int32),
        "pat_dep": np.array([100, 150, 220, 270, 340, 390,
                             240, 300, 360, 420, 480, 540], np.int32),
        "pat_arr": np.array([100, 160, 220, 280, 340, 400,
                             240, 300, 360, 420, 480, 540], np.int32),
        "ras_off": np.array([0, 1, 3, 4], np.int32),
        "ras_pat": np.array([0, 0, 1, 1], np.int32),
        "ras_pos": np.array([0, 1, 0, 1], np.int32),
        "tr_off": np.array([0, 0, 0, 0], np.int32),
        "tr_to": np.empty(0, np.int32),
        "tr_time": np.empty(0, np.int32),
    }
    legs = {
        "commit_home": np.array([100, 220, 0, 0], np.int64),
        "commit_kind": np.array([2, 2, 1, 0], np.int64),
        "commit_walk0": np.array([0, 0, 0, 0], np.int64),
        "commit_pi": np.array([0, 0, 0, 0], np.int64),
        "commit_bpos": np.array([0, 0, 0, 0], np.int64),
        "commit_apos": np.array([1, 1, 0, 0], np.int64),
        "commit_as": np.array([1, 1, 0, 0], np.int64),
    }
    deadlines = np.arange(300, 661, 60, dtype=np.int64)
    egress_g = np.array([2], dtype=np.int32)
    egress_w_scaled = np.array([0], dtype=np.int64)
    perfect = np.array([5, 5, 17, -1], np.int64)
    delta0_all, slope_all = engine._mc_draw_arrays(draws, seed)
    return {
        "R": R,
        "engine": engine,
        "data": data,
        "egress_g": egress_g,
        "egress_w": egress_w_scaled,
        "deadlines": np.ascontiguousarray(deadlines, dtype=np.int64),
        "legs": {key: np.ascontiguousarray(value, dtype=np.int64) for key, value in legs.items()},
        "perfect": np.ascontiguousarray(perfect, dtype=np.int64),
        "delta0_all": np.ascontiguousarray(delta0_all, dtype=np.float64),
        "slope_all": np.ascontiguousarray(slope_all, dtype=np.float64),
        "max_rounds": int(max_rounds),
        "tree": tree_kind,
        "walk_scalar": float(walk_scalar),
    }


def _stage_once(workload):
    """Run four isolated stages sequentially and return exact assembled MC arrays plus timings."""
    from core import raptor_numba as RN

    R = workload["R"]
    flat = R._mc_flat_args(workload["data"])
    (n_stops, pat_nstops, pat_ntrips, pat_stop_off, pat_mat_off, pat_trip_off,
     pat_stops, pat_dep, pat_arr, ras_off, ras_pat, ras_pos, tr_off, tr_to, tr_time) = flat
    deadlines = workload["deadlines"]
    legs = workload["legs"]
    n_cells = legs["commit_home"].size
    n_draws = workload["delta0_all"].shape[0]
    commute_all = np.empty((n_cells, n_draws), dtype=np.float64)
    tail_lag = np.empty((n_draws, int(n_stops), deadlines.size), dtype=np.uint16)
    tail_valid = np.empty(n_draws, dtype=np.uint8)
    stage_s = {"perturb": 0.0, "profile": 0.0, "score": 0.0, "tail_encode": 0.0}
    for draw in range(n_draws):
        dep_r = np.empty(pat_dep.size, dtype=np.int64)
        arr_r = np.empty(pat_arr.size, dtype=np.int64)
        started = time.perf_counter()
        RN._perturb(
            pat_nstops, pat_ntrips, pat_mat_off, pat_trip_off, pat_dep, pat_arr,
            workload["delta0_all"][draw], workload["slope_all"][draw], dep_r, arr_r)
        stage_s["perturb"] += time.perf_counter() - started

        started = time.perf_counter()
        latest = RN._profile(
            n_stops, pat_nstops, pat_ntrips, pat_stop_off, pat_mat_off, pat_stops,
            dep_r, arr_r, ras_off, ras_pat, ras_pos, tr_off, tr_to, tr_time,
            workload["egress_g"], workload["egress_w"], deadlines, 60,
            workload["max_rounds"])
        stage_s["profile"] += time.perf_counter() - started

        started = time.perf_counter()
        RN._mc_stage_score_committed_draw(
            pat_nstops, pat_ntrips, pat_mat_off, deadlines, 60,
            legs["commit_home"], legs["commit_kind"], legs["commit_walk0"],
            legs["commit_pi"], legs["commit_bpos"], legs["commit_apos"], legs["commit_as"],
            workload["perfect"], workload["engine"].max_min, latest, dep_r, arr_r,
            commute_all[:, draw])
        stage_s["score"] += time.perf_counter() - started

        started = time.perf_counter()
        tail_valid[draw] = RN._mc_stage_encode_tail(latest, deadlines, tail_lag[draw])
        stage_s["tail_encode"] += time.perf_counter() - started
    return commute_all, tail_lag, tail_valid, {
        key: elapsed * 1000.0 for key, elapsed in stage_s.items()
    }


def _production_once(workload):
    """Call the unmodified production kernel as the stage-equivalence reference."""
    from core import raptor_numba as RN

    R = workload["R"]
    flat = R._mc_flat_args(workload["data"])
    legs = workload["legs"]
    return RN.montecarlo_committed(
        *flat, workload["egress_g"], workload["egress_w"], workload["deadlines"], 60,
        workload["max_rounds"], legs["commit_home"], legs["commit_kind"],
        legs["commit_walk0"], legs["commit_pi"], legs["commit_bpos"],
        legs["commit_apos"], legs["commit_as"], workload["perfect"],
        workload["engine"].max_min, workload["delta0_all"], workload["slope_all"],
        True)


def _workload_identity(workload, tree_kind, walk_scalar):
    data = workload["data"]
    legs = workload["legs"]
    return {
        "source": "synthetic",
        "service_date": str(workload["engine"].service_date),
        "build_version": int(data.get("build_version", -1)),
        "tree": tree_kind,
        "walk_scalar": float(walk_scalar),
        "draws": int(workload["delta0_all"].shape[0]),
        "deadlines": int(workload["deadlines"].size),
        "stops": int(data["n_stops"]),
        "patterns": int(np.asarray(data["pat_nstops"]).size),
        "trips": int(np.asarray(data["pat_ntrips"], dtype=np.int64).sum()),
        "cells": int(legs["commit_home"].size),
        "transit_cells": int(np.sum(legs["commit_kind"] == 2)),
        "egress_stops": int(workload["egress_g"].size),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", choices=("planned", "arriveby"), default="planned",
                        help="normal server-like planned tree, or an arrive-by diagnostic tree")
    parser.add_argument("--walk-scalar", type=float, default=1.0)
    parser.add_argument("--draws", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None, help="write JSON report to this path")
    args = parser.parse_args(argv)
    if args.draws <= 0 or args.repeats <= 0 or args.max_rounds <= 0:
        parser.error("--draws, --repeats, and --max-rounds must all be positive")
    if args.walk_scalar <= 0:
        parser.error("--walk-scalar must be positive")
    workload = _load_workload(
        args.tree, args.walk_scalar, args.draws, args.seed, args.max_rounds)

    # Compile every exact stage specialization and the independent production implementation before
    # timing.  This intentionally does not time engine/data setup or any HTTP/server work.
    staged_warm = _stage_once(workload)
    production_warm = _production_once(workload)
    if not (np.array_equal(staged_warm[0], production_warm[0])
            and np.array_equal(staged_warm[1], production_warm[1])
            and np.array_equal(staged_warm[2], production_warm[2])):
        raise RuntimeError("warmup stage assembly differs from production committed kernel")

    samples = {"perturb": [], "profile": [], "score": [], "tail_encode": [],
               "staged_total": [], "production_kernel": []}
    for _ in range(args.repeats):
        started = time.perf_counter()
        staged = _stage_once(workload)
        samples["staged_total"].append((time.perf_counter() - started) * 1000.0)
        for key, value in staged[3].items():
            samples[key].append(value)
        started = time.perf_counter()
        production = _production_once(workload)
        samples["production_kernel"].append((time.perf_counter() - started) * 1000.0)
        if not (np.array_equal(staged[0], production[0])
                and np.array_equal(staged[1], production[1])
                and np.array_equal(staged[2], production[2])):
            raise RuntimeError("stage assembly differs from production committed kernel")

    try:
        import numba
        numba_threads = int(numba.get_num_threads())
    except Exception:
        numba_threads = None
    report = {
        "schema_version": 1,
        "kind": "committed_mc_stage_attribution",
        "limits": [
            "sequential per-draw stages include Python call boundaries and are attribution, not endpoint latency",
            "production_kernel is the same process's parallel committed kernel and remains the served comparison",
            "no server, HTTP, data-build, tree-build, statistics, or planned-overlay time is included",
        ],
        "workload": _workload_identity(workload, args.tree, args.walk_scalar),
        "execution": {
            "repeats": int(args.repeats),
            "numba_threads": numba_threads,
            "env_numba_num_threads": os.environ.get("NUMBA_NUM_THREADS"),
            "loadavg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
            "rss_peak_bytes": _rss_bytes(),
        },
        "direct_array_equality": {
            "commute_all": True,
            "tail_lag": True,
            "tail_valid": True,
        },
        "timing_ms": {key: _distribution(value) for key, value in samples.items()},
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return report


if __name__ == "__main__":
    main()
