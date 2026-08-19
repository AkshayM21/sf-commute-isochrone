#!/usr/bin/env python3
"""Controlled real-data stage benchmark for the committed Monte-Carlo kernel.

This is deliberately an *offline diagnostic*, not a server benchmark and not a new routing
path.  It prepares one normal ``RaptorEngine`` workload from an existing public oracle, warms
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
      --oracle tests/raptor_golden/oracle_downtown.npz --repeats 5 --output out/mc-stages.json
    NUMBA_NUM_THREADS=2 .venv/bin/python scripts/mc_kernel_stage_benchmark.py \\
      --tree planned --walk-scalar 1.0 --repeats 5

The oracle is a public, checked-in workplace fixture.  It supplies the normal egress/pure-walk
arrays; the engine loads the ordinary GTFS build and baked access table.  A data-less checkout
will fail with an actionable message instead of starting a server or fabricating a workload.
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


def _purewalk_aligned(engine, oracle):
    purewalk = np.full(len(engine.cell_ids), -1, dtype=np.int64)
    oracle_pos = {cell: i for i, cell in enumerate(oracle["cell_ids"].astype(str))}
    for i, cell in enumerate(engine.cell_ids):
        j = oracle_pos.get(cell)
        if j is not None:
            purewalk[i] = int(oracle["purewalk"][j])
    return purewalk


def _default_oracle():
    preferred = REPO / "tests" / "raptor_golden" / "oracle_downtown.npz"
    if preferred.exists():
        return preferred
    choices = sorted((REPO / "tests" / "raptor_golden").glob("oracle_*.npz"))
    if not choices:
        raise FileNotFoundError(
            "no public oracle fixture found; pass --oracle tests/raptor_golden/oracle_*.npz")
    return choices[0]


def _load_workload(oracle_path, tree_kind, walk_scalar, draws, seed, max_rounds):
    from core import raptor as R
    from core import raptor_engine as RE

    oracle = np.load(oracle_path, allow_pickle=True)
    engine = RE.RaptorEngine(verbose=False)
    egress_g = np.asarray(oracle["egress_g"], dtype=np.int32)
    egress_w = np.asarray(oracle["egress_w"], dtype=np.int64)
    purewalk = _purewalk_aligned(engine, oracle)
    if tree_kind == "planned":
        tree = engine.journey_tree_departafter(
            egress_g, egress_w, purewalk, percentile=50.0, walk_scalar=walk_scalar,
            max_rounds=max_rounds, planned=True)
    else:
        tree = engine.journey_tree(
            egress_g, egress_w, purewalk, walk_scalar=walk_scalar, max_rounds=max_rounds)
    perfect, _ = tree.commute_and_dominant()
    legs = tree.committed_first_legs()
    egress_w_scaled, _, _ = engine._scale_walk(egress_w, purewalk, walk_scalar)
    deadlines = RE._committed_deadline_prefix(engine.Tgrid_mc, legs, engine.max_min)
    data = getattr(tree, "d", engine.data)
    delta0_all, slope_all = engine._mc_draw_arrays(draws, seed)
    return {
        "R": R,
        "engine": engine,
        "oracle": oracle,
        "data": data,
        "egress_g": egress_g,
        "egress_w": egress_w_scaled,
        "deadlines": np.ascontiguousarray(deadlines, dtype=np.int64),
        "legs": {key: np.ascontiguousarray(value, dtype=np.int64) for key, value in legs.items()},
        "perfect": np.ascontiguousarray(perfect, dtype=np.int64),
        "delta0_all": np.ascontiguousarray(delta0_all, dtype=np.float64),
        "slope_all": np.ascontiguousarray(slope_all, dtype=np.float64),
        "max_rounds": int(max_rounds),
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
    """Call the unmodified production kernel, retained as the equality oracle."""
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


def _workload_identity(workload, oracle_path, tree_kind, walk_scalar):
    data = workload["data"]
    legs = workload["legs"]
    oracle = workload["oracle"]
    oracle_name = str(oracle["name"]) if "name" in oracle.files else Path(oracle_path).stem
    return {
        "oracle_path": str(oracle_path),
        "oracle_name": oracle_name,
        "service_date": str(workload["engine"].service_date),
        "build_version": int(data.get("build_version", -1)),
        "gtfs_fingerprint": str(data.get("gtfs_fp", "unknown")),
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
    parser.add_argument("--oracle", type=Path, default=None,
                        help="public oracle .npz (default: oracle_downtown when present)")
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
    oracle_path = (args.oracle or _default_oracle()).resolve()
    if not oracle_path.exists():
        parser.error(f"oracle does not exist: {oracle_path}")
    try:
        workload = _load_workload(
            oracle_path, args.tree, args.walk_scalar, args.draws, args.seed, args.max_rounds)
    except FileNotFoundError as error:
        parser.error(str(error))

    # Compile every exact stage specialization and the independent production oracle before
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
        "workload": _workload_identity(workload, oracle_path, args.tree, args.walk_scalar),
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
