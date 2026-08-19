#!/usr/bin/env python3
"""Precompile every production RAPTOR/Numba signature before serving traffic.

Run after dependencies/data are installed with the same ``NUMBA_CACHE_DIR`` and Python user as the
systemd service. The fixed destination and cell are public test fixtures; no saved user address is
read. Exercising the real progressive flow is intentional: it covers the profile, compact traced
parent, committed Monte-Carlo capture, retained-tail replay, and isolated planned-tail kernels with
their actual array mutability/dtype signatures.
"""
from __future__ import annotations

import time

import numpy as np


PUBLIC_DEST = (37.7714154, -122.4030885)       # 650 Townsend St test fixture
PUBLIC_CELL = "1916"                           # Mission Dolores public hotspot fixture


def _warm_planned_one_tail_discovery(dispatcher):
    """Compile the isolated one-transfer discovery ABI without service topology.

    The ordinary public pin below intentionally exercises production-shaped data, but a
    particular service day need not contain a first leg with a downstream transfer.  The
    dispatcher is nevertheless used by a later pin when that topology does occur.  A tiny,
    valid schedule here compiles its exact production array/scalar signature regardless of
    which routes happen to run on the warmup date.
    """
    i32 = np.int32
    i64 = np.int64
    # One two-stop trip, one routes-at-stop record, and no walking edges.  Its sole alight has
    # no second-board candidate, so the result is deliberately empty while every array access
    # remains valid.  Preserve the mixed int32/int64 layout emitted by raptor_build; Numba
    # dispatches on dtype as well as shape, so an all-int64 toy would warm the wrong ABI.
    result = dispatcher(
        np.array([2], dtype=i32),                 # pat_nstops
        np.array([1], dtype=i32),                 # pat_ntrips
        np.array([0], dtype=i64),                 # pat_stop_off
        np.array([0], dtype=i64),                 # pat_mat_off
        np.array([0, 1], dtype=i32),              # pat_stops
        np.array([0, 120], dtype=i32),            # pat_dep
        np.array([0, 120], dtype=i32),            # pat_arr
        np.array([0, 1, 1], dtype=i64),           # ras_off
        np.array([0], dtype=i32),                 # ras_pat
        np.array([0], dtype=i32),                 # ras_pos
        np.array([0, 0, 0], dtype=i64),           # tr_off
        np.empty(0, dtype=i32),                   # tr_to
        np.empty(0, dtype=i32),                   # tr_time
        np.array([2 ** 60, 60], dtype=i64),       # egress_sec
        0, 0, 0, 60, 2 ** 60,                     # first board + sentinels
    )
    if not isinstance(result, np.ndarray) or result.shape != (0, 12):
        raise RuntimeError(
            "synthetic planned one-tail warmup returned an unexpected result: "
            f"{getattr(result, 'shape', result)!r}")


def main():
    # Keep this import inside ``main`` so the synthetic ABI fixture can be unit-tested without
    # booting the service or loading local GTFS data.
    import server

    sr = server.sr
    lat, lon = PUBLIC_DEST
    if PUBLIC_CELL not in sr._RAPTOR.cell_index:
        raise RuntimeError(
            f"public warmup cell {PUBLIC_CELL!r} is absent; choose a new committed route hotspot")
    cid = PUBLIC_CELL
    olat, olon = sr.ORIGIN_LL[cid]
    speed = "med"
    scalar = server.config.WALK_KMH / server.WALK_SPEEDS[speed]
    steps = [
        ("compute", lambda: sr.compute_raptor(lat, lon, sr.DEFAULT_MAX_RIDES, speed, scalar)),
        ("variance", lambda: sr.raptor_mc(lat, lon, sr.DEFAULT_MAX_RIDES, speed, scalar)),
        ("pin", lambda: sr.itinerary_departafter(
            cid, olat, olon, lat, lon, sr.DEFAULT_MAX_RIDES, speed, scalar, pin=True)),
    ]
    for name, build in steps:
        started = time.perf_counter()
        result = build()
        if not isinstance(result, dict) or result.get("error"):
            raise RuntimeError(f"{name} warmup failed: {result!r}")
        print(f"[warm-numba] {name}: {(time.perf_counter() - started) * 1000:.0f}ms")

    # A successful endpoint can take a fallback branch and leave a claimed dispatcher cold. Make
    # the service restart fail loudly instead of letting the first visitor discover that omission.
    from core import raptor_numba as rn
    from core import raptor_planned_numba as rpn
    _warm_planned_one_tail_discovery(rpn.discover_one_tail_variants)
    required = {
        "reverse profile": rn._profile,
        "compact traced parents": rn._traced_compact,
        "planned selector": rn.select_planned_departafter_arith,
        "committed Monte Carlo": rn.montecarlo_committed,
        "retained-tail replay": rn.montecarlo_committed_from_tail,
        "planned committed extraction": rpn.extract_planned_committed_group,
        "planned one-tail discovery": rpn.discover_one_tail_variants,
    }
    missing = [name for name, dispatcher in required.items()
               if not getattr(dispatcher, "signatures", ())]
    if missing:
        raise RuntimeError("warmup did not exercise compiled signatures: " + ", ".join(missing))
    print("[warm-numba] compiled signatures verified: " + ", ".join(required))


if __name__ == "__main__":
    main()
