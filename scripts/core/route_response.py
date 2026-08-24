"""Pure response-formatting helpers for traced RAPTOR journeys.

The helpers in this module operate only on the small leg dictionaries, raw arrays, and
scalar values passed to them.  They deliberately do not know about ``JourneyTree`` or
any engine/runtime state.  ``raptor_journey`` keeps compatibility wrappers for its
historical private/public names, including the old monkeypatch seams.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence
from typing import Any

import numpy as np


_TINY_HOP_MIN = 2.0
EGRESS_INF = np.int64(1 << 40)

Leg = MutableMapping[str, Any]


def reconcile_legs(legs: list[Leg], total_min: int) -> dict[str, Any]:
    """Reconcile rounded leg components to the integer itinerary total.

    ``legs`` is the already-rounded integer leg list (``mode``, ``line``, ``min`` and
    optional ``wait``).  Zero-minute walk legs are dropped, then any residual is
    applied to the final walk (or represented by a positive residual walk), preserving
    the historical response schema and mutation behavior.
    """
    total_min = int(total_min)
    legs = [l for l in legs if not (l["mode"] == "walk" and l["min"] <= 0)]
    rides_kept = sum(1 for l in legs if l["mode"] != "walk")
    cur = sum(l["min"] for l in legs) + sum(l.get("wait", 0) for l in legs)
    diff = total_min - cur
    if diff != 0:
        walks = [l for l in legs if l["mode"] == "walk"]
        if walks:
            walks[-1]["min"] = max(0, walks[-1]["min"] + diff)
        elif diff > 0:
            legs.append({"mode": "walk", "line": None, "min": diff})
        legs = [l for l in legs if not (l["mode"] == "walk" and l["min"] <= 0)]
    return {"total": total_min, "xfers": max(0, rides_kept - 1), "legs": legs}


def format_legs(
    out: Sequence[Leg],
    total_min: int,
    *,
    reconcile_fn: Callable[[list[Leg], int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Round folded exact-second legs and reconcile them to ``total_min``.

    This is the former ``JourneyTree._format`` implementation.  ``reconcile_fn`` is
    injectable so the compatibility method can continue honoring monkeypatches of the
    historical ``raptor_journey.reconcile_legs`` symbol.
    """
    if reconcile_fn is None:
        reconcile_fn = reconcile_legs

    # Round to minutes while keeping displayed components close to their real seconds.
    # Largest-remainder rounding is followed by reconciliation because callers may
    # provide a target minute that is not simply ceil(total seconds).
    legs: list[Leg] = []
    comps: list[tuple[Leg, str, float]] = []
    for l in out:
        if l["mode"] == "walk":
            d: Leg = {"mode": "walk", "line": None, "min": 0}
            # Planned depart-after traces split the first walk into actual street time
            # and controllable pre-board allowance. Keep that identity on the formatted
            # dict, even when a zero-rounded access walk is later dropped.
            if "schedule_allowance_sec" in l:
                d["physical_min"] = int(l.get("physical_sec", 0)) / 60.0
                d["schedule_allowance_min"] = int(l["schedule_allowance_sec"]) / 60.0
            comps.append((d, "min", float(l["sec"]) / 60.0))
        else:
            d = {"mode": "transit", "line": l["line"], "min": 0, "wait": 0}
            comps.append((d, "min", float(l["sec"]) / 60.0))
            comps.append((d, "wait", float(l["wait_sec"]) / 60.0))
        if "segs" in l:
            d["segs"] = l["segs"]
        legs.append(d)

    base_sum = 0
    ranked: list[tuple[float, int, Leg, str]] = []
    for idx, (d, field, exact) in enumerate(comps):
        base = int(np.floor(exact))
        d[field] = base
        base_sum += base
        if exact > 1e-9:
            ranked.append((-(exact - base), idx, d, field))
    remaining = max(0, int(total_min) - base_sum)
    for _frac, _idx, d, field in sorted(ranked)[:min(remaining, len(ranked))]:
        d[field] += 1
    return reconcile_fn(legs, total_min)


def push_walk(out: list[Leg], sec: int, seg: Any = None) -> None:
    """Append a positive walk duration, merging it into an adjacent walk leg."""
    if sec <= 0:
        return
    if out and out[-1]["mode"] == "walk":
        out[-1]["sec"] += sec
        if seg is not None:
            out[-1].setdefault("segs", []).append(seg)
    else:
        d: Leg = {"mode": "walk", "line": None, "sec": sec}
        if seg is not None:
            d["segs"] = [seg]
        out.append(d)


_push_walk = push_walk


def footpath_sec(tr_off, tr_to, tr_time, s: int, j: int) -> int:
    """Return the directed transfer duration, or zero for same/missing edges."""
    for k in range(int(tr_off[s]), int(tr_off[s + 1])):
        if int(tr_to[k]) == j:
            return int(tr_time[k])
    return 0


_footpath_sec = footpath_sec


def min_overshoot_alight(
    pat_arr,
    pat_stops,
    eg_sec,
    trow: int,
    sbase: int,
    bpos: int,
    ns: int,
    apos: int,
    nd_egress: int,
) -> tuple[int, int]:
    """Choose the no-overshoot final-ride alight and its egress duration.

    The candidate minimizing ``arrival + egress walk`` wins. Equal finishes retain
    the historical shorter-walk, then later-position tie-break.
    """
    best_p = int(apos)
    best_w = int(eg_sec[int(pat_stops[sbase + best_p])])
    if best_w >= EGRESS_INF:
        best_w = int(nd_egress) if nd_egress >= 0 else 0
    best_arrW = int(pat_arr[trow + best_p]) + best_w
    for p in range(int(bpos) + 1, int(ns)):
        w = int(eg_sec[int(pat_stops[sbase + p])])
        if w >= EGRESS_INF:
            continue
        cand = int(pat_arr[trow + p]) + w
        if cand < best_arrW or (cand == best_arrW and (w < best_w or (w == best_w and p > best_p))):
            best_arrW = cand
            best_p = p
            best_w = w
    return best_p, best_w


_min_overshoot_alight = min_overshoot_alight
