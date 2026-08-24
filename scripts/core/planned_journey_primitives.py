"""Pure helpers shared by the planned-journey tracer.

The functions in this module deliberately operate on the tracer's small, documented
tuple/dict boundary types.  They do not know about Flask, the server, cache state, or
the mutable ``DepartAfterJourneyTree`` object.  The class keeps compatibility wrappers
for the historical private method names; callers can therefore migrate incrementally
without changing itinerary tuples, ranking, or monkeypatch seams.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, MutableMapping, Sequence
from typing import Any


LegDict = MutableMapping[str, Any]
RawLeg = Sequence[Any]
Candidate = MutableMapping[str, Any]


def fold_first_visible_wait(out: list[LegDict]) -> None:
    """Fold the first transit wait into the preceding access walk.

    The physical walk and schedule allowance remain separate fields while ``sec`` stays
    equal to their sum.  If a zero-second access walk was omitted, an allowance-only
    access leg is inserted; geometry callers receive the historical ``("access",)`` tag.
    """
    for i, leg in enumerate(out):
        if leg.get("mode") != "transit":
            continue
        wait = int(leg.get("wait_sec", 0))
        if wait <= 0:
            return
        leg["wait_sec"] = 0
        for j in range(i - 1, -1, -1):
            if out[j].get("mode") == "walk":
                walk = out[j]
                physical = int(walk.get("physical_sec", walk.get("sec", 0)))
                allowance = int(walk.get("schedule_allowance_sec", 0)) + wait
                walk["physical_sec"] = physical
                walk["schedule_allowance_sec"] = allowance
                walk["sec"] = physical + allowance
                return
        first: LegDict = {
            "mode": "walk", "line": None, "sec": wait,
            "physical_sec": 0, "schedule_allowance_sec": wait,
        }
        if "segs" in leg:
            first["segs"] = [("access",)]
        out.insert(0, first)
        return


def attach_walk_truth(
    res: MutableMapping[str, Any],
    out: Iterable[LegDict],
    geom: list[LegDict] | None = None,
) -> None:
    """Copy planned physical/allowance walk fields onto formatted and geometry legs."""
    source = list(out)
    display_walks = [leg for leg in (res.get("legs") or ()) if leg.get("mode") == "walk"]
    if display_walks and not any("schedule_allowance_min" in leg for leg in display_walks):
        source_walks = [leg for leg in source
                        if leg.get("mode") == "walk" and int(leg.get("sec", 0)) > 0]
        if len(source_walks) == len(display_walks):
            for source_leg, leg in zip(source_walks, display_walks):
                if "schedule_allowance_sec" in source_leg:
                    leg["physical_min"] = int(source_leg.get("physical_sec", 0)) / 60.0
                    leg["schedule_allowance_min"] = (
                        int(source_leg["schedule_allowance_sec"]) / 60.0
                    )
    for leg_i, leg in enumerate(res.get("legs") or ()):
        if leg.get("mode") != "walk" or "schedule_allowance_min" not in leg:
            continue
        if geom is not None and leg_i < len(geom) and geom[leg_i].get("mode") == "walk":
            geom[leg_i]["physical_min"] = leg["physical_min"]
            geom[leg_i]["schedule_allowance_min"] = leg["schedule_allowance_min"]


def reconcile_planned_target(out: list[LegDict], total_sec: int, target_sec: int) -> int:
    """Apply a planned target residual without shortening physical walking.

    Positive residual is schedule allowance on the first access walk.  Negative residual
    consumes schedule allowance only; the returned value is any remaining inconsistency.
    """
    residual = int(target_sec) - int(total_sec)
    if residual > 0:
        for leg in out:
            if leg.get("mode") != "walk":
                if leg.get("mode") == "transit":
                    break
                continue
            physical = int(leg.get("physical_sec", leg.get("sec", 0)))
            allowance = int(leg.get("schedule_allowance_sec", 0)) + residual
            leg["physical_sec"] = physical
            leg["schedule_allowance_sec"] = allowance
            leg["sec"] = physical + allowance
            return 0
        access: LegDict = {
            "mode": "walk", "line": None, "sec": residual,
            "physical_sec": 0, "schedule_allowance_sec": residual,
        }
        if any("segs" in leg for leg in out):
            access["segs"] = [("access",)]
        out.insert(0, access)
        return 0

    excess = -residual
    if excess <= 0:
        return 0
    for leg in out:
        if excess <= 0:
            return 0
        if leg.get("mode") != "walk" or "schedule_allowance_sec" not in leg:
            continue
        allowance = int(leg["schedule_allowance_sec"])
        physical = int(leg.get("physical_sec", 0))
        take_allowance = min(excess, allowance)
        allowance -= take_allowance
        excess -= take_allowance
        leg["physical_sec"] = physical
        leg["schedule_allowance_sec"] = allowance
        leg["sec"] = physical + allowance
    return excess


def geom_route_label(geom: Iterable[MutableMapping[str, Any]] | None) -> str:
    """Return the ordered public transit-label string used for a geometry candidate."""
    names: list[str] = []
    for leg in geom or ():
        if leg.get("mode") != "transit":
            continue
        name = leg.get("name") or leg.get("line")
        if name:
            names.append(str(name))
    return " > ".join(names) if names else "walk only"


def geom_route_sig(geom: Iterable[MutableMapping[str, Any]] | None) -> tuple[tuple[str, str, int, int], ...]:
    """Return the stable display signature used after structural candidate collapse."""
    return tuple(
        (
            str(leg.get("mode") or ""),
            str(leg.get("name") or leg.get("line") or ""),
            int(leg.get("min") or 0),
            int(leg.get("wait") or 0),
        )
        for leg in (geom or ())
    )


def raw_final_walk_sec(raw: Iterable[RawLeg] | None) -> int:
    """Return the final egress/walk duration, or a large sentinel when absent."""
    raw_list = list(raw or ())
    if not raw_list:
        return 1 << 60
    last = raw_list[-1]
    return int(last[1]) if last[0] in ("egress", "walk") else 0


def raw_access_sec(raw: Iterable[RawLeg] | None) -> int:
    """Return physical seconds before the first transit board."""
    total = 0
    for leg in raw or ():
        if leg[0] == "ride":
            break
        if leg[0] in ("access", "walk", "walk_t", "egress"):
            total += max(0, int(leg[1]))
    return total


def raw_transfer_walk_sec(raw: Iterable[RawLeg] | None) -> int:
    """Return physical transfer-walk seconds."""
    return sum(max(0, int(leg[1])) for leg in raw or () if leg[0] == "walk_t")


def raw_transfer_count(raw: Iterable[RawLeg] | None) -> int:
    """Return transit rides minus one, floored at zero."""
    return max(0, sum(1 for leg in raw or () if leg[0] == "ride") - 1)


def raw_transit_sec(raw: Iterable[RawLeg] | None) -> int:
    """Return the sum of scheduled ride durations."""
    total = 0
    for leg in raw or ():
        if leg[0] == "ride":
            total += max(0, int(leg[3]) - int(leg[2]))
    return total


def planned_raw_total_sec(raw: Iterable[RawLeg] | None, latest_home: int) -> int:
    """Compute exact elapsed seconds using the historical implicit-wait clock."""
    t = int(latest_home)
    for leg in raw or ():
        if leg[0] in ("access", "walk", "walk_t", "egress"):
            t += int(leg[1])
        elif leg[0] == "ride":
            t = int(leg[3])
    return max(0, t - int(latest_home))


def alight_tail_better(
    finish: int,
    apos: int,
    eg: int,
    best: tuple[int, int, int, int] | None,
) -> bool:
    """Compare equal-finish alight candidates using the existing deterministic tie-break."""
    if best is None:
        return True
    best_finish, best_apos, _best_stop, best_eg = best
    return (int(finish), int(eg), -int(apos)) < (
        int(best_finish), int(best_eg), -int(best_apos)
    )


def planned_candidate_quality(
    cand: Candidate,
    *,
    raw_total_sec: Callable[[Iterable[RawLeg] | None, int], int] = planned_raw_total_sec,
    access_sec: Callable[[Iterable[RawLeg] | None], int] = raw_access_sec,
    transfer_count: Callable[[Iterable[RawLeg] | None], int] = raw_transfer_count,
    transfer_walk_sec: Callable[[Iterable[RawLeg] | None], int] = raw_transfer_walk_sec,
    final_walk_sec: Callable[[Iterable[RawLeg] | None], int] = raw_final_walk_sec,
) -> tuple[Any, ...]:
    """Return the exact planned-candidate quality tuple.

    Helper callables are explicit so the compatibility wrapper can preserve the old
    class-level monkeypatch seams while this function remains independently testable.
    """
    raw = cand.get("raw") or ()
    home = int(cand.get("home", 0))
    metric_sec = int(cand.get("metric_sec", raw_total_sec(raw, home)))
    board_anchor = int(cand.get("board_anchor", home + access_sec(raw)))
    return (
        metric_sec,
        int(cand.get("total", math.ceil(metric_sec / 60.0))),
        -board_anchor,
        access_sec(raw),
        transfer_count(raw),
        transfer_walk_sec(raw),
        final_walk_sec(raw),
        cand.get("route_key") or (),
    )


def planned_candidate_cheap_rank(
    cand: Candidate,
    *,
    quality: Callable[[Candidate], tuple[Any, ...]] = planned_candidate_quality,
) -> tuple[Any, ...]:
    """Return the geometry-free planned candidate rank."""
    return quality(cand)


def planned_candidate_rank(
    cand: Candidate,
    *,
    quality: Callable[[Candidate], tuple[Any, ...]] = planned_candidate_quality,
) -> tuple[Any, ...]:
    """Return the full planned candidate rank including display signature."""
    return (*quality(cand), cand.get("sig", ()))


def planned_candidate_better(
    cand: Candidate,
    cur: Candidate | None,
    *,
    rank: Callable[[Candidate], tuple[Any, ...]] = planned_candidate_rank,
) -> bool:
    """Return whether ``cand`` wins the historical deterministic candidate ordering."""
    return cur is None or rank(cand) < rank(cur)
