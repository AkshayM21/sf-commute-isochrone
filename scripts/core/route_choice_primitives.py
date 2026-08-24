"""Pure route-choice measurements and recommendation primitives.

These helpers operate only on the small route/leg dictionaries emitted by the RAPTOR
builders.  They deliberately do not know about the engine, caches, geometry providers,
family discovery, dominance, or breadth selection.  ``server_raptor`` keeps its historical
private names as compatibility wrappers and supplies those wrappers as callbacks where a
ranking helper has an existing server-owned tie-break seam.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, MutableMapping, Sequence
from typing import Any


Leg = MutableMapping[str, Any]
Alternative = MutableMapping[str, Any]


def legs_have_transit(legs: Iterable[Leg] | None) -> bool:
    return any((leg or {}).get("mode") == "transit" for leg in (legs or ()))


def route_sig(legs: Iterable[Leg] | None) -> tuple[tuple[str, str, int, int], ...]:
    """Signature of the displayed route fields (excluding geometry)."""
    out = []
    for leg in legs or ():
        if not leg:
            continue
        out.append((leg.get("mode") or "",
                    str(leg.get("name") or leg.get("line") or ""),
                    int(leg.get("min") or 0),
                    int(leg.get("wait") or 0)))
    return tuple(out)


def leg_geom_sig(leg: Leg | None) -> str:
    points = (leg or {}).get("pts") or []
    if not points:
        return ""
    indexes = [0, (len(points) - 1) // 2, len(points) - 1]
    out = []
    for index in dict.fromkeys(indexes):
        try:
            lat, lon = points[index]
            out.append(f"{round(float(lat) * 10000)}_{round(float(lon) * 10000)}")
        except Exception:
            return ""
    return "_".join(out)


def route_trace_sig(
    legs: Iterable[Leg] | None,
    *,
    leg_geom: Callable[[Leg | None], str] = leg_geom_sig,
) -> tuple[tuple[str, str, int, int, str], ...]:
    out = []
    for leg in legs or ():
        if not leg:
            continue
        out.append((leg.get("mode") or "",
                    str(leg.get("name") or leg.get("line") or ""),
                    int(leg.get("min") or 0),
                    int(leg.get("wait") or 0),
                    leg_geom(leg)))
    return tuple(out)


def route_label(legs: Iterable[Leg] | None) -> str:
    names = []
    for leg in legs or ():
        if not leg or leg.get("mode") != "transit":
            continue
        name = leg.get("name") or leg.get("line")
        if name:
            names.append(str(name))
    return " > ".join(names)


def alt_total(alt: Alternative) -> int:
    nested = alt.get("typical") or alt.get("best") or {}
    return int(nested.get("total", alt.get("min", alt.get("total", 10 ** 6))))


def alt_metric_total(alt: Alternative, metric: str = "r") -> int:
    """Return the displayed total for the selected practical time metric."""
    if metric == "b":
        nested = alt.get("best") or alt.get("typical") or {}
        value = nested.get("total", alt.get("min", alt.get("total")))
    else:
        value = alt.get("real")
        if value is None:
            nested = alt.get("typical") or alt.get("best") or {}
            value = nested.get("total", alt.get("min", alt.get("total")))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 10 ** 6


def leg_name(leg: Leg | None) -> str:
    return str((leg or {}).get("name") or (leg or {}).get("line") or "")


def alt_display_legs(alt: Alternative) -> list[Leg]:
    for key in ("typical", "best"):
        legs = ((alt.get(key) or {}).get("legs") or [])
        if legs:
            return legs
    for key in ("legs", "geom"):
        legs = alt.get(key) or []
        if legs:
            return legs
    branch = alt.get("_branch") or {}
    itinerary = branch.get("it") or {}
    return itinerary.get("geom") or itinerary.get("legs") or []


def alt_transit_legs(
    alt: Alternative,
    *,
    display_legs: Callable[[Alternative], list[Leg]] = alt_display_legs,
) -> list[Leg]:
    return [leg for leg in display_legs(alt) if (leg or {}).get("mode") == "transit"]


def alt_raw_legs(alt: Alternative) -> Sequence[Any]:
    branch = alt.get("_branch") or {}
    return branch.get("raw") or []


def leg_physical_walk_min(leg: Leg | None) -> float:
    """Physical walking minutes, excluding folded schedule allowance."""
    leg = leg or {}
    value = leg.get("physical_min", leg.get("min", 0))
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def alt_access_walk_min(
    alt: Alternative,
    *,
    display_legs: Callable[[Alternative], list[Leg]] = alt_display_legs,
    raw_legs: Callable[[Alternative], Sequence[Any]] = alt_raw_legs,
    physical_walk_min: Callable[[Leg | None], float] = leg_physical_walk_min,
) -> float:
    # Prefer explicit physical walk truth when present; older payloads use raw/display fallbacks.
    legs = display_legs(alt)
    physical = [leg for leg in legs if (leg or {}).get("mode") == "walk"
                and (leg or {}).get("physical_min") is not None]
    if physical:
        total = 0
        for leg in legs:
            if (leg or {}).get("mode") == "transit":
                break
            if (leg or {}).get("mode") == "walk":
                total += physical_walk_min(leg)
        return total
    raw = raw_legs(alt)
    if raw:
        sec = 0
        for leg in raw:
            if leg[0] == "ride":
                break
            if leg[0] in ("access", "walk", "walk_t", "egress"):
                sec += int(leg[1])
        return sec / 60.0
    total = 0
    for leg in display_legs(alt):
        if leg.get("mode") == "transit":
            break
        if leg.get("mode") == "walk":
            total += int(leg.get("min") or 0)
    return total


def alt_final_walk_min(
    alt: Alternative,
    *,
    display_legs: Callable[[Alternative], list[Leg]] = alt_display_legs,
    raw_legs: Callable[[Alternative], Sequence[Any]] = alt_raw_legs,
    physical_walk_min: Callable[[Leg | None], float] = leg_physical_walk_min,
) -> float:
    legs = display_legs(alt)
    for leg in reversed(legs):
        if (leg or {}).get("mode") == "walk":
            return physical_walk_min(leg)
        if (leg or {}).get("mode") == "transit":
            break
    raw = raw_legs(alt)
    if raw:
        last = raw[-1]
        if last[0] in ("egress", "walk"):
            return int(last[1]) / 60.0
        return 0.0
    for leg in reversed(display_legs(alt)):
        if leg.get("mode") == "walk":
            return physical_walk_min(leg)
        if leg.get("mode") == "transit":
            break
    return 0.0


def alt_physical_walk_min(
    alt: Alternative,
    *,
    display_legs: Callable[[Alternative], list[Leg]] = alt_display_legs,
    raw_legs: Callable[[Alternative], Sequence[Any]] = alt_raw_legs,
    physical_walk_min: Callable[[Leg | None], float] = leg_physical_walk_min,
) -> float:
    legs = display_legs(alt)
    if any((leg or {}).get("physical_min") is not None for leg in legs):
        return sum(physical_walk_min(leg) for leg in legs
                   if (leg or {}).get("mode") == "walk")
    raw = raw_legs(alt)
    if raw:
        return sum(int(leg[1]) for leg in raw
                   if leg and leg[0] in ("access", "walk", "walk_t", "egress")) / 60.0
    return sum(physical_walk_min(leg) for leg in legs
               if (leg or {}).get("mode") == "walk")


def alt_exact_seconds(
    alt: Alternative,
    *,
    raw_legs: Callable[[Alternative], Sequence[Any]] = alt_raw_legs,
    total: Callable[[Alternative], int] = alt_total,
) -> int:
    """Unrounded door-to-door seconds when a planned raw trace is available."""
    raw = raw_legs(alt)
    branch = alt.get("_branch") or {}
    try:
        metric_sec = branch.get("metric_sec")
        if metric_sec is not None:
            return max(0, int(metric_sec))
    except (TypeError, ValueError):
        pass
    if raw and branch.get("home") is not None:
        try:
            home = int(branch["home"])
            clock = home
            for leg in raw:
                if leg[0] in ("access", "walk", "walk_t", "egress"):
                    clock += int(leg[1])
                elif leg[0] == "ride":
                    clock = int(leg[3])
            return max(0, clock - home)
        except (IndexError, TypeError, ValueError):
            pass
    return total(alt) * 60


def alt_transfers(
    alt: Alternative,
    *,
    transit_legs: Callable[[Alternative], list[Leg]] = alt_transit_legs,
) -> int:
    return max(0, len(transit_legs(alt)) - 1)


def alt_fragility(alt: Alternative) -> int:
    try:
        return max(0, int(alt.get("frag", 0) or 0))
    except (TypeError, ValueError):
        return 0


def alt_latest_board_anchor(alt: Alternative, *, raw_legs=alt_raw_legs) -> int:
    for leg in raw_legs(alt):
        if leg and leg[0] == "ride" and len(leg) >= 3:
            try:
                return int(leg[2])
            except (TypeError, ValueError):
                break
    return -1


def is_token_transit_long_walk(
    alt: Alternative,
    *,
    transit_legs: Callable[[Alternative], list[Leg]] = alt_transit_legs,
    final_walk_min: Callable[[Alternative], float] = alt_final_walk_min,
) -> bool:
    transit = transit_legs(alt)
    ride_min = sum(max(0, int((leg or {}).get("min") or 0)) for leg in transit)
    return bool(transit and ride_min <= 2 and final_walk_min(alt) >= 8)


def _default_selection_tie_key(alt: Alternative) -> tuple[str, str]:
    return ("rendered-route", repr(route_trace_sig(alt_display_legs(alt))))


def alt_quality_rank(
    alt: Alternative,
    *,
    total: Callable[[Alternative], int] = alt_total,
    exact_seconds: Callable[[Alternative], int] = alt_exact_seconds,
    access_walk_min: Callable[[Alternative], float] = alt_access_walk_min,
    transfers: Callable[[Alternative], int] = alt_transfers,
    final_walk_min: Callable[[Alternative], float] = alt_final_walk_min,
    fragility: Callable[[Alternative], int] = alt_fragility,
    latest_board_anchor: Callable[[Alternative], int] = alt_latest_board_anchor,
    token_transit_long_walk: Callable[[Alternative], bool] = is_token_transit_long_walk,
    selection_tie_key: Callable[[Alternative], Any] = _default_selection_tie_key,
) -> tuple[Any, ...]:
    return (
        total(alt),
        exact_seconds(alt),
        access_walk_min(alt),
        transfers(alt),
        final_walk_min(alt),
        fragility(alt),
        -latest_board_anchor(alt),
        1 if token_transit_long_walk(alt) else 0,
        selection_tie_key(alt),
    )


def alt_recommendation_rank(
    alt: Alternative,
    metric: str = "r",
    *,
    metric_total: Callable[[Alternative, str], int] = alt_metric_total,
    physical_walk_min: Callable[[Alternative], float] = alt_physical_walk_min,
    transfers: Callable[[Alternative], int] = alt_transfers,
    fragility: Callable[[Alternative], int] = alt_fragility,
    exact_seconds: Callable[[Alternative], int] = alt_exact_seconds,
    latest_board_anchor: Callable[[Alternative], int] = alt_latest_board_anchor,
    selection_tie_key: Callable[[Alternative], Any] = _default_selection_tie_key,
) -> tuple[Any, ...]:
    return (
        metric_total(alt, metric),
        physical_walk_min(alt),
        transfers(alt),
        fragility(alt),
        exact_seconds(alt),
        -latest_board_anchor(alt),
        selection_tie_key(alt),
    )


def recommend_route_choice(
    primary: Alternative | None,
    alternatives: Iterable[Alternative] | None,
    metric: str = "r",
    *,
    recommendation_rank: Callable[[Alternative, str], tuple[Any, ...]] = alt_recommendation_rank,
) -> Alternative | None:
    options = ([primary] if primary is not None else []) + list(alternatives or ())
    return min(options, key=lambda option: recommendation_rank(option, metric)) if options else None


def recommend_route_choices(
    primary: Alternative | None,
    alternatives: Iterable[Alternative] | None,
    *,
    recommendation: Callable[[Alternative | None, Iterable[Alternative] | None, str],
                             Alternative | None] = recommend_route_choice,
) -> dict[str, Alternative | None]:
    return {metric: recommendation(primary, alternatives, metric) for metric in ("r", "b")}


# Private aliases make the extracted module convenient for focused compatibility tests while the
# server's historical private names remain wrappers (rather than aliases) for monkeypatch seams.
_legs_have_transit = legs_have_transit
_route_sig = route_sig
_route_trace_sig = route_trace_sig
_route_label = route_label
_alt_total = alt_total
_alt_metric_total = alt_metric_total
_leg_name = leg_name
_alt_display_legs = alt_display_legs
_alt_transit_legs = alt_transit_legs
_leg_geom_sig = leg_geom_sig
_alt_raw_legs = alt_raw_legs
_alt_access_walk_min = alt_access_walk_min
_alt_final_walk_min = alt_final_walk_min
_leg_physical_walk_min = leg_physical_walk_min
_alt_physical_walk_min = alt_physical_walk_min
_alt_exact_seconds = alt_exact_seconds
_alt_transfers = alt_transfers
_alt_fragility = alt_fragility
_alt_latest_board_anchor = alt_latest_board_anchor
_is_token_transit_long_walk = is_token_transit_long_walk
_alt_quality_rank = alt_quality_rank
_alt_recommendation_rank = alt_recommendation_rank
_recommend_route_choice = recommend_route_choice
_recommend_route_choices = recommend_route_choices
_alt_representative_rank = alt_quality_rank
