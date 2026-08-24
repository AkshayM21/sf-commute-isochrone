"""Pure structural identity helpers for rendered route alternatives.

The functions in this module operate on route/leg dictionaries only.  Runtime route-id
resolution, display extraction, and the historical private names remain in ``server_raptor``
as callbacks/wrappers, so this module has no engine, cache, Flask, or mutable runtime state.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, MutableMapping
from typing import Any

from . import route_choice_primitives


Leg = MutableMapping[str, Any]
Alternative = MutableMapping[str, Any]


def leg_service_sig(leg: Leg | None) -> str:
    vals = [str((leg or {}).get(key) or "") for key in ("feed", "tmode")]
    return "|".join(value for value in vals if value)


def leg_route_id(leg: Leg | None, *, resolver: Callable[[Leg], str] | None = None) -> str:
    """Return an explicit route id, or ask an optional pure caller-supplied resolver."""
    leg = leg or {}
    explicit = leg.get("route_id") or leg.get("routeId")
    if explicit not in (None, ""):
        return str(explicit)
    if resolver is None:
        return ""
    value = resolver(leg)
    return "" if value in (None, "") else str(value)


def leg_service_meta(
    leg: Leg | None,
    *,
    route_id: Callable[[Leg], str] | None = None,
) -> dict[str, str]:
    """Return readable structural service identity and display metadata."""
    leg = leg or {}
    feed = str(leg.get("feed") or "")
    mode = str(leg.get("tmode") or "")
    name = leg_name(leg) or "Transit"
    resolved = leg_route_id(leg, resolver=route_id)
    if resolved:
        key = f"service:feed={feed or 'unknown'};mode={mode or 'unknown'};route={resolved}"
    else:
        key = f"service:feed={feed or 'unknown'};mode={mode or 'unknown'};display={name}"
    return {"key": key, "name": name, "feed": feed, "mode": mode}


def leg_geom_sig(leg: Leg | None) -> str:
    return route_choice_primitives.leg_geom_sig(leg)


def leg_name(leg: Leg | None) -> str:
    return route_choice_primitives.leg_name(leg)


def leg_dir_sig(leg: Leg | None) -> str:
    pts = (leg or {}).get("pts") or []
    parsed = []
    for point in pts:
        try:
            parsed.append((float(point[0]), float(point[1])))
        except Exception:
            continue
    if len(parsed) < 2:
        return ""
    lat0, lon0 = parsed[0]
    nxt = next(((lat, lon) for lat, lon in parsed[1:]
                if abs(lat - lat0) + abs(lon - lon0) > 1e-7), None)
    if nxt is None:
        return ""
    lat1, lon1 = nxt
    board = f"{round(lat0 * 1000)}_{round(lon0 * 1000)}"
    dy = lat1 - lat0
    dx = (lon1 - lon0) * math.cos(math.radians((lat0 + lat1) / 2.0))
    angle = math.atan2(dx, dy) % (2.0 * math.pi)
    heading = int(round(angle / (2.0 * math.pi / 16.0))) % 16
    return f"{board}_h{heading}"


def leg_boarding_profile(leg: Leg | None) -> dict[str, Any] | None:
    pts = []
    for point in (leg or {}).get("pts") or ():
        try:
            parsed = (float(point[0]), float(point[1]))
        except Exception:
            continue
        if not pts or parsed != pts[-1]:
            pts.append(parsed)
    if len(pts) < 2:
        return None
    lat0, lon0 = pts[0]
    nxt = next(((lat, lon) for lat, lon in pts[1:]
                if abs(lat - lat0) + abs(lon - lon0) > 1e-7), None)
    if nxt is None:
        return None
    lat1, lon1 = nxt
    north = (lat1 - lat0) * 111_320.0
    east = ((lon1 - lon0) * 111_320.0
            * math.cos(math.radians((lat1 + lat0) / 2.0)))
    norm = math.hypot(east, north)
    if norm <= 0:
        return None
    return {"service": leg_service_sig(leg), "board": (lat0, lon0),
            "heading": (east / norm, north / norm), "prefix": tuple(pts[:4])}


def leg_boarding_direction_guard(leg: Leg | None) -> str:
    profile = leg_boarding_profile(leg)
    if profile is None:
        return ""
    east, north = profile["heading"]
    angle = math.atan2(east, north) % (2.0 * math.pi)
    return f"h{int(round(angle / (2.0 * math.pi / 8.0))) % 8}"


def point_distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat = math.radians((a[0] + b[0]) / 2.0)
    north = (a[0] - b[0]) * 111_320.0
    east = (a[1] - b[1]) * 111_320.0 * math.cos(lat)
    return math.hypot(east, north)


def point_segment_distance_m(point, start, end) -> float:
    lat = math.radians((point[0] + start[0] + end[0]) / 3.0)
    scale_x = 111_320.0 * math.cos(lat)
    px, py = point[1] * scale_x, point[0] * 111_320.0
    ax, ay = start[1] * scale_x, start[0] * 111_320.0
    bx, by = end[1] * scale_x, end[0] * 111_320.0
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def prefixes_overlap(a, b, radius_m: float = 120.0, min_overlap_m: float = 200.0) -> bool:
    def nearest(point, path):
        if len(path) < 2:
            return min((point_distance_m(point, p) for p in path), default=10 ** 9)
        return min(point_segment_distance_m(point, path[i], path[i + 1])
                   for i in range(len(path) - 1))

    def covered_length(path, other):
        covered = 0.0
        for i in range(len(path) - 1):
            start, end = path[i], path[i + 1]
            midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
            if nearest(midpoint, other) <= radius_m:
                covered += point_distance_m(start, end)
        return covered

    return (covered_length(a, b) >= min_overlap_m
            and covered_length(b, a) >= min_overlap_m)


def same_boarding_profiles(pa, pb, a_name: str = "", b_name: str = "") -> bool:
    if pa is None or pb is None or pa["service"] != pb["service"]:
        return False
    board_distance = point_distance_m(pa["board"], pb["board"])
    dot = pa["heading"][0] * pb["heading"][0] + pa["heading"][1] * pb["heading"][1]
    same_line = bool(a_name and a_name == b_name)
    if same_line and board_distance <= 300.0 and dot >= math.cos(math.radians(100.0)):
        return True
    if board_distance > 150.0:
        return False
    if dot < math.cos(math.radians(40.0)):
        return False
    return prefixes_overlap(pa["prefix"], pb["prefix"])


def same_boarding_corridor(a: Leg | None, b: Leg | None) -> bool:
    return same_boarding_profiles(leg_boarding_profile(a), leg_boarding_profile(b),
                                  leg_name(a), leg_name(b))


def leg_corridor_sig(leg: Leg | None) -> str:
    service = leg_service_sig(leg)
    direction = leg_dir_sig(leg)
    if not direction:
        return ""
    return f"{service or 'service'}|board:{direction}"


def alt_display_legs(a: Alternative) -> list[Leg]:
    return route_choice_primitives.alt_display_legs(a)


def alt_transit_legs(
    a: Alternative,
    *,
    display_legs: Callable[[Alternative], list[Leg]] = alt_display_legs,
) -> list[Leg]:
    return [leg for leg in display_legs(a) if (leg or {}).get("mode") == "transit"]


def alt_corridor_sig(
    a: Alternative,
    *,
    transit_legs: Callable[[Alternative], list[Leg]] = alt_transit_legs,
) -> str:
    legs = transit_legs(a)
    return leg_corridor_sig(legs[0]) if legs else ""


def discover_family_keys(
    options: Iterable[Alternative] | None,
    *,
    transit_legs: Callable[[Alternative], list[Leg]] = alt_transit_legs,
    display_legs: Callable[[Alternative], list[Leg]] = alt_display_legs,
    same_corridor: Callable[[Leg | None, Leg | None], bool] = same_boarding_corridor,
    same_corridor_default: Callable[[Leg | None, Leg | None], bool] = same_boarding_corridor,
    same_profile: Callable[[Any, Any, str, str], bool] = same_boarding_profiles,
    route_id: Callable[[Leg], str] | None = None,
    geometry_sig: Callable[[Leg | None], str] = leg_geom_sig,
    total: Callable[[Alternative], int] = route_choice_primitives.alt_total,
) -> dict[int, str]:
    options = list(options or ())
    first_legs = [next(iter(transit_legs(option)), None) for option in options]
    profiles = [leg_boarding_profile(leg) for leg in first_legs]
    services = [leg_service_sig(leg) if leg is not None else None for leg in first_legs]
    names = [leg_name(leg) if leg is not None else "" for leg in first_legs]
    descriptors = []
    for i, leg in enumerate(first_legs):
        pts = []
        for point in (leg or {}).get("pts") or ():
            try:
                pts.append(f"{float(point[0]):.6f},{float(point[1]):.6f}")
            except Exception:
                continue
        structural = "|".join(pts)
        descriptor = (f"{services[i]}|path:{structural}" if structural
                      else f"line:{leg_name(leg) if leg else 'other'}")
        descriptors.append(descriptor)
    components = []
    components_by_service = {}
    raw_corridor_override = same_corridor is not same_corridor_default

    for i in sorted(range(len(options)), key=lambda idx: (descriptors[idx], idx)):
        service = services[i]
        service_components = components_by_service.setdefault(service, [])
        compatible = [component for component in service_components if all(
            profiles[i] is not None and profiles[j] is not None and (
                same_corridor(first_legs[i], first_legs[j]) if raw_corridor_override else
                same_profile(profiles[i], profiles[j], names[i], names[j]))
            for j in component
        )]
        if compatible:
            min(compatible, key=lambda component: min(descriptors[j] for j in component)).append(i)
        else:
            component = [i]
            components.append(component)
            service_components.append(component)
    records = []
    for indices in components:
        members = "\x1f".join(sorted(descriptors[i] for i in indices))

        def route_structure(option):
            return tuple(
                (str(leg.get("mode") or ""), leg_service_sig(leg),
                 leg_route_id(leg, resolver=route_id), int(leg.get("min") or 0),
                 int(leg.get("wait") or 0), geometry_sig(leg))
                for leg in display_legs(option) if leg
            )

        option_sigs = tuple(sorted((total(options[i]), route_structure(options[i]))
                                   for i in indices))
        records.append({"indices": indices, "order": (members, option_sigs, min(indices))})
    ordered_records = sorted(records, key=lambda record: record["order"])
    keys = {}
    for ordinal, record in enumerate(ordered_records, 1):
        for i in record["indices"]:
            keys[id(options[i])] = f"corridor:{ordinal}"
    return keys


def alt_family_key(
    a: Alternative,
    context=None,
    *,
    transit_legs: Callable[[Alternative], list[Leg]] = alt_transit_legs,
) -> str:
    if context and id(a) in context:
        return context[id(a)]
    legs = transit_legs(a)
    first = leg_name(legs[0]) if legs else ""
    sig = alt_corridor_sig(a, transit_legs=transit_legs)
    return f"corridor:{sig}" if sig else (first or "other")


def leg_arrival_sig(leg: Leg | None) -> str:
    pts = []
    for point in (leg or {}).get("pts") or ():
        try:
            parsed = (float(point[0]), float(point[1]))
        except Exception:
            continue
        if not pts or parsed != pts[-1]:
            pts.append(parsed)
    if len(pts) >= 2:
        end_lat, end_lon = pts[-1]
        prev_lat, prev_lon = pts[-2]
        north = (end_lat - prev_lat) * 111_320.0
        east = ((end_lon - prev_lon) * 111_320.0
                * math.cos(math.radians((end_lat + prev_lat) / 2.0)))
        angle = math.atan2(east, north) % (2.0 * math.pi)
        heading = int(round(angle / (2.0 * math.pi / 16.0))) % 16
        endpoint = f"{round(end_lat * 10000)}_{round(end_lon * 10000)}"
        return f"{leg_service_sig(leg) or 'service'}|arrive:{endpoint}_h{heading}"
    return f"{leg_service_sig(leg) or 'service'}|line:{leg_name(leg) or 'transit'}"


def branch_key_for_transit_legs(legs: Iterable[Leg] | None) -> str:
    legs = list(legs or ())
    if not legs:
        return "walk:only"
    if len(legs) == 1:
        return f"walk:{leg_arrival_sig(legs[0])}"
    return f"tail:{leg_arrival_sig(legs[-1])}"


def alt_branch_key(a: Alternative, fam=None, *, transit_legs=alt_transit_legs) -> str:
    return branch_key_for_transit_legs(transit_legs(a))


def alt_slot_legs(
    a: Alternative,
    slot: str,
    *,
    display_legs: Callable[[Alternative], list[Leg]] = alt_display_legs,
) -> list[Leg]:
    legs = ((a.get(slot) or {}).get("legs") or [])
    return legs or display_legs(a)


def journey_choice_key(
    legs: Iterable[Leg] | None,
    *,
    service_meta: Callable[[Leg], dict[str, str]] = leg_service_meta,
    direction_guard: Callable[[Leg], str] = leg_boarding_direction_guard,
) -> tuple[Any, str]:
    transit = [leg for leg in legs or () if (leg or {}).get("mode") == "transit"]
    sequence = tuple(service_meta(leg)["key"] for leg in transit)
    direction = direction_guard(transit[0]) if transit else ""
    return (direction, sequence), branch_key_for_transit_legs(transit)


def journey_choice_bucket(
    legs: Iterable[Leg] | None,
    *,
    service_meta: Callable[[Leg], dict[str, str]] = leg_service_meta,
) -> tuple[Any, str]:
    transit = [leg for leg in legs or () if (leg or {}).get("mode") == "transit"]
    return (tuple(service_meta(leg)["key"] for leg in transit),
            branch_key_for_transit_legs(transit))


def alt_choice_key(
    a: Alternative,
    *,
    slot_legs: Callable[[Alternative, str], list[Leg]] = alt_slot_legs,
    service_meta: Callable[[Leg], dict[str, str]] = leg_service_meta,
    direction_guard: Callable[[Leg], str] = leg_boarding_direction_guard,
) -> tuple[Any, Any]:
    return (journey_choice_key(slot_legs(a, "best"), service_meta=service_meta,
                               direction_guard=direction_guard),
            journey_choice_key(slot_legs(a, "typical"), service_meta=service_meta,
                               direction_guard=direction_guard))


def public_choice_key(a: Alternative, *, choice_key: Callable[[Alternative], Any] = alt_choice_key) -> str:
    return "choice:" + json.dumps(choice_key(a), separators=(",", ":"), ensure_ascii=True)


def alt_choice_bucket(
    a: Alternative,
    *,
    slot_legs: Callable[[Alternative, str], list[Leg]] = alt_slot_legs,
    service_meta: Callable[[Leg], dict[str, str]] = leg_service_meta,
) -> tuple[Any, Any]:
    return (journey_choice_bucket(slot_legs(a, "best"), service_meta=service_meta),
            journey_choice_bucket(slot_legs(a, "typical"), service_meta=service_meta))


def journey_choice_equivalent(
    left_legs: Iterable[Leg] | None,
    right_legs: Iterable[Leg] | None,
    *,
    service_meta: Callable[[Leg], dict[str, str]] = leg_service_meta,
    direction_guard: Callable[[Leg], str] = leg_boarding_direction_guard,
) -> bool:
    left = [leg for leg in left_legs or () if (leg or {}).get("mode") == "transit"]
    right = [leg for leg in right_legs or () if (leg or {}).get("mode") == "transit"]
    if tuple(service_meta(leg)["key"] for leg in left) != tuple(
            service_meta(leg)["key"] for leg in right):
        return False
    if branch_key_for_transit_legs(left) != branch_key_for_transit_legs(right):
        return False
    if not left:
        return True
    lp = leg_boarding_profile(left[0]); rp = leg_boarding_profile(right[0])
    if lp is None or rp is None:
        return direction_guard(left[0]) == direction_guard(right[0])
    dot = lp["heading"][0] * rp["heading"][0] + lp["heading"][1] * rp["heading"][1]
    return dot >= math.cos(math.radians(100.0))


def alt_choice_equivalent(
    left: Alternative,
    right: Alternative,
    *,
    slot_legs: Callable[[Alternative, str], list[Leg]] = alt_slot_legs,
    service_meta: Callable[[Leg], dict[str, str]] = leg_service_meta,
    direction_guard: Callable[[Leg], str] = leg_boarding_direction_guard,
) -> bool:
    return all(journey_choice_equivalent(slot_legs(left, slot), slot_legs(right, slot),
                                          service_meta=service_meta,
                                          direction_guard=direction_guard)
               for slot in ("best", "typical"))


def alt_dedupe_key(
    a: Alternative,
    family_keys=None,
    *,
    choice_key: Callable[[Alternative], Any] = alt_choice_key,
) -> Any:
    return choice_key(a)
