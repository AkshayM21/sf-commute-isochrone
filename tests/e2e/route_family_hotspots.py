"""Deterministic route-family hotspot discovery and route-inspector assertions.

This module deliberately talks only to an already-running featured server.  It does not boot the
application, geocode addresses, or import server internals.  The fixed public coordinates make a
scan repeatable for a given service date; the seed controls deterministic destination/cell ties.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from urllib.parse import quote


PUBLIC_DESTINATIONS = (
    {"slug": "market", "label": "1 Market St", "lat": 37.79360, "lon": -122.39580},
    {"slug": "townsend", "label": "650 Townsend St", "lat": 37.7714154, "lon": -122.4030885},
    {"slug": "mission_bay", "label": "UCSF Mission Bay", "lat": 37.76670, "lon": -122.38920},
    {"slug": "city_hall", "label": "1 Dr Carlton B Goodlett Pl", "lat": 37.77930, "lon": -122.41920},
    {"slug": "parnassus", "label": "505 Parnassus Ave", "lat": 37.76310, "lon": -122.45860},
    {"slug": "stonestown", "label": "3251 20th Ave", "lat": 37.72830, "lon": -122.47600},
    {"slug": "outer_sunset", "label": "Judah St and 46th Ave", "lat": 37.76020, "lon": -122.50630},
    {"slug": "pier_70", "label": "Pier 70", "lat": 37.75930, "lon": -122.38750},
)

# Public coordinates already covered by the live API regression in tests/test_api.py.  This case
# has historically exposed both a walk-finish and a transit-tail branch, making it useful even when
# the opt-in citywide scan is disabled.
SAVED_HOTSPOT = {
    "slug": "mission_to_townsend",
    "label": "Mission Dolores to 650 Townsend",
    "destination": PUBLIC_DESTINATIONS[1],
    "origin": {"lat": 37.76640, "lon": -122.42670},
    "cell_id": "1916",
    "speed": "med",
}

DEFAULT_SEED = int(os.environ.get("ROUTE_FAMILY_HOTSPOT_SEED", "20260712"))
ARTIFACT_PATH = Path(os.environ.get(
    "ROUTE_FAMILY_HOTSPOT_ARTIFACT",
    str(Path(__file__).parent / "screens" / "route_family_hotspots.json"),
))

# API-observable dimensions from AGENT_UI_TESTING_PLAYBOOK's complexity rubric.  Scroll debt and
# viewport coverage are intentionally absent: they only exist after browser rendering and are
# measured by measure_family_card instead.  Keep this tuple public in the artifact so a later agent
# can tell exactly which trade-offs made a case non-dominated.
PARETO_DIMENSIONS = (
    "routes",
    "families",
    "branches",
    "max_branches_per_family",
    "max_options_per_family",
    "max_transfers",
    "unique_lines",
    "unique_services",
    "label_chars",
    "longest_label",
    "time_spread",
    "geometry_points",
    "service_branch_cross_product",
)


def _stable_rank(seed, *parts):
    """Stable seeded ordering for test sampling, not a cryptographic digest.

    The original result string is a collision-safe secondary ordering key.  The primary is a
    small 32-bit polynomial mixer so this remains deterministic across Python processes without
    carrying a production-style integrity primitive into a test-only sampler.
    """
    raw = "\x1f".join([str(seed), *(str(part) for part in parts)])
    state = int(seed) & 0xFFFFFFFF
    for byte in raw.encode("utf-8"):
        state = ((state * 1_103_515_245) + byte + 12_345) & 0xFFFFFFFF
    return state, raw


def _api_get(api, base_url, path, *, attempts=2, timeout=45_000):
    """GET JSON sequentially, honoring the server's Busy/rate-limit retry contract."""
    url = base_url.rstrip("/") + path
    response = None
    for attempt in range(attempts):
        response = api.get(url, timeout=timeout)
        if response.status not in (429, 503) or attempt + 1 >= attempts:
            break
        try:
            wait_s = float(response.headers.get("retry-after", "4"))
        except (TypeError, ValueError):
            wait_s = 4.0
        time.sleep(max(0.05, min(wait_s, 8.0)))
    assert response is not None and response.ok, (
        f"GET {path} failed: status={getattr(response, 'status', None)} "
        f"body={response.text()[:500] if response is not None else '<none>'}"
    )
    return response.json()


def _speed_query(speed):
    return "" if speed == "med" else f"&speed={quote(speed)}"


def _slot_legs(route, slot):
    nested = route.get(slot) or {}
    return nested.get("geom") or nested.get("legs") or route.get("geom") or route.get("legs") or []


def _route_slots(route):
    best = _slot_legs(route, "best")
    typical = _slot_legs(route, "typical")
    return best or typical, typical or best


def _transit_legs(legs):
    return [leg for leg in legs or () if (leg or {}).get("mode") == "transit"]


def _line_name(leg):
    return str((leg or {}).get("name") or (leg or {}).get("line") or "")


def _points(legs):
    out = []
    for leg in legs or ():
        for point in (leg or {}).get("pts") or ():
            try:
                out.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError, IndexError):
                continue
    return out


def _heading_bucket(points, *, arrival=False):
    if len(points) < 2:
        return None
    pairs = zip(reversed(points[1:]), reversed(points[:-1])) if arrival else zip(points, points[1:])
    for start, end in pairs:
        if arrival:
            start, end = end, start
        if abs(end[0] - start[0]) + abs(end[1] - start[1]) <= 1e-7:
            continue
        north = end[0] - start[0]
        east = (end[1] - start[1]) * math.cos(math.radians((start[0] + end[0]) / 2.0))
        angle = math.atan2(east, north) % (2.0 * math.pi)
        return int(round(angle / (2.0 * math.pi / 16.0))) % 16
    return None


def _journey_choice_signature(legs):
    transit = _transit_legs(legs)
    if not transit:
        return ((), None, None, None)
    sequence = tuple((str(leg.get("feed") or ""), str(leg.get("tmode") or ""),
                      _line_name(leg)) for leg in transit)
    first_points = _points([transit[0]])
    last_points = _points([transit[-1]])
    endpoint = (round(last_points[-1][0], 4), round(last_points[-1][1], 4)) if last_points else None
    # Initial direction protects loop/opposite-direction choices while intentionally ignoring the
    # exact boarding coordinate, so same-direction access variants remain duplicate candidates.
    return (sequence, _heading_bucket(first_points), endpoint,
            _heading_bucket(last_points, arrival=True))


def _option_choice_signature(route):
    best, typical = _route_slots(route)
    return (_journey_choice_signature(best), _journey_choice_signature(typical))


def _haversine_m(a, b):
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6_371_000.0 * 2.0 * math.asin(min(1.0, math.sqrt(h)))


def _walk_private_keys(value, found=None):
    found = found if found is not None else []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"_family_seed", "_primary_family_seed", "_family_catalog", "_branch"}:
                found.append(key)
            _walk_private_keys(child, found)
    elif isinstance(value, list):
        for child in value:
            _walk_private_keys(child, found)
    return found


def _active_total(route, slot="typical"):
    nested = route.get(slot) or {}
    value = nested.get("total")
    if value is None:
        value = route.get("min", route.get("total"))
    return int(value) if value is not None else None


def validate_route_response(body, destination):
    """Return concrete API/label false-advertising errors for one pinned response."""
    errors = []
    if body.get("error"):
        return [f"route error: {body.get('error')}"]
    options = [body, *(body.get("alts") or [])]
    if private := _walk_private_keys(body):
        errors.append(f"private selection keys leaked: {sorted(set(private))}")

    families = {}
    branches = {}
    choices = {}
    public_choice_keys = set()
    origin = (body.get("olat"), body.get("olon"))
    dest = (destination["lat"], destination["lon"])
    primary_total = _active_total(body, "typical")
    for index, route in enumerate(options):
        prefix = "primary" if index == 0 else f"alt[{index - 1}]"
        public_choice_key = str(route.get("choice_key") or "")
        if not public_choice_key:
            errors.append(f"{prefix}: missing durable choice_key")
        elif public_choice_key in public_choice_keys:
            errors.append(f"{prefix}: duplicate choice_key {public_choice_key!r}")
        else:
            public_choice_keys.add(public_choice_key)
        family = route.get("family") or {}
        branch = route.get("branch") or {}
        fkey, bkey = family.get("key"), branch.get("key")
        if not fkey or not bkey:
            errors.append(f"{prefix}: incomplete family/branch metadata")
            continue
        if branch.get("kind") not in ("walk", "transit"):
            errors.append(f"{prefix}: invalid branch kind {branch.get('kind')!r}")
        families.setdefault(str(fkey), {"meta": family, "members": []})["members"].append(route)
        branches.setdefault((str(fkey), str(bkey)), {"meta": branch, "members": []})["members"].append(route)

        signature = _option_choice_signature(route)
        if signature in choices:
            errors.append(f"{prefix}: same two-slot service/terminal choice as {choices[signature]}")
        else:
            choices[signature] = prefix

        best, typical = _route_slots(route)
        if index > 0 and not (_transit_legs(best) or _transit_legs(typical)):
            errors.append(f"{prefix}: advertised alternative has no transit ride")
        for slot, legs in (("best", best), ("typical", typical)):
            if not legs:
                errors.append(f"{prefix}.{slot}: no drawable legs")
                continue
            points = _points(legs)
            if not points:
                errors.append(f"{prefix}.{slot}: no geometry points")
                continue
            if all(value is not None for value in origin) and _haversine_m(points[0], origin) > 300:
                errors.append(f"{prefix}.{slot}: starts more than 300m from origin")
            if _haversine_m(points[-1], dest) > 300:
                errors.append(f"{prefix}.{slot}: ends more than 300m from destination")
            total = _active_total(route, slot)
            leg_total = sum(int(leg.get("min") or 0) + int(leg.get("wait") or 0) for leg in legs)
            if total is not None and leg_total != total:
                errors.append(f"{prefix}.{slot}: legs total {leg_total} != {total}")
            for leg_index, leg in enumerate(legs):
                if (leg or {}).get("mode") != "walk" or "physical_min" not in (leg or {}):
                    continue
                try:
                    physical = float(leg.get("physical_min") or 0)
                    allowance = float(leg.get("schedule_allowance_min") or 0)
                    visible = float(leg.get("min") or 0)
                except (TypeError, ValueError):
                    errors.append(f"{prefix}.{slot}.leg[{leg_index}]: invalid walk truth fields")
                    continue
                # ``leg.min`` is an integer allocated by whole-itinerary reconciliation so every
                # displayed leg sums to the integer headline. Physical/allowance fields preserve
                # exact seconds and may therefore differ from that allocated integer by up to one
                # minute; they must never be negative or drift farther than one rounding unit.
                if (physical < 0 or allowance < 0
                        or abs((physical + allowance) - visible) > 1.000001):
                    errors.append(
                        f"{prefix}.{slot}.leg[{leg_index}]: physical {physical} + allowance "
                        f"{allowance} != visible {visible}")

        route_total = _active_total(route, "typical")
        if index > 0 and primary_total is not None and route_total is not None:
            if route_total < primary_total:
                errors.append(
                    f"{prefix}: route total {route_total} is faster than primary {primary_total}")

    # A pinned response deliberately separates the route that first painted the map from the
    # route the server recommends for decision-making.  Both are public, durable keys and may be
    # different; neither may point outside the advertised choice set.
    map_choice_key = str(body.get("map_choice_key") or body.get("choice_key") or "")
    recommended_choice_key = str(body.get("recommended_choice_key") or "")
    if not map_choice_key:
        errors.append("pinned response: missing map_choice_key")
    elif map_choice_key not in public_choice_keys:
        errors.append(f"pinned response: map_choice_key {map_choice_key!r} is not a public choice")
    # Unpinned hover responses have no recommendation role.  Pinned responses must make it
    # explicit so clients never infer it from option order.
    if "recommended_choice_key" in body and not recommended_choice_key:
        errors.append("pinned response: empty recommended_choice_key")
    elif recommended_choice_key and recommended_choice_key not in public_choice_keys:
        errors.append(
            f"pinned response: recommended_choice_key {recommended_choice_key!r} is not a public choice")

    primary_family = str((body.get("family") or {}).get("key") or "")
    primary_marked = [key for key, value in families.items()
                      if value["meta"].get("sub") == "primary boarding corridor"]
    if primary_marked != [primary_family]:
        errors.append(f"primary family advertising mismatch: {primary_marked} vs {primary_family!r}")

    for fkey, value in families.items():
        meta, members = value["meta"], value["members"]
        visible_services = {
            (_line_name(transit[0]), str(transit[0].get("feed") or ""),
             str(transit[0].get("tmode") or ""))
            for route in members if (transit := _transit_legs(_route_slots(route)[1]))
        }
        service_rows = meta.get("services") or []
        service_by_key = {}
        for service in service_rows:
            key = str((service or {}).get("key") or "")
            name = str((service or {}).get("name") or "")
            if not key or not name:
                errors.append(f"family {fkey}: service row lacks opaque key/name")
                continue
            if key in service_by_key:
                errors.append(f"family {fkey}: duplicate service key {key}")
            service_by_key[key] = service
            if not isinstance(service.get("shown"), bool):
                errors.append(f"family {fkey}: service {key} has no boolean shown flag")
            if not isinstance(service.get("branchKeys"), list):
                errors.append(f"family {fkey}: service {key} has no branchKeys list")
        catalog_services = {
            (str(service.get("name") or ""), str(service.get("feed") or ""),
             str(service.get("mode") or ""))
            for service in service_rows
        }
        missing_visible = visible_services - catalog_services
        if missing_visible:
            errors.append(f"family {fkey}: visible first services absent from catalog: {missing_visible}")
        for service in service_rows:
            identity = (str(service.get("name") or ""), str(service.get("feed") or ""),
                        str(service.get("mode") or ""))
            if bool(service.get("shown")) != (identity in visible_services):
                errors.append(f"family {fkey}: service {service.get('key')} has false shown state")
        advertised = set(str(line) for line in (meta.get("lines") or []))
        catalog_names = {str(service.get("name") or "") for service in service_rows}
        if advertised != catalog_names:
            errors.append(f"family {fkey}: lines are inconsistent with its service catalog")
        tags = set(str(tag) for tag in (meta.get("tags") or []))
        branch_count = sum(1 for family_key, _branch_key in branches if family_key == fkey)
        if ("shared corridor" in tags) != (len(service_rows) > 1):
            errors.append(f"family {fkey}: shared-corridor tag is not supported by its services")
        if ("multiple finishes" in tags) != (branch_count > 1):
            errors.append(f"family {fkey}: multiple-finishes tag is not supported by its branches")

        catalog_branch_keys = {branch_key for family_key, branch_key in branches if family_key == fkey}
        for service_key, service in service_by_key.items():
            row_branch_keys = set(str(key) for key in (service.get("branchKeys") or []))
            if not row_branch_keys.issubset(catalog_branch_keys):
                errors.append(f"family {fkey}: service {service_key} names an unknown branch")
            expected = {
                branch_key for (family_key, branch_key), branch_value in branches.items()
                if family_key == fkey and service_key in set(
                    str(key) for key in (branch_value["meta"].get("serviceKeys") or []))
            }
            if row_branch_keys != expected:
                errors.append(f"family {fkey}: service {service_key} branchKeys disagree with branches")

    for (fkey, bkey), value in branches.items():
        meta, members = value["meta"], value["members"]
        family_services = {
            str(service.get("key")): service
            for service in (families[fkey]["meta"].get("services") or [])
        }
        branch_services = meta.get("services") or []
        branch_service_keys = [str(key) for key in (meta.get("serviceKeys") or [])]
        row_keys = [str((service or {}).get("key") or "") for service in branch_services]
        if branch_service_keys != row_keys:
            errors.append(f"branch {fkey}/{bkey}: serviceKeys do not match service rows")
        if not set(branch_service_keys).issubset(family_services):
            errors.append(f"branch {fkey}/{bkey}: service catalog is not a family subset")
        for service in branch_services:
            key = str((service or {}).get("key") or "")
            family_service = family_services.get(key) or {}
            for field in ("name", "feed", "mode"):
                if str(service.get(field) or "") != str(family_service.get(field) or ""):
                    errors.append(
                        f"branch {fkey}/{bkey}: service {key} disagrees on {field}")
        tail_lines = set()
        transit_counts = []
        for route in members:
            transit = _transit_legs(_route_slots(route)[1])
            transit_counts.append(len(transit))
            for leg in transit[1:]:
                line_name = _line_name(leg)
                if line_name:
                    tail_lines.add(line_name)
        advertised = set(str(line) for line in (meta.get("lines") or []))
        # The server may expose a proven pre-cap tail service that has no separate visible option;
        # public-response validation can prove that every visible tail is advertised, while the
        # symbolic server suite proves that extra catalog tails came from hydrated itineraries.
        if not tail_lines.issubset(advertised):
            errors.append(
                f"branch {fkey}/{bkey}: visible tails {sorted(tail_lines)} are absent from "
                f"advertised lines {sorted(advertised)}")
        if meta.get("kind") == "walk":
            allowed_counts = {0} if bkey == "walk:only" else {1}
            if any(count not in allowed_counts for count in transit_counts):
                errors.append(
                    f"branch {fkey}/{bkey}: walk branch has incompatible transit-leg count")
        if meta.get("kind") == "transit" and any(count < 2 for count in transit_counts):
            errors.append(f"branch {fkey}/{bkey}: transit-finish branch contains a one-seat route")
    return errors


def route_metrics(body):
    options = [body, *(body.get("alts") or [])]
    families = {}
    branches = {}
    unique_lines = set()
    unique_services = set()
    max_transfers = 0
    geometry_points = 0
    totals = []
    for route in options:
        family, branch = route.get("family") or {}, route.get("branch") or {}
        fkey, bkey = str(family.get("key") or ""), str(branch.get("key") or "")
        families.setdefault(fkey, family)
        branches.setdefault((fkey, bkey), branch)
        _best, typical = _route_slots(route)
        transit = _transit_legs(typical)
        unique_lines.update(_line_name(leg) for leg in transit if _line_name(leg))
        unique_services.update(
            (_line_name(leg), str(leg.get("feed") or ""), str(leg.get("tmode") or ""))
            for leg in transit if _line_name(leg)
        )
        max_transfers = max(max_transfers, max(0, len(transit) - 1))
        geometry_points += len(_points(typical))
        total = _active_total(route, "typical")
        if total is not None:
            totals.append(total)
    branches_per_family = {key: 0 for key in families}
    options_per_family = {key: 0 for key in families}
    for fkey, _bkey in branches:
        branches_per_family[fkey] = branches_per_family.get(fkey, 0) + 1
    for route in options:
        fkey = str((route.get("family") or {}).get("key") or "")
        options_per_family[fkey] = options_per_family.get(fkey, 0) + 1
    labels = [str(meta.get(key) or "") for meta in families.values() for key in ("name", "sub")]
    labels += [str(tag) for meta in families.values() for tag in (meta.get("tags") or [])]
    labels += [str(meta.get("name") or "") for meta in branches.values()]
    label_chars = sum(len(label) for label in labels)
    unique_lines.update(str(line) for meta in families.values() for line in (meta.get("lines") or []))
    service_branch_cross_product = 0
    for fkey, family in families.items():
        unique_services.update(
            (str((service or {}).get("name") or ""), str((service or {}).get("feed") or ""),
             str((service or {}).get("mode") or ""))
            for service in (family.get("services") or [])
            if str((service or {}).get("name") or "")
        )
        services = {
            str((service or {}).get("key") or (service or {}).get("name") or "")
            for service in (family.get("services") or [])
        }
        services.discard("")
        # Older fixture responses may expose only family.lines.  They still have a meaningful
        # service/branch product, so use those names as the compatibility catalog.
        if not services:
            services = {str(line) for line in (family.get("lines") or []) if str(line)}
        service_branch_cross_product += len(services) * branches_per_family.get(fkey, 0)
    route_count, family_count, branch_count = len(options), len(families), len(branches)
    # Scalar order is deliberately secondary to Pareto retention.  It rewards the dimensions that
    # most directly expand the card; geometry/time-spread/longest-label extremes remain protected
    # by the frontier even though they are not folded into this score.
    score = (4 * max(0, route_count - 1) + 5 * max(0, family_count - 1)
             + 4 * max(0, branch_count - family_count) + 2 * max_transfers
             + len(unique_lines) + math.ceil(label_chars / 24.0))
    return {
        "routes": route_count,
        "families": family_count,
        "branches": branch_count,
        "max_branches_per_family": max(branches_per_family.values(), default=0),
        "max_options_per_family": max(options_per_family.values(), default=0),
        "max_transfers": max_transfers,
        "unique_lines": len(unique_lines),
        "unique_services": len(unique_services),
        "label_chars": label_chars,
        "longest_label": max((len(label) for label in labels), default=0),
        "geometry_points": geometry_points,
        "time_spread": max(totals) - min(totals) if totals else 0,
        "service_branch_cross_product": service_branch_cross_product,
        "score": score,
    }


def _complexity_vector(row):
    metrics = row.get("metrics") or {}
    vector = []
    for dimension in PARETO_DIMENSIONS:
        try:
            vector.append(max(0, int(metrics.get(dimension) or 0)))
        except (TypeError, ValueError):
            vector.append(0)
    return tuple(vector)


def _dominates(left, right):
    """Return whether left is no simpler anywhere and strictly harder somewhere."""
    return all(a >= b for a, b in zip(left, right)) and any(
        a > b for a, b in zip(left, right))


def _hotspot_scalar_key(row, seed, stratum):
    """Documented scalar fallback with a stable, seed-controlled final tie-break."""
    metrics = row.get("metrics") or {}
    destination = row.get("destination") or {}
    identity = (destination.get("slug") or "", row.get("speed") or "", row.get("cell_id") or "")
    try:
        score = int(metrics.get("score", -1))
    except (TypeError, ValueError):
        score = -1
    return (-score, _stable_rank(seed, "pareto", stratum, *identity))


def _rank_hotspots(rows, *, seed, limit=None):
    """Return a bounded, deterministic frontier-first hotspot ranking.

    Every first-frontier case is retained ahead of every dominated case.  The scalar score only
    orders peers within the frontier and fills any remaining bounded slots.  Equal vectors do not
    dominate one another, matching the strict Pareto definition in the testing playbook.
    """
    entries = [(index, row, _complexity_vector(row)) for index, row in enumerate(rows)]
    frontier = {
        index for index, _row, vector in entries
        if not any(other_index != index and _dominates(other_vector, vector)
                   for other_index, _other_row, other_vector in entries)
    }
    ordered = sorted(
        (entry for entry in entries if entry[0] in frontier),
        key=lambda entry: _hotspot_scalar_key(entry[1], seed, "frontier"),
    )
    ordered += sorted(
        (entry for entry in entries if entry[0] not in frontier),
        key=lambda entry: _hotspot_scalar_key(entry[1], seed, "fill"),
    )
    if limit is not None:
        ordered = ordered[:max(0, int(limit))]
    ranked = []
    for _index, row, vector in ordered:
        annotated = dict(row)
        annotated["pareto_frontier"] = _index in frontier
        annotated["pareto_vector"] = dict(zip(PARETO_DIMENSIONS, vector))
        ranked.append(annotated)
    return ranked


def _candidate_ids(cells, variance, *, seed, config_key, limit):
    reachable = [(str(cid), int(value[1])) for cid, value in cells.items()
                 if isinstance(value, list) and len(value) > 1 and value[1] is not None]
    if not reachable:
        return []
    by_id = dict(reachable)
    selected = []

    def add(cid):
        cid = str(cid)
        if cid in by_id and cid not in selected and len(selected) < limit:
            selected.append(cid)

    def alt_count(cid):
        alt = (variance.get(str(cid)) or {}).get("alt") or []
        return len(alt)

    ranked_alts = sorted(reachable, key=lambda item: (
        -alt_count(item[0]), _stable_rank(seed, config_key, "alt", item[0])))
    for cid, _minutes in ranked_alts[:2]:
        add(cid)
    ranked_frag = sorted(reachable, key=lambda item: (
        -int((variance.get(item[0]) or {}).get("frag") or 0),
        _stable_rank(seed, config_key, "frag", item[0])))
    if ranked_frag:
        add(ranked_frag[0][0])

    # Seeded travel-time strata find deterministic branch-rich cells that have no variance chips.
    for band_name, predicate in (
        ("near", lambda minutes: minutes <= 30),
        ("mid", lambda minutes: 30 < minutes <= 50),
        ("far", lambda minutes: minutes > 50),
    ):
        pool = sorted((cid for cid, minutes in reachable if predicate(minutes)),
                      key=lambda cid: _stable_rank(seed, config_key, band_name, cid))
        if pool:
            add(pool[0])
    for cid, _minutes in sorted(reachable,
                                key=lambda item: _stable_rank(seed, config_key, "fill", item[0])):
        add(cid)
    return selected


def scan_hotspots(api, base_url, *, destinations, speeds, seed=DEFAULT_SEED, per_config=5):
    """Run compute -> variance -> pinned itineraries and return a frontier-first artifact."""
    health = _api_get(api, base_url, "/healthz", attempts=1, timeout=10_000)
    assert health.get("engine") == "raptor", f"hotspot scan requires RAPTOR, got {health}"
    rows = []
    configs = []
    for destination in destinations:
        for speed in speeds:
            speed_q = _speed_query(speed)
            dlat, dlon = destination["lat"], destination["lon"]
            compute = _api_get(api, base_url, f"/compute?lat={dlat}&lon={dlon}{speed_q}")
            variance_body = _api_get(api, base_url, f"/variance?dlat={dlat}&dlon={dlon}{speed_q}")
            variance = variance_body.get("variance") or {}
            config_key = f"{destination['slug']}:{speed}"
            candidate_ids = _candidate_ids(compute.get("cells") or {}, variance, seed=seed,
                                           config_key=config_key, limit=per_config)
            configs.append({"destination": destination["slug"], "speed": speed,
                            "candidate_ids": candidate_ids})
            for cid in candidate_ids:
                body = _api_get(
                    api, base_url,
                    f"/itinerary?id={quote(cid)}&dlat={dlat}&dlon={dlon}{speed_q}&pin=1",
                )
                errors = validate_route_response(body, destination)
                row = {
                    "destination": destination,
                    "speed": speed,
                    "cell_id": cid,
                    "origin": {"lat": body.get("olat"), "lon": body.get("olon")},
                    "metrics": route_metrics(body) if not body.get("error") else {},
                    "errors": errors,
                }
                rows.append(row)
    # The request budget has already bounded this universe to per_config cases per config.  Keep the
    # explicit cap here as a guard against a future candidate sampler accidentally widening it.
    max_cases = len(destinations) * len(speeds) * per_config
    rows = _rank_hotspots(rows, seed=seed, limit=max_cases)
    return {
        "seed": seed,
        "health": health,
        "configs": configs,
        "pareto_dimensions": list(PARETO_DIMENSIONS),
        "pareto_frontier_count": sum(bool(row["pareto_frontier"]) for row in rows),
        "hotspots": rows,
    }


def write_artifact(artifact, path=ARTIFACT_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return path


def open_destination(page, base_url, destination, speed):
    """Boot the app at exact public coordinates via its permalink, with no geocoder dependency."""
    base = base_url.rstrip("/")
    # Clear persistence before the app's boot IIFE runs. Navigating to `/` first and then changing
    # only the hash is a same-document navigation; there is intentionally no hashchange listener,
    # so that pattern never re-runs applyHash and leaves the map blank.
    page.add_init_script("() => { try { localStorage.clear(); } catch (e) {} }")
    label = quote(destination["label"], safe="")
    url = (f"{base}/#wp={destination['lat']:.7f},{destination['lon']:.7f},{label}"
           f"&metric=r&cmode=time&colors=on&mt=any&sp={speed}&th=auto")
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_function("() => document.querySelectorAll('#list .nb').length > 0", timeout=30_000)
    page.wait_for_function(
        "() => (typeof VAR !== 'undefined' && Object.keys(VAR).length > 0) "
        "|| (typeof REAL !== 'undefined' && Object.keys(REAL).length > 0) "
        "|| (typeof varianceSettled === 'function' && varianceSettled())",
        timeout=45_000,
    )


def origin_container_point(page, origin):
    page.evaluate("([lat,lon]) => { map.setView([lat,lon], 13); }",
                  [origin["lat"], origin["lon"]])
    page.wait_for_timeout(450)
    return page.evaluate(
        "([lat,lon]) => { const p=map.latLngToContainerPoint(L.latLng(lat,lon));"
        " const r=document.getElementById('map').getBoundingClientRect();"
        " return {x:r.left+p.x,y:r.top+p.y}; }",
        [origin["lat"], origin["lon"]],
    )


def measure_family_card(page):
    """Return objective crowding, reachability, and collision metrics for the inspector.

    The inspector has three layouts and Plan can be reparented into the desktop sidecar.  Keep
    this browser audit tied to the visible *active pane*, rather than assuming Choices is always
    the scroll host or that a globally mounted Plan CTA exists.  This is deliberately a metric
    collector: callers can record a hotspot before an individual viewport has enough content to
    exercise every failure mode.
    """
    return page.evaluate(
        """() => {
          const card=document.getElementById('pincard'), body=document.getElementById('pinbody');
          const choices=document.getElementById('route-choices-panel'),plan=document.getElementById('route-plan-panel');
          const panel=document.getElementById('panel');
          const legend=document.getElementById('legend');
          const rect=el=>{const r=el.getBoundingClientRect();return {l:r.left,t:r.top,r:r.right,b:r.bottom,w:r.width,h:r.height};};
          const overlap=(a,b)=>Math.max(0,Math.min(a.r,b.r)-Math.max(a.l,b.l))*Math.max(0,Math.min(a.b,b.b)-Math.max(a.t,b.t));
          const visible=el=>{const s=getComputedStyle(el),r=el.getBoundingClientRect();return s.display!=='none'&&s.visibility!=='hidden'&&r.width>0&&r.height>0;};
          const visibleIn=el=>visible(el)&&el.getBoundingClientRect().bottom>0&&el.getBoundingClientRect().top<innerHeight;
          const scrollable=el=>{if(!el||!visible(el))return false;const s=getComputedStyle(el);return /auto|scroll/.test(s.overflowY)&&el.scrollHeight>el.clientHeight+1;};
          const activePlan=[...document.querySelectorAll('#route-sidecar .route-plan-pane,#pincard .route-plan-pane')]
            .find(visible)||null;
          const activeChoices=[...document.querySelectorAll('#pincard .route-choices-pane')].find(visible)||null;
          const activePane=activePlan||activeChoices||null;
          const scrollHost=scrollable(activePane)?activePane:
            [activePane,body,card].find(scrollable)||activePane||body;
          const paneKind=activePlan?'plan':activeChoices?'choices':'none';
          // Destination/settings use intentional one-line ellipsis. Audit route decision copy,
          // which must remain readable rather than treating that designed truncation as clipping.
          const inspectorRoots=[card,document.getElementById('route-sidecar')].filter(Boolean);
          const labels=inspectorRoots.flatMap(root=>[...root.querySelectorAll('.pin-place,.route-name,.route-tradeoffs,.route-action,.route-time,.route-fact-value,.boarding-heading span,.plan-title,.route-copy,.route-detail')]);
          const clipped=labels.filter(el=>{const s=getComputedStyle(el);return visible(el)&&((['hidden','clip'].includes(s.overflowX)&&el.scrollWidth>el.clientWidth+1)||(['hidden','clip'].includes(s.overflowY)&&el.scrollHeight>el.clientHeight+1));}).map(el=>el.textContent.trim());
          const targets=inspectorRoots.flatMap(root=>[...root.querySelectorAll('button,a[href],summary,[role="button"]')]).filter(visible);
          const sizes=targets.map(el=>{const r=el.getBoundingClientRect();return Math.min(r.width,r.height);});
          const interactives=[...((activePane||card).querySelectorAll('.route-choice,[data-route-plan-for],.route-directions li,.plan-google'))].filter(visible);
          const oldTop=scrollHost&&'scrollTop'in scrollHost?scrollHost.scrollTop:0;
          if(scrollHost&&'scrollTop'in scrollHost)scrollHost.scrollTop=scrollHost.scrollHeight;
          const br=scrollHost&&scrollHost.getBoundingClientRect?scrollHost.getBoundingClientRect():null;
          const last=interactives.length?interactives[interactives.length-1].getBoundingClientRect():null;
          // Fixed Plan footers remain part of the pane.  The last meaningful action/content must
          // be entirely inside the active scroll region after a max-scroll, not merely exist below
          // a transformed mobile sheet.
          const reachesLast=!last||!br||(last.bottom<=br.bottom+1&&last.top>=br.top-1&&last.bottom<=innerHeight+1);
          scrollHost.scrollTop=oldTop;
          const cr=rect(card),pr=rect(panel),lr=rect(legend);
          const interactiveSelector='button,a[href],summary,[role="button"]';
          const nested=inspectorRoots.flatMap(root=>[...root.querySelectorAll(interactiveSelector)]).filter(el=>{
            const parent=el.parentElement&&el.parentElement.closest(interactiveSelector);
            return parent&&inspectorRoots.some(root=>root.contains(parent));
          });
          const selectedKeys=[...new Set([...card.querySelectorAll('.route-choice[aria-pressed="true"]')]
            .map(el=>el.dataset.choiceKey||el.dataset.key).filter(Boolean))];
          const routePlanActions=[...card.querySelectorAll('[data-route-plan-for]')];
          const inspectorCloseControls=inspectorRoots.flatMap(root=>[...root.querySelectorAll('[data-close-pin],#pinx')])
            .filter((el,index,all)=>visible(el)&&all.indexOf(el)===index);
          const planCloseControls=inspectorRoots.flatMap(root=>[...root.querySelectorAll('[data-route-plan-close]')])
            .filter((el,index,all)=>visible(el)&&all.indexOf(el)===index);
          const closeControls=inspectorRoots.flatMap(root=>[...root.querySelectorAll('[data-close-pin],#pinx,[data-route-plan-close],button[aria-label*="Close" i]')])
            .filter((el,index,all)=>visible(el)&&all.indexOf(el)===index);
          const closeRects=closeControls.map(el=>({label:(el.getAttribute('aria-label')||el.textContent||'').trim(),rect:rect(el)}));
          const closeCollisions=[];
          for(let i=0;i<closeRects.length;i++)for(let j=i+1;j<closeRects.length;j++){
            const area=overlap(closeRects[i].rect,closeRects[j].rect);
            if(area>1)closeCollisions.push({left:closeRects[i].label,right:closeRects[j].label,area});
          }
          const offscreenRight=[...document.body.querySelectorAll('*')].map(el=>{
            const r=el.getBoundingClientRect(),s=getComputedStyle(el);
            return {tag:el.tagName.toLowerCase(),id:el.id||'',cls:String(el.className||'').slice(0,80),
                    left:Math.round(r.left),right:Math.round(r.right),width:Math.round(r.width),
                    position:s.position,transform:s.transform};
          }).filter(x=>x.width>0&&x.right>innerWidth+1).sort((a,b)=>b.right-a.right).slice(0,12);
          const isSheet=card.dataset.layoutCapability==='bottom-sheet';
          const visibleSheetHeight=Math.max(0,Math.min(cr.b,innerHeight)-Math.max(cr.t,0));
          const cssSheetHeight=parseFloat(getComputedStyle(card).getPropertyValue('--sheet-height'));
          return {
            viewport:{w:innerWidth,h:innerHeight},card:cr,
            families:new Set([...card.querySelectorAll('.route-choice')].map(row=>row.dataset.family)).size,
            branches:new Set([...card.querySelectorAll('.route-choice')]
              .map(row=>`${row.dataset.family}|${row.dataset.branch}`)).size,
            route_options:card.querySelectorAll('.route-choice').length,
            route_choice_rows:card.querySelectorAll('.route-choice').length,
            recommended_rows:card.querySelectorAll('.route-recommendations .route-choice.recommended').length,
            selected_keys:selectedKeys,
            nested_interactives:nested.length,
            active_pane:paneKind,
            active_scroll_host:scrollHost===activePlan?'plan':scrollHost===activeChoices?'choices':scrollHost===body?'body':scrollHost===card?'card':'none',
            scroll_debt:scrollHost?Math.max(0,scrollHost.scrollHeight-scrollHost.clientHeight):0,
            horizontal_overflow:scrollHost?Math.max(0,scrollHost.scrollWidth-scrollHost.clientWidth):0,
            document_horizontal_overflow:Math.max(0,document.documentElement.scrollWidth-innerWidth),
            offscreen_right:offscreenRight,
            clipped_labels:clipped,
            scroll_reaches_last:reachesLast,
            route_plan_actions:routePlanActions.length,
            route_plan_keys:routePlanActions.map(el=>el.dataset.routePlanFor||''),
            displayed_route_plan_keys:routePlanActions.filter(visible).map(el=>el.dataset.routePlanFor||''),
            visible_route_plan_keys:routePlanActions.filter(visibleIn).map(el=>el.dataset.routePlanFor||''),
            close_visible_count:closeControls.length,
            inspector_close_visible_count:inspectorCloseControls.length,
            plan_close_visible_count:planCloseControls.length,
            close_controls:closeRects.map(item=>item.label),
            close_collisions:closeCollisions,
            sheet:{
              is_bottom_sheet:isSheet,
              css_height:Number.isFinite(cssSheetHeight)?cssSheetHeight:null,
              actual_height:cr.h,
              visible_height:visibleSheetHeight,
              height_delta:Number.isFinite(cssSheetHeight)?Math.abs(cr.h-cssSheetHeight):null,
              hidden_height:Math.max(0,cr.h-visibleSheetHeight),
              bottom_delta:Math.abs(cr.b-innerHeight),
            },
            min_target_px:sizes.length?Math.min(...sizes):0,
            label_chars:labels.reduce((n,el)=>n+(el.textContent||'').trim().length,0),
            panel_overlap_px2:visible(panel)?overlap(cr,pr):0,
            legend_overlap_px2:visible(legend)?overlap(cr,lr):0,
            close_visible:closeControls.some(el=>el.id==='pinx'&&visible(el)),
            card_in_view:cr.l>=-1&&cr.t>=-1&&cr.r<=innerWidth+1&&cr.b<=innerHeight+1,
            choices_visible:visible(choices),plan_visible:visible(plan),
          };
        }"""
    )


def assert_inspector_health(metrics, *, tolerance_px=1.5):
    """Assert state-independent visual health from :func:`measure_family_card`.

    Hotspot callers use this after opening the route inspector in whichever responsive state they
    are exercising.  The assertions intentionally avoid prescribing a particular pane, snap, or
    number of visible route-plan controls: those are presentation-specific.  They do, however,
    catch the invariants that must hold in every applicable state.
    """
    assert metrics.get("nested_interactives", 0) == 0, metrics
    assert metrics.get("horizontal_overflow", 0) <= tolerance_px, metrics
    assert metrics.get("document_horizontal_overflow", 0) <= tolerance_px, metrics
    assert not metrics.get("clipped_labels"), metrics
    assert not metrics.get("close_collisions"), metrics

    # A Plan's contextual close action can legitimately coexist with the inspector's close.  Only
    # the latter is singular; never mistake two different semantic actions for a duplicate close.
    inspector_close_count = metrics.get("inspector_close_visible_count", 0)
    if inspector_close_count:
        assert inspector_close_count == 1, metrics

    # When a scrollable Choices or Plan pane is present, its final content/action must be reachable
    # at max scroll.  A compact Peek has no active pane, so it is deliberately not constrained.
    if metrics.get("active_pane") != "none":
        assert metrics.get("scroll_reaches_last"), metrics

    sheet = metrics.get("sheet") or {}
    if sheet.get("is_bottom_sheet"):
        # The mobile card must occupy its real visible height; transformed full-height sheets make
        # the scroller report a false bottom before its last action is visible.
        assert sheet.get("hidden_height", 0) <= tolerance_px, metrics
        assert sheet.get("bottom_delta", 0) <= tolerance_px, metrics
        css_delta = sheet.get("height_delta")
        if css_delta is not None:
            assert css_delta <= tolerance_px, metrics


def dom_family_snapshot(page):
    return page.evaluate(
        """() => {
          const rows=[...document.querySelectorAll('#pincard .route-choice')];
          const visible=el=>{const s=getComputedStyle(el),r=el.getBoundingClientRect();return s.display!=='none'&&s.visibility!=='hidden'&&r.width>0&&r.height>0;};
          const cards=[...document.querySelectorAll('#pincard .route-choice-card')];
          const planActions=[...document.querySelectorAll('#pincard [data-route-plan-for]')];
          const plan=document.getElementById('route-plan-panel');
          const planRoot=plan&&plan.closest('#route-sidecar,#pincard');
          const families=[...new Map(rows.map(row=>[row.dataset.family,{
            key:row.dataset.family,name:row.dataset.familyName||''
          }])).values()];
          return ({
          families,
          choices:rows.map(b=>({
            key:b.dataset.choiceKey||b.dataset.key,family:b.dataset.family,branch:b.dataset.branch,
            family_name:b.dataset.familyName||'',branch_name:b.dataset.branchName||'',
            name:(b.querySelector('.route-name')||{}).textContent?.trim()||'',
            time:(b.querySelector('.route-time')||{}).childNodes?.[0]?.textContent?.trim()||'',
            facts:Object.fromEntries([...b.querySelectorAll('[data-fact]')].map(f=>[
              f.dataset.fact,(f.querySelector('.route-fact-value')||{}).textContent?.trim()||''
            ])),pressed:b.getAttribute('aria-pressed'),
          })),
          route_plan_actions:planActions.map(button=>({
            key:button.dataset.routePlanFor||'',controls:button.getAttribute('aria-controls')||'',
            expanded:button.getAttribute('aria-expanded'),visible:visible(button),
            selected:button.closest('.route-choice-card')?.dataset.selected||'false',
            label:(button.getAttribute('aria-label')||button.textContent||'').trim(),
          })),
          route_cards:cards.map(card=>({
            key:card.dataset.choiceCardKey||'',selected:card.dataset.selected||'false',
            route_buttons:card.querySelectorAll('.route-choice').length,
            route_plan_actions:card.querySelectorAll('[data-route-plan-for]').length,
          })),
          recommended:[...document.querySelectorAll('#pincard .route-recommendations .route-choice.recommended')].map(r=>({
            key:r.dataset.choiceKey||r.dataset.key,family:r.dataset.family,branch:r.dataset.branch,
          })),
          selected_keys:[...new Set([...document.querySelectorAll('#pincard .route-choice[aria-pressed="true"]')]
            .map(r=>r.dataset.choiceKey||r.dataset.key).filter(Boolean))],
          nested_interactives:[...document.querySelectorAll('#pincard button,#pincard a[href],#pincard summary,#pincard [role="button"]')]
            .filter(el=>{const p=el.parentElement&&el.parentElement.closest('button,a[href],summary,[role="button"]');return p&&document.getElementById('pincard').contains(p);}).length,
          stale_controls:document.querySelectorAll('#pincard .cmp .family,#pincard .cmp .branch,#pincard .cmp .strip').length,
          all_routes_present:!!document.getElementById('allroutes'),
          all_routes_expanded:document.getElementById('allroutes')?.open||false,
          legacy_global_plan_controls:document.querySelectorAll('#pincard [data-route-plan-control],#pincard [data-route-plan-toggle],#pincard #view-route-plan,#pincard #selected-plan-cta').length,
          // `data-sheet-snap` on #pincard is the current state hook, not retired UI.  Restrict
          // this regression guard to the former interactive tab/snap controls themselves.
          legacy_mobile_navigation:document.querySelectorAll('#pincard .pin-view,#pincard .pin-peek-actions,#pincard [data-sheet-snap-action],#pincard [data-sheet-tab]').length,
          map_choice_key:document.querySelector('#pincard .pin-shell')?.dataset.mapChoiceKey||null,
          recommended_choice_key:document.querySelector('#pincard .pin-shell')?.dataset.recommendedChoiceKey||null,
          selected_choice_key:plan?.dataset.selectedChoiceKey||
            document.querySelector('#pincard .pin-shell')?.dataset.selectedChoiceKey||null,
          choices_visible:!!document.getElementById('route-choices-panel')&&
            getComputedStyle(document.getElementById('route-choices-panel')).display!=='none',
          plan_visible:!!plan&&visible(plan),
          plan_root:planRoot?.id||null,
          plan_selected_choice_key:plan?.dataset.selectedChoiceKey||null,
          plan_step_count:plan?.querySelectorAll('.route-directions li').length||0,
          plan_has_collapsed_directions:!!plan?.querySelector('#route-directions summary,details.route-directions'),
          plan_has_google_maps_link:!!plan?.querySelector('.plan-google[href]'),
          drawn:(typeof DRAWN!=='undefined'&&DRAWN)?{multi:!!DRAWN.multi,key:DRAWN.key||null,family:DRAWN.famKey||null,branch:DRAWN.branchKey||null}:null,
          route_layers:(typeof routeLayer!=='undefined')?routeLayer.getLayers().length:0,
        })}"""
    )


def _display_route_title(family, branch):
    """Mirror the generic duplicate-catalog compaction used by the inspector."""
    family_name = str(family.get("name") or "Route")
    branch_name = str(branch.get("name") or "")
    compact = lambda value: " ".join(str(value or "").split()).lower()
    prefix = "walk after "
    normalized_branch = compact(branch_name)
    if (normalized_branch.startswith(prefix)
            and normalized_branch[len(prefix):] == compact(family_name)):
        return f"{family_name} → walk to destination"
    return (f"{family_name} → {branch_name}"
            if branch_name and branch_name != family_name else family_name)


def assert_api_matches_dom(body, snapshot):
    options = [body, *(body.get("alts") or [])]
    family_members = {}
    branch_members = {}
    for route in options:
        family, branch = route["family"], route["branch"]
        fkey, bkey = str(family["key"]), str(branch["key"])
        family_members.setdefault(fkey, {"meta": family, "routes": []})["routes"].append(route)
        branch_members.setdefault((fkey, bkey), {"meta": branch, "routes": []})["routes"].append(route)
    dom_families = {item["key"]: item for item in snapshot["families"]}
    dom_choices = {item["key"]: item for item in snapshot["choices"]}
    dom_branches = {(item["family"], item["branch"]) for item in snapshot["choices"]}
    api_choices = {str(route.get("choice_key") or ""): route for route in options}
    assert len(dom_families) == len(snapshot["families"]), "duplicate expert family sections"
    assert "" not in api_choices, "API route is missing a durable choice_key"
    assert len(api_choices) == len(options), "API repeated a choice_key"
    assert len(dom_choices) == len(snapshot["choices"]), "inspector repeated a route choice key"
    assert set(dom_families) == set(family_members), (
        f"DOM/API family mismatch: dom={set(dom_families)} api={set(family_members)}")
    assert dom_branches == set(branch_members), (
        f"DOM/API branch mismatch: dom={set(dom_branches)} api={set(branch_members)}")
    assert set(dom_choices) == set(api_choices), (
        f"DOM/API choice mismatch: dom={set(dom_choices)} api={set(api_choices)}")
    if snapshot["all_routes_present"]:
        assert snapshot["all_routes_expanded"], (
            "remaining-choice disclosure was present but not expanded for audit")
    assert snapshot["stale_controls"] == 0, "retired nested family/branch controls returned"
    assert snapshot["nested_interactives"] == 0, "route inspector nested interactive controls"
    assert snapshot["legacy_global_plan_controls"] == 0, "retired global Route Plan control returned"
    assert snapshot["legacy_mobile_navigation"] == 0, "retired mobile tab/snap controls returned"
    assert snapshot["drawn"] and snapshot["drawn"]["multi"], "pinned card is not a family diagram"
    assert snapshot["route_layers"] > 0, "family card advertises routes but the map drew none"
    for key, value in family_members.items():
        dom, meta = dom_families[key], value["meta"]
        assert dom["name"] == str(meta.get("name") or "")
    for choice_key, route in api_choices.items():
        dom = dom_choices[choice_key]
        family, meta = route["family"], route["branch"]
        key = (str(family["key"]), str(meta["key"]))
        expected_name = _display_route_title(family, meta)
        assert dom["name"] == expected_name
        assert (dom["family"], dom["branch"]) == key
        assert dom["time"] == f"{_active_total(route)} min"
        assert dom["facts"].get("walk"), f"route {key} is missing compact walking fact"
        assert dom["facts"].get("transfers") is not None, f"route {key} is missing transfers fact"
        assert dom["facts"].get("bad-day"), f"route {key} is missing bad-day fact"
        assert dom["key"], f"expert route {key} has no authoritative option key"
        assert dom["pressed"] in {"true", "false"}, f"route {key} lacks aria-pressed state"

    cards_by_key = {item["key"]: item for item in snapshot["route_cards"]}
    actions_by_key = {item["key"]: item for item in snapshot["route_plan_actions"]}
    assert set(cards_by_key) == set(api_choices), "route-card/API identity mismatch"
    assert set(actions_by_key) == set(api_choices), "each advertised route needs one local Plan action"
    for choice_key, card in cards_by_key.items():
        assert card["route_buttons"] == 1, f"route {choice_key} has {card['route_buttons']} selection buttons"
        assert card["route_plan_actions"] == 1, f"route {choice_key} has {card['route_plan_actions']} Plan actions"
        action = actions_by_key[choice_key]
        assert action["controls"] == "route-plan-panel", (
            f"route {choice_key} Plan action controls {action['controls']!r}, not the active Plan")
        assert action["expanded"] in {"true", "false"}, f"route {choice_key} Plan action lacks aria-expanded"
        assert action["label"], f"route {choice_key} Plan action lacks an accessible name"

    assert len(snapshot["recommended"]) == 1, "expected one recommendation-first row"
    recommended = snapshot["recommended"][0]
    map_choice_key = str(body.get("map_choice_key") or body["choice_key"])
    recommended_choice_key = str(body.get("recommended_choice_key") or map_choice_key)
    assert snapshot["map_choice_key"] == map_choice_key
    assert snapshot["recommended_choice_key"] == recommended_choice_key
    assert recommended["key"] == recommended_choice_key
    assert snapshot["selected_keys"] == [recommended["key"]], (
        f"expected one selected route key, got {snapshot['selected_keys']}")
    assert snapshot["selected_choice_key"] == recommended_choice_key
    assert snapshot["drawn"]["key"] == recommended["key"]
    selected_actions = [item for item in snapshot["route_plan_actions"] if item["selected"] == "true"]
    assert [item["key"] for item in selected_actions] == [recommended_choice_key]
    if snapshot["plan_visible"]:
        assert snapshot["plan_selected_choice_key"] == recommended_choice_key
        assert snapshot["plan_step_count"] > 0, "open Route Plan has no step-by-step directions"
        assert not snapshot["plan_has_collapsed_directions"], "Route Plan directions are still collapsed"
        assert snapshot["plan_has_google_maps_link"], "open Route Plan lacks Google Maps handoff"
