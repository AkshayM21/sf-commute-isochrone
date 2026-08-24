"""Pure route-geometry and reliability hydration helpers.

The route server owns RAPTOR trees, walk-path providers, caches, and the Monte-Carlo
replay that produces reliability values.  This module owns only the small boundary
between those runtime objects and the JSON-shaped route options: copying a traced
journey, hydrating a geometry-free planned branch, and projecting an already-computed
reliability result onto selected response options.

All runtime behavior is supplied as callbacks.  In particular, this module never
imports ``server_raptor`` and never reaches into a tree, cache, Flask request, or engine.
That keeps the helpers straightforward to test and prevents a geometry extraction from
silently changing route selection.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any


Option = MutableMapping[str, Any]
Journey = Mapping[str, Any]


def journey_payload(journey: Journey | None) -> dict[str, Any] | None:
    """Return an independent public journey payload.

    Cached journey dictionaries contain nested leg and geometry lists.  A shallow
    response copy is not sufficient when a caller annotates a pinned route or a test
    fixture mutates its result.  Deep-copying only the selected journey keeps the cache
    immutable at the response boundary while preserving the existing field contract.
    """
    if journey is None:
        return None
    payload = {
        key: deepcopy(journey[key])
        for key in ("total", "xfers")
        if key in journey
    }
    # Alternative cards intentionally call the geometric journey ``legs``.  The legacy
    # server-side _add/hydrate code sourced this field from itinerary['geom']; formatted
    # display legs do not contain the transit ``pts`` consumed by drawCompareRoutes.
    # Prefer geom whenever both fields exist, and avoid duplicating the same large polyline
    # under a second public key.
    source_legs = journey.get("geom")
    if source_legs is None:
        source_legs = journey.get("legs")
    if source_legs is not None:
        payload["legs"] = deepcopy(source_legs)
    return payload


def hydrate_planned_proxy(
    option: Option,
    *,
    formatter: Callable[..., Journey | None] | None,
    transit_predicate: Callable[[Sequence[Mapping[str, Any]] | None], bool],
    cell: int,
    provider: Any,
) -> dict[str, Any] | None:
    """Hydrate one geometry-free planned branch option.

    Branch enumeration deliberately omits street paths so structural selection can run
    without paying for a Dijkstra per candidate.  Once selection has retained a branch,
    the caller supplies the tree's historical ``_format_planned_raw`` callback and the
    shared geometry provider.  ``None`` means the branch became invalid when exact
    geometry was requested (unreachable or no transit leg).

    The input option is never mutated.  Private selection handles are retained in the
    returned copy because the server removes them only after recommendation identity
    has been resolved; this preserves the old object-identity behavior at that layer
    without making this pure helper stateful.
    """
    if not option.get("_needs_hydration", False):
        # ``_branch`` can contain a live JourneyTree and its numpy-backed tables.  It is
        # a server-only selection handle, not response data, so preserve that handle by
        # reference while copying ordinary public fields.
        copied = dict(option)
        for key, value in list(copied.items()):
            if key != "_branch":
                copied[key] = deepcopy(value)
        return copied
    if formatter is None:
        return None
    branch = option.get("_branch") or {}
    itinerary = formatter(
        cell,
        branch.get("stop"),
        branch.get("raw"),
        branch.get("home"),
        branch.get("jt"),
        geom_provider=provider,
        planned_total=branch.get("total"),
        planned_target_sec=branch.get("target_sec", branch.get("metric_sec")),
    )
    if itinerary is None or not transit_predicate(itinerary.get("geom")):
        return None
    hydrated = dict(option)
    for key, value in list(hydrated.items()):
        if key not in ("_branch", "typical", "best"):
            hydrated[key] = deepcopy(value)
    hydrated.pop("_needs_hydration", None)
    payload = journey_payload(itinerary)
    hydrated["typical"] = payload
    hydrated["best"] = deepcopy(payload)
    return hydrated


def alternative_payload(
    line: Any,
    typical: Journey | None,
    best: Journey | None,
    *,
    source: str = "window",
    via_stop: int | None = None,
    route_label: Callable[[Any], str],
    trace_signature: Callable[[Any], Any],
    transit_predicate: Callable[[Sequence[Mapping[str, Any]] | None], bool],
) -> tuple[dict[str, Any] | None, Any | None]:
    """Build one independent alternative card from traced journey results.

    Returns ``(option, signature)``.  ``option`` is ``None`` for a walk-only route or
    an exact duplicate; duplicate filtering remains server-owned because the server
    owns the request-local ``seen`` set.  The helper preserves the historical rule that
    the displayed label comes from the traced geometry and retains ``chip_line`` when
    that label differs from the original MC chip.
    """
    if typical is None and best is None:
        return None, None
    has_best = best is not None and transit_predicate(best.get("geom"))
    has_typical = typical is not None and transit_predicate(typical.get("geom"))
    if not (has_best or has_typical):
        return None, None
    best_source = best or typical
    typical_source = typical or best
    sig = (trace_signature(best_source.get("geom")),
           trace_signature(typical_source.get("geom")))
    displayed = route_label(typical_source.get("geom")) or route_label(best_source.get("geom")) or line
    option: dict[str, Any] = {"line": displayed, "source": source}
    if displayed != line:
        option["chip_line"] = line
    if via_stop is not None:
        option["via_stop"] = int(via_stop)
    if typical is not None:
        option["typical"] = journey_payload(typical)
    if best is not None:
        option["best"] = journey_payload(best)
    return option, sig


def reliability_projection(
    result: Mapping[str, Any] | None,
    *,
    metric: str,
) -> tuple[tuple[int, int] | None, list[tuple[int, int] | None]]:
    """Normalize a route-typicals result for response assembly.

    ``result`` is the runtime-owned MC replay output.  This helper only validates and
    copies its scalar pairs, returning a primary pair and alternative pairs aligned to
    the supplied route list.  It intentionally does not invent reliability when the
    replay is absent.  ``metric`` documents the compatibility shape (``r``/``b``) and
    is validated here so a future caller cannot accidentally apply a best-case replay
    to the scheduled route.
    """
    if metric not in ("r", "b"):
        raise ValueError("metric must be 'r' or 'b'")
    if result is None:
        return None, []

    def pair(value: Any) -> tuple[int, int] | None:
        if value is None:
            return None
        try:
            if len(value) < 2:
                return None
            return int(value[0]), int(value[1])
        except (TypeError, ValueError, IndexError, OverflowError):
            return None

    primary = pair(result.get("prim"))
    alternatives = [pair(value) for value in (result.get("alts") or ())]
    return primary, alternatives


def apply_reliability(
    primary: MutableMapping[str, Any],
    alternatives: Sequence[MutableMapping[str, Any]],
    result: Mapping[str, Any] | None,
    *,
    metric: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return copied routes with reliability fields projected onto them.

    The input route dictionaries are never mutated and every returned route is
    independent.  Existing fields are preserved; a missing MC pair leaves the route
    unchanged.  The pair is interpreted as the historical ``(realistic, fragility)``
    tuple used by the arrive-by response.
    """
    prim_pair, alt_pairs = reliability_projection(result, metric=metric)
    prim = deepcopy(primary)
    alts = [deepcopy(option) for option in alternatives]
    if prim_pair is not None:
        prim["real"], prim["frag"] = prim_pair
    for option, values in zip(alts, alt_pairs):
        if values is not None:
            option["real"], option["frag"] = values
    return prim, alts


def apply_departafter_reliability(
    primary: MutableMapping[str, Any],
    alternatives: Sequence[MutableMapping[str, Any]],
    result: Mapping[str, Any] | None,
    *,
    metric: str,
) -> tuple[int | None, list[int | None], dict[str, Any], list[dict[str, Any]]]:
    """Project depart-after fragility pairs while retaining the scheduled totals.

    Depart-after responses expose only ``frag`` because their displayed totals are the
    bare scheduled p50.  The MC helper returns ``(committed_p50, unused, committed_p90)``
    internally; callers pass the already-derived ``(scheduled_total, fragility)`` pairs
    through ``result`` as ``prim_frag``/``alt_frags``.  This adapter keeps that mapping
    immutable and centralized.
    """
    if metric not in ("r", "b"):
        raise ValueError("metric must be 'r' or 'b'")
    prim = deepcopy(primary)
    alts = [deepcopy(option) for option in alternatives]
    if result is None:
        return None, [None for _ in alts], prim, alts
    prim_frag = result.get("prim_frag")
    prim_frag = None if prim_frag is None else max(0, int(prim_frag))
    alt_frags = []
    for option, frag in zip(alts, result.get("alt_frags") or ()):
        if frag is None:
            alt_frags.append(None)
        else:
            value = max(0, int(frag))
            option["frag"] = value
            alt_frags.append(value)
    alt_frags.extend([None] * (len(alts) - len(alt_frags)))
    if prim_frag is not None:
        prim["frag"] = prim_frag
    return prim_frag, alt_frags, prim, alts
