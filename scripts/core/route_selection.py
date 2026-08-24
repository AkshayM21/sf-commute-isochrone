"""Pure route-choice dominance, diversity, and selection primitives.

The route server owns the callbacks that understand its hydrated itinerary shape.  This module
owns only the policy that combines those callbacks: exact-choice de-duplication, Pareto pruning,
and breadth/diversity selection.  Keeping the callback boundary explicit makes the policy usable
with small fixtures and prevents it from reaching into Flask, RAPTOR state, or mutable caches.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence


Option = MutableMapping[str, Any]
FamilyKeys = Mapping[int, Any]


@dataclass(frozen=True)
class SelectionOps:
    """Behavioral callbacks supplied by the route representation layer.

    The callbacks deliberately operate on the existing option dictionaries.  Selection mutates
    only the same private family metadata fields the legacy implementation attached to options;
    callers remain responsible for copying options when they need isolation.
    """

    family_key: Callable[[Option, Optional[FamilyKeys]], Any]
    branch_key: Callable[[Option, Any], Any]
    choice_bucket: Callable[[Option], Any]
    choice_equivalent: Callable[[Option, Option], bool]
    quality_rank: Callable[[Option], Any]
    total: Callable[[Option], Any]
    exact_seconds: Callable[[Option], Any]
    access_walk_min: Callable[[Option], Any]
    transfers: Callable[[Option], Any]
    final_walk_min: Callable[[Option], Any]
    physical_walk_min: Callable[[Option], Any]
    fragility: Callable[[Option], Any]
    transit_legs: Callable[[Option], Sequence[Mapping[str, Any]]]
    leg_name: Callable[[Mapping[str, Any]], str]
    service_meta: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    discover_family_keys: Callable[[Sequence[Option]], Mapping[int, Any]]


def _family(ops: SelectionOps, option: Option, family_keys: Optional[FamilyKeys]) -> Any:
    return ops.family_key(option, family_keys)


def _branch(ops: SelectionOps, option: Option, family: Any) -> Any:
    return ops.branch_key(option, family)


def alt_dominates(
    simple: Option,
    detour: Option,
    family_keys: Optional[FamilyKeys],
    *,
    ops: SelectionOps,
) -> bool:
    """Return whether ``simple`` Pareto-dominates ``detour`` in one branch."""

    sfam = _family(ops, simple, family_keys)
    dfam = _family(ops, detour, family_keys)
    sbranch = _branch(ops, simple, sfam)
    dbranch = _branch(ops, detour, dfam)
    if sfam != dfam or sbranch != dbranch:
        return False
    sq = (
        ops.total(simple),
        ops.exact_seconds(simple),
        ops.access_walk_min(simple),
        ops.transfers(simple),
        ops.final_walk_min(simple),
        ops.physical_walk_min(simple),
        ops.fragility(simple),
    )
    dq = (
        ops.total(detour),
        ops.exact_seconds(detour),
        ops.access_walk_min(detour),
        ops.transfers(detour),
        ops.final_walk_min(detour),
        ops.physical_walk_min(detour),
        ops.fragility(detour),
    )
    return all(left <= right for left, right in zip(sq, dq)) and any(
        left < right for left, right in zip(sq, dq)
    )


def prune_dominated_alts(
    alts: Iterable[Option],
    context: Iterable[Option] = (),
    family_keys: Optional[FamilyKeys] = None,
    *,
    ops: SelectionOps,
) -> list[Option]:
    """De-duplicate exact choices and remove strictly dominated alternatives."""

    context = list(context or ())
    alts = list(alts)
    comparisons = context + alts
    family_keys = family_keys or ops.discover_family_keys(comparisons)

    by_choice: list[list[Option]] = []
    groups_by_bucket: dict[Any, list[list[Option]]] = {}
    for alt in sorted(alts, key=ops.quality_rank):
        bucket = ops.choice_bucket(alt)
        bucket_groups = groups_by_bucket.setdefault(bucket, [])
        compatible = [
            group
            for group in bucket_groups
            if all(ops.choice_equivalent(alt, member) for member in group)
        ]
        if compatible:
            compatible[0].append(alt)
        else:
            group = [alt]
            by_choice.append(group)
            bucket_groups.append(group)

    deduped: list[Option] = []
    for options in by_choice:
        winner = min(options, key=ops.quality_rank)
        winner_bucket = ops.choice_bucket(winner)
        primary_matches = [
            option
            for option in context
            if ops.choice_bucket(option) == winner_bucket
            if ops.choice_equivalent(option, winner)
        ]
        if primary_matches and min(ops.total(option) for option in primary_matches) <= ops.total(winner):
            continue
        deduped.append(winner)

    comparisons = context + deduped
    by_dominance: dict[tuple[Any, Any], list[Option]] = {}
    by_exact_choice: dict[Any, list[Option]] = {}
    for option in comparisons:
        family = _family(ops, option, family_keys)
        branch = _branch(ops, option, family)
        by_dominance.setdefault((family, branch), []).append(option)
        by_exact_choice.setdefault(ops.choice_bucket(option), []).append(option)

    kept: list[Option] = []
    for option in deduped:
        family = _family(ops, option, family_keys)
        branch = _branch(ops, option, family)
        candidates: list[Option] = []
        candidate_ids: set[int] = set()
        for pool in (
            by_dominance.get((family, branch), ()),
            by_exact_choice.get(ops.choice_bucket(option), ()),
        ):
            for candidate in pool:
                if id(candidate) not in candidate_ids:
                    candidate_ids.add(id(candidate))
                    candidates.append(candidate)
        if not any(
            candidate is not option
            and alt_dominates(candidate, option, family_keys, ops=ops)
            for candidate in candidates
        ):
            kept.append(option)
    return kept


def family_representative(family: Any, options: Sequence[Option], *, ops: SelectionOps) -> Optional[Option]:
    """Pick the fastest truthful member of a family/branch pool."""

    del family  # retained in the signature to mirror the server's compatibility helper
    return min(options, key=ops.quality_rank) if options else None


def build_family_service_catalog(
    routes: Iterable[Option],
    family_keys: Optional[FamilyKeys],
    *,
    ops: SelectionOps,
) -> "OrderedDict[Any, OrderedDict[Any, dict[str, Any]]]":
    """Catalog proven first-board services for each selected family and branch."""

    catalog: "OrderedDict[Any, OrderedDict[Any, dict[str, Any]]]" = OrderedDict()
    for route in routes or ():
        transit = ops.transit_legs(route)
        if not transit:
            continue
        family = _family(ops, route, family_keys)
        branch = _branch(ops, route, family)
        service = dict(ops.service_meta(transit[0]))
        branches = catalog.setdefault(family, OrderedDict())
        branch_meta = branches.setdefault(branch, {"services": OrderedDict(), "tails": []})
        services = branch_meta["services"]
        services.setdefault(service["key"], service)
        tail = tuple(
            name
            for leg in transit[1:]
            if (name := ops.leg_name(leg))
        )
        if tail and tail not in branch_meta["tails"]:
            branch_meta["tails"].append(tail)
    return catalog


def select_diverse_alts(
    alts: Iterable[Option],
    cap: int,
    *,
    primary: Optional[Option] = None,
    complete_selected_families: bool = False,
    force_include: Any = None,
    near_tie_min: int = 3,
    ops: SelectionOps,
) -> list[Option]:
    """Select bounded alternatives while reserving corridor and tail breadth."""

    alts = list(alts)
    universe = ([primary] if primary else []) + alts
    family_keys = ops.discover_family_keys(universe)
    family_catalog = build_family_service_catalog(universe, family_keys, ops=ops)
    primary_family = _family(ops, primary, family_keys) if primary else None
    if primary is not None:
        primary["_family_seed"] = primary_family
        primary["_family_catalog"] = family_catalog
    alts = prune_dominated_alts(
        alts, [primary] if primary else (), family_keys, ops=ops
    )
    ordered = sorted(alts, key=ops.quality_rank)
    selected: list[Option] = []
    selected_ids: set[int] = set()

    def add(option: Option) -> bool:
        ident = id(option)
        if ident in selected_ids or len(selected) >= cap:
            return False
        selected.append(option)
        selected_ids.add(ident)
        return True

    families: "OrderedDict[Any, list[Option]]" = OrderedDict()
    for option in ordered:
        families.setdefault(_family(ops, option, family_keys), []).append(option)

    primary_branch = _branch(ops, primary, primary_family) if primary else None
    family_order = [item for item in families.items() if item[0] != primary_family]
    if primary_family in families:
        family_order.append((primary_family, families[primary_family]))
    reserved: Optional[Option] = None
    reservation_ceiling: Optional[Any] = None
    if cap >= 3:
        missing_items = [(fam, opts) for fam, opts in family_order if fam != primary_family]
        missing_order = [fam for fam, _opts in missing_items]
        eligible_families = set(missing_order[: max(0, cap - 1)])
        if primary_family is not None:
            eligible_families.add(primary_family)
        displaced = (
            family_representative(*missing_items[cap - 1], ops=ops)
            if len(missing_items) >= cap
            else None
        )
        reservation_ceiling = (
            ops.total(displaced) + near_tie_min if displaced is not None else None
        )
        reserve_candidates = []
        for fam, options in family_order:
            if fam not in eligible_families:
                continue
            if fam == primary_family:
                visible_branches = {primary_branch} if primary_branch else set()
                family_head = ops.total(primary)
            else:
                anchor = family_representative(fam, options, ops=ops)
                if anchor is None:
                    continue
                visible_branches = {_branch(ops, anchor, fam)}
                family_head = ops.total(anchor)
            by_branch: "OrderedDict[Any, list[Option]]" = OrderedDict()
            for option in options:
                branch = _branch(ops, option, fam)
                if branch not in visible_branches:
                    by_branch.setdefault(branch, []).append(option)
            sibling_reps = [
                family_representative(fam, branch_opts, ops=ops)
                for branch_opts in by_branch.values()
            ]
            sibling_reps = [option for option in sibling_reps if option is not None]
            if sibling_reps:
                sibling = min(sibling_reps, key=ops.quality_rank)
                if reservation_ceiling is not None and ops.total(sibling) > reservation_ceiling:
                    continue
                reserve_candidates.append(
                    (0 if fam != primary_family else 1,
                     family_head, ops.total(sibling), fam, sibling)
                )
        if reserve_candidates:
            reserved = min(reserve_candidates, key=lambda item: item[:4])[4]

    breadth_cap = max(0, cap - (1 if reserved is not None else 0))
    for fam, options in family_order:
        if fam == primary_family:
            continue
        representative = family_representative(fam, options, ops=ops)
        if representative is not None:
            add(representative)
        if len(selected) >= breadth_cap:
            break
    if reserved is not None:
        add(reserved)

    while len(selected) < cap:
        progressed = False
        for fam, options in family_order:
            represented = {
                _branch(ops, option, fam)
                for option in selected
                if _family(ops, option, family_keys) == fam
            }
            if fam == primary_family and primary_branch:
                represented.add(primary_branch)
            by_branch: "OrderedDict[Any, list[Option]]" = OrderedDict()
            for option in options:
                branch = _branch(ops, option, fam)
                if branch not in represented:
                    by_branch.setdefault(branch, []).append(option)
            if not by_branch:
                continue
            branch_options = next(iter(by_branch.values()))
            representative = family_representative(fam, branch_options, ops=ops)
            if representative is not None and add(representative):
                progressed = True
            if len(selected) >= cap:
                break
        if not progressed:
            break

    for option in ordered:
        add(option)
        if len(selected) >= cap:
            break

    if complete_selected_families and cap > 0:
        selected_families = {_family(ops, option, family_keys) for option in selected}
        if primary_family is not None:
            selected_families.add(primary_family)
        represented = {
            (_family(ops, option, family_keys),
             _branch(ops, option, _family(ops, option, family_keys)))
            for option in selected
        }
        if primary is not None and primary_family is not None and primary_branch is not None:
            represented.add((primary_family, primary_branch))

        def add_completion(option: Option, fam: Any, branch: Any) -> None:
            if (fam, branch) in represented:
                return
            if reservation_ceiling is not None and ops.total(option) > reservation_ceiling:
                return
            selected.append(option)
            selected_ids.add(id(option))
            represented.add((fam, branch))

        for fam, options in family_order:
            if fam not in selected_families:
                continue
            family_ceiling = min(ops.total(option) for option in options) + near_tie_min
            by_branch = OrderedDict()
            for option in options:
                branch = _branch(ops, option, fam)
                if (fam, branch) not in represented:
                    by_branch.setdefault(branch, []).append(option)
            for branch, branch_options in by_branch.items():
                representative = family_representative(fam, branch_options, ops=ops)
                if representative is not None and ops.total(representative) <= family_ceiling:
                    add_completion(representative, fam, branch)

    result = sorted(selected, key=ops.quality_rank)
    if not complete_selected_families:
        result = result[:cap]
    forced = (
        list(force_include)
        if isinstance(force_include, (list, tuple, set))
        else [force_include]
    )
    for forced_option in forced:
        if (
            forced_option is not None
            and any(forced_option is option for option in alts)
            and not any(forced_option is option for option in result)
        ):
            result.append(forced_option)
    result.sort(key=ops.quality_rank)
    for option in result:
        option["_family_seed"] = _family(ops, option, family_keys)
        option["_family_catalog"] = family_catalog
        if primary_family is not None:
            option["_primary_family_seed"] = primary_family
    return result
