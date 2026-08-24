"""Bake graph-backed stop transfers into deterministic CSR arrays.

This module is deliberately an offline, side-effect-free boundary between the GTFS
transfer tables and the pedestrian graph.  It produces two explicit views: forward
source-to-target CSR for geometry, and reverse target-to-source CSR for RAPTOR's
backward sweep.  The two views share one rounded timing contract.

There are two kinds of physical transfer edges:

* Ordinary candidates are discovered by the caller's radius.  Their physical time
  and geometry come from one :meth:`core.walk.PathTree.route_result` call, so the
  served time and the drawn path cannot select different snap connectors.  A
  candidate whose endpoints are outside the graph support boundary or whose route
  is unreachable is omitted.
* A ``pathways.txt`` edge is an explicit, directed station connection.  Its
  ``traversal_seconds`` is authoritative and is used as the physical duration
  (not added to a graph duration).  Street geometry is never substituted for an
  indoor pathway.  The display geometry is the two endpoint coordinates, explicitly
  flagged as a fallback, which keeps the connection visible without claiming that
  the outdoor pedestrian graph describes it.

``transfers.txt`` fixed minimums are constraints, not walking costs.  The effective
duration is therefore ``max(physical_duration, fixed_minimum)``.  Unconditional
prohibitions remove a directed pair.  Route/trip-scoped minimums and prohibitions
are retained by :mod:`core.transfer_rules` but are intentionally not applied here
because this bake has no arriving/departing route context; they must never become
global rules.

The forward path representation is ``tr_forward_path_off`` plus
``tr_forward_path_points`` rather than graph node IDs.  ``PathTree.route_result`` intentionally exposes rounded ``[lat, lon]``
coordinates, not predecessor IDs, and pathway fallbacks contain stop endpoints that
are not graph nodes.  The coordinate representation is compact (two float columns),
preserves the exact geometry returned by the shared route API, and can represent both
ordinary and explicit-pathway edges without sentinel node IDs.

The result remains in memory; publication belongs to the access-artifact builder.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .transfer_rules import PathwayEdge, ScopedTransferRule, StopKey, TransferRuleSet
from .walk import WalkGraph


EARTH_RADIUS_M = 6_371_008.8
DEFAULT_RADIUS_M = 250.0
MAX_CANDIDATE_RADIUS_M = 2_000.0
DEFAULT_CAP_REF_SEC = 30.0 * 60.0


@dataclass(frozen=True)
class ScopedTransferRecord:
    """One preserved route/trip-scoped rule aligned to baked stop indexes.

    The graph bake cannot evaluate these rules because it has no arriving or
    departing route context.  Keeping the original immutable rule alongside its
    integer stop indexes lets the future route-aware kernel apply it without
    reparsing GTFS or promoting it to a global prohibition.  When a directed
    graph route is defensible, ``physical_seconds`` and ``path_points`` preserve
    that route as metadata only; they never enter global RAPTOR CSR.
    """

    source: int
    target: int
    rule: ScopedTransferRule
    # A route-aware consumer may need the physical route even though the current
    # one-label RAPTOR kernel cannot safely apply this rule.  These fields are
    # metadata only and never enter the global CSR.
    physical_seconds: float | None = None
    path_points: tuple[tuple[float, float], ...] = ()
    pathway_fallback: bool = False


@dataclass(frozen=True)
class PathwayTransferRecord:
    """One preserved pathway row aligned to baked stop indexes."""

    source: int
    target: int
    edge: PathwayEdge


@dataclass(frozen=True)
class GraphTransferBake:
    """Immutable forward and reverse arrays for directed stop transfers.

    All edge arrays are parallel and have length ``tr_off[-1]`` for the forward
    view (and ``tr_reverse_off[-1]`` for the reverse view).  ``tr_time`` and
    ``tr_reverse_time`` are integer effective reference seconds produced by one
    explicit half-up rounding policy.  ``tr_walk_time`` and ``tr_min_time`` retain
    the physical and fixed reference values as finite floats so runtime pace
    scaling can leave fixed minimums and authoritative pathways unchanged.  The
    forward source-to-target view is used for geometry; the reverse target-to-source
    view is used by RAPTOR's backward sweep.

    ``tr_forward_path_points`` stores ``[lat, lon]`` rows.  For edge ``e``, its
    path is ``tr_forward_path_points[tr_forward_path_off[e]:tr_forward_path_off[e + 1]]``.
    ``tr_forward_path_fallback[e]``
    is true for an explicit pathway, whose endpoint-to-endpoint segment is
    display-only geometry.  ``tr_pathway_metadata[e]`` retains every distinct
    pathway record for that edge in deterministic order; its first timed entry
    supplies the authoritative shortest traversal duration.  ``scoped_rules`` and
    ``pathway_records`` preserve all aligned source records even when no CSR edge
    can safely be emitted from the record alone.
    """

    tr_off: np.ndarray
    tr_to: np.ndarray
    tr_time: np.ndarray
    tr_walk_time: np.ndarray
    tr_min_time: np.ndarray
    tr_path_off: np.ndarray
    tr_path_points: np.ndarray
    tr_path_fallback: np.ndarray
    tr_pathway_metadata: tuple[tuple[PathwayEdge, ...], ...]
    tr_reverse_off: np.ndarray
    tr_reverse_to: np.ndarray
    tr_reverse_time: np.ndarray
    tr_reverse_walk_time: np.ndarray
    tr_reverse_min_time: np.ndarray
    tr_reverse_path_fallback: np.ndarray
    scoped_rules: tuple[ScopedTransferRecord, ...]
    pathway_records: tuple[PathwayTransferRecord, ...]

    @property
    def tr_forward_off(self):
        return self.tr_off

    @property
    def tr_forward_to(self):
        return self.tr_to

    @property
    def tr_forward_time(self):
        return self.tr_time

    @property
    def tr_forward_walk_time(self):
        return self.tr_walk_time

    @property
    def tr_forward_min_time(self):
        return self.tr_min_time

    @property
    def tr_forward_path_off(self):
        return self.tr_path_off

    @property
    def tr_forward_path_points(self):
        return self.tr_path_points

    @property
    def tr_forward_path_fallback(self):
        return self.tr_path_fallback

    @property
    def edge_count(self) -> int:
        return int(self.tr_to.size)

    def path_points(self, edge_index: int) -> np.ndarray:
        """Return a copy of one edge's ``[lat, lon]`` path."""

        edge_index = int(edge_index)
        if edge_index < 0 or edge_index >= self.edge_count:
            raise IndexError(edge_index)
        start, end = self.tr_path_off[edge_index:edge_index + 2]
        return self.tr_path_points[int(start):int(end)].copy()


@dataclass(frozen=True)
class _Candidate:
    source: int
    target: int
    physical_seconds: float
    fixed_seconds: float
    points: tuple[tuple[float, float], ...]
    pathway_fallback: bool
    pathway_metadata: tuple[PathwayEdge, ...]


def bake_graph_transfers(
    stop_keys: Sequence[StopKey],
    stop_lon: Sequence[float],
    stop_lat: Sequence[float],
    walk_graph: WalkGraph,
    rule_sets: TransferRuleSet | Iterable[TransferRuleSet] = (),
    *,
    radius_m: float = DEFAULT_RADIUS_M,
    cap_ref_sec: float = DEFAULT_CAP_REF_SEC,
) -> GraphTransferBake:
    """Build graph-backed directed transfers for aligned, namespaced stops.

    ``stop_keys``, ``stop_lon``, and ``stop_lat`` must describe the same stop
    ordering.  The ordinary radius is only a candidate-discovery mechanism; every
    emitted ordinary edge is validated by the directed pedestrian graph.  Explicit
    global transfer rows and pathway rows are unioned with the radius candidates,
    so a valid explicit pair outside the radius is still considered.

    ``rule_sets`` may be one :class:`TransferRuleSet` or an iterable of rule sets.
    Stop pairs whose keys are absent from ``stop_keys`` are ignored: a feed's
    optional tables may mention stops filtered out of the baked network.

    The output is deterministic for a fixed stop ordering and graph.  One edge is
    emitted per directed stop pair.  If a pair has a timed explicit pathway, the
    shortest deterministic timed record wins over an ordinary duplicate because
    the explicit station traversal is authoritative.  Otherwise a single graph
    route is used when the pair was independently discovered or constrained.
    """

    keys, lon, lat, valid = _validate_stops(stop_keys, stop_lon, stop_lat)
    radius = _candidate_radius(radius_m)
    cap = _finite_nonnegative(cap_ref_sec, "cap_ref_sec")
    rules = _normalise_rule_sets(rule_sets)
    by_key = {key: index for index, key in enumerate(keys)}

    fixed_by_pair, prohibited, pathways, scoped_rules, pathway_records = _collect_rules(
        rules, by_key)
    ordinary_pairs = _ordinary_pairs(lon, lat, valid, walk_graph, radius)
    explicit_pairs = {
        (by_key[pair[0]], by_key[pair[1]])
        for pair in (set(fixed_by_pair) | set(prohibited) | set(pathways))
    }

    # A route tree is the expensive, graph-sized operation.  Group candidates by
    # source and release each tree before advancing.  This gives the bake one live
    # predecessor tree at a time instead of O(number of source stops) retained RAM.
    pairs_by_source: list[list[int]] = [[] for _ in keys]
    for source, target in sorted(ordinary_pairs | explicit_pairs):
        pairs_by_source[source].append(target)

    candidates: list[_Candidate] = []
    for source, targets in enumerate(pairs_by_source):
        tree = None
        for target in targets:
            pair = (keys[source], keys[target])
            if pair in prohibited:
                continue
            if not (bool(valid[source]) and bool(valid[target])):
                # Preserve the source rule metadata, but never invent a path or a cost for a
                # malformed coordinate.  The caller retains this gid in the build identity.
                continue

            pathway_metadata = pathways.get(pair, ())
            points: tuple[tuple[float, float], ...]
            fallback = False
            timed_pathways = tuple(
                edge for edge in pathway_metadata if edge.traversal_seconds is not None)
            if timed_pathways:
                # pathways.txt describes an indoor/station connection.  A street
                # route is not evidence of that indoor geometry, even when the
                # endpoints happen to snap to a connected pedestrian graph.  Keep
                # the authoritative traversal time and mark the endpoint segment as
                # a display-only fallback until an indoor geometry source exists.
                physical = float(timed_pathways[0].traversal_seconds)
                points = (
                    (float(lat[source]), float(lon[source])),
                    (float(lat[target]), float(lon[target])),
                )
                fallback = True
            else:
                # Ordinary and explicit non-pathway transfers need a real street
                # route.  In particular, an explicit transfer row outside the
                # supported SF graph is not enough to fabricate one.  A pathway
                # without traversal_time is preserved in pathway_records, but it
                # cannot by itself supply a safely costed transfer edge.
                if ((source, target) not in ordinary_pairs
                        and pair not in fixed_by_pair):
                    continue
                if not _supported_pair(walk_graph, lon, lat, source, target):
                    continue
                if tree is None:
                    tree = walk_graph.path_tree(
                        (float(lon[source]), float(lat[source])), cap)
                result = tree.route_result((float(lon[target]), float(lat[target])))
                if result is None:
                    continue
                physical = _route_seconds(result)
                points = _normalise_points(
                    result.points, (float(lat[source]), float(lon[source])),
                    (float(lat[target]), float(lon[target])))

            fixed = float(fixed_by_pair.get(pair, 0))
            candidates.append(_Candidate(
                source=source,
                target=target,
                physical_seconds=physical,
                fixed_seconds=fixed,
                points=points,
                pathway_fallback=fallback,
                pathway_metadata=pathway_metadata,
            ))
        # Make the memory lifetime intentional.  Clearing before the next source's
        # path_tree call also makes this invariant observable in a spy test.
        tree = None

    # Scoped rows are never promoted into the global RAPTOR CSR.  Still, preserve a
    # graph-backed physical path/time for a future route-aware kernel.  Reuse the
    # global candidate when the pair was already ordinary/global/pathway-discovered;
    # otherwise route the scoped-only pair outside the ordinary radius, unless an
    # untimed pathway is the only evidence (that pathway remains metadata only).
    candidate_by_pair = {(candidate.source, candidate.target): candidate
                         for candidate in candidates}
    scoped_by_source: list[list[ScopedTransferRecord]] = [[] for _ in keys]
    for record in scoped_rules:
        scoped_by_source[record.source].append(record)
    enriched_scoped: list[ScopedTransferRecord] = []
    for source, records in enumerate(scoped_by_source):
        tree = None
        for record in records:
            pair = (keys[record.source], keys[record.target])
            candidate = candidate_by_pair.get((record.source, record.target))
            if candidate is not None:
                enriched_scoped.append(ScopedTransferRecord(
                    record.source, record.target, record.rule,
                    candidate.physical_seconds, candidate.points,
                    candidate.pathway_fallback))
                continue
            pathway_metadata = pathways.get(pair, ())
            timed_pathways = tuple(
                edge for edge in pathway_metadata if edge.traversal_seconds is not None)
            if timed_pathways:
                physical = float(timed_pathways[0].traversal_seconds)
                points = (
                    (float(lat[record.source]), float(lon[record.source])),
                    (float(lat[record.target]), float(lon[record.target])),
                ) if bool(valid[record.source]) and bool(valid[record.target]) else ()
                if points:
                    enriched_scoped.append(ScopedTransferRecord(
                        record.source, record.target, record.rule,
                        physical, points, True))
                else:
                    enriched_scoped.append(record)
                continue
            # An untimed pathway cannot justify inventing a graph edge.  A scoped
            # rule paired with it retains only the rule/pathway metadata.
            if pathway_metadata or not (bool(valid[record.source]) and bool(valid[record.target])):
                enriched_scoped.append(record)
                continue
            if not _supported_pair(walk_graph, lon, lat, record.source, record.target):
                enriched_scoped.append(record)
                continue
            if tree is None:
                tree = walk_graph.path_tree(
                    (float(lon[record.source]), float(lat[record.source])), cap)
            result = tree.route_result((float(lon[record.target]), float(lat[record.target])))
            if result is None:
                enriched_scoped.append(record)
                continue
            physical = _route_seconds(result)
            points = _normalise_points(
                result.points,
                (float(lat[record.source]), float(lon[record.source])),
                (float(lat[record.target]), float(lon[record.target])))
            enriched_scoped.append(ScopedTransferRecord(
                record.source, record.target, record.rule, physical, points, False))
        tree = None

    return _pack_csr(candidates, len(keys), tuple(enriched_scoped), pathway_records)


def _validate_stops(
    stop_keys: Sequence[StopKey],
    stop_lon: Sequence[float],
    stop_lat: Sequence[float],
) -> tuple[tuple[StopKey, ...], np.ndarray, np.ndarray, np.ndarray]:
    keys = tuple(stop_keys)
    lon = np.asarray(stop_lon, dtype=np.float64).reshape(-1)
    lat = np.asarray(stop_lat, dtype=np.float64).reshape(-1)
    if len(keys) != len(lon) or len(keys) != len(lat):
        raise ValueError("stop_keys, stop_lon, and stop_lat must have equal lengths")
    if len(set(keys)) != len(keys):
        raise ValueError("stop_keys must be unique")
    # Keep malformed stops aligned to their gids, but drop them from candidate discovery and
    # route generation instead of aborting an otherwise usable feed bake.
    valid = (np.isfinite(lon) & np.isfinite(lat)
             & (lon >= -180.0) & (lon <= 180.0)
             & (lat >= -90.0) & (lat <= 90.0))
    return keys, lon, lat, valid


def _finite_nonnegative(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _candidate_radius(value: float) -> float:
    radius = _finite_nonnegative(value, "radius_m")
    if radius > MAX_CANDIDATE_RADIUS_M:
        raise ValueError(
            f"radius_m must be at most {MAX_CANDIDATE_RADIUS_M:g} metres")
    return radius


def _normalise_rule_sets(
    rule_sets: TransferRuleSet | Iterable[TransferRuleSet],
) -> tuple[TransferRuleSet, ...]:
    if isinstance(rule_sets, TransferRuleSet):
        return (rule_sets,)
    return tuple(rule_sets)


def _collect_rules(
    rule_sets: Iterable[TransferRuleSet],
    by_key: Mapping[StopKey, int],
) -> tuple[
    dict[tuple[StopKey, StopKey], int],
    set[tuple[StopKey, StopKey]],
    dict[tuple[StopKey, StopKey], tuple[PathwayEdge, ...]],
    tuple[ScopedTransferRecord, ...],
    tuple[PathwayTransferRecord, ...],
]:
    fixed: dict[tuple[StopKey, StopKey], int] = {}
    prohibited: set[tuple[StopKey, StopKey]] = set()
    pathways: dict[tuple[StopKey, StopKey], set[PathwayEdge]] = {}
    scoped: dict[tuple[object, ...], ScopedTransferRecord] = {}
    for rules in rule_sets:
        for pair, seconds in rules.min_transfer_seconds.items():
            if pair[0] in by_key and pair[1] in by_key:
                fixed[pair] = max(fixed.get(pair, 0), int(seconds))
        for pair in rules.prohibited_pairs:
            if pair[0] in by_key and pair[1] in by_key:
                prohibited.add(pair)
        for edge in rules.pathway_edges:
            pair = edge.pair
            if pair[0] not in by_key or pair[1] not in by_key:
                continue
            pathways.setdefault(pair, set()).add(edge)
        for rule in rules.route_scoped_rules:
            pair = rule.pair
            if pair[0] not in by_key or pair[1] not in by_key:
                continue
            record = ScopedTransferRecord(by_key[pair[0]], by_key[pair[1]], rule)
            scoped[_scoped_record_key(record)] = record
    sorted_pathways = {
        pair: tuple(sorted(edges, key=_pathway_key))
        for pair, edges in pathways.items()
    }
    sorted_scoped = tuple(scoped[key] for key in sorted(scoped))
    pathway_records = tuple(
        PathwayTransferRecord(by_key[pair[0]], by_key[pair[1]], edge)
        for pair in sorted(sorted_pathways, key=lambda item: (by_key[item[0]], by_key[item[1]]))
        for edge in sorted_pathways[pair]
    )
    return fixed, prohibited, sorted_pathways, sorted_scoped, pathway_records


def _pathway_key(edge: PathwayEdge) -> tuple[object, ...]:
    return (
        edge.traversal_seconds is None,
        int(edge.traversal_seconds) if edge.traversal_seconds is not None else 0,
        edge.length_meters is None,
        float(edge.length_meters) if edge.length_meters is not None else 0.0,
        edge.pathway_id,
        edge.pathway_mode or "",
        bool(edge.reversed_from_bidirectional),
    )


def _scoped_record_key(record: ScopedTransferRecord) -> tuple[object, ...]:
    rule = record.rule
    return (
        record.source,
        record.target,
        rule.from_route_id or "",
        rule.to_route_id or "",
        rule.from_trip_id or "",
        rule.to_trip_id or "",
        rule.transfer_type or "",
        rule.min_transfer_seconds is None,
        rule.min_transfer_seconds if rule.min_transfer_seconds is not None else 0,
        rule.source,
        bool(rule.prohibited),
    )


def _ordinary_pairs(
    lon: np.ndarray,
    lat: np.ndarray,
    valid: np.ndarray,
    walk_graph: WalkGraph,
    radius_m: float,
) -> set[tuple[int, int]]:
    if len(lon) < 2 or radius_m <= 0.0:
        return set()
    # Use the walk graph's local metres projection only to cheaply discover
    # neighbors, then apply a haversine check so the public radius is geographic.
    xy = np.column_stack((lon * float(walk_graph.mlon), lat * float(walk_graph.mlat)))
    valid_indexes = np.flatnonzero(valid)
    if len(valid_indexes) < 2:
        return set()
    tree = cKDTree(xy[valid_indexes])
    pairs: set[tuple[int, int]] = set()
    for source in valid_indexes:
        nearby = tree.query_ball_point(xy[source], radius_m)
        for target_local in nearby:
            target = int(valid_indexes[int(target_local)])
            if target == source:
                continue
            if _great_circle_m(lon[source], lat[source], lon[target], lat[target]) <= radius_m:
                pairs.add((source, target))
    return pairs


def _great_circle_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return EARTH_RADIUS_M * 2.0 * math.asin(math.sqrt(min(1.0, a)))


def _supported_pair(
    walk_graph: WalkGraph,
    lon: np.ndarray,
    lat: np.ndarray,
    source: int,
    target: int,
) -> bool:
    # supports_point is intentionally called at the transfer-bake boundary.  It
    # prevents permissive historical snapping from fabricating a graph route for
    # a stop beyond the supported pedestrian graph.
    return bool(
        walk_graph.supports_point(float(lon[source]), float(lat[source]))
        and walk_graph.supports_point(float(lon[target]), float(lat[target]))
    )


def _route_seconds(result: object) -> float:
    try:
        value = float(result.seconds)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("PathTree.route_result returned an invalid duration") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("PathTree.route_result returned an invalid duration")
    return value


def _normalise_points(
    points: object,
    source_ll: tuple[float, float] | None = None,
    target_ll: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], ...]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("PathTree.route_result returned invalid path points")
    if (np.any(array[:, 0] < -90.0) or np.any(array[:, 0] > 90.0)
            or np.any(array[:, 1] < -180.0) or np.any(array[:, 1] > 180.0)):
        raise ValueError("PathTree.route_result returned out-of-range path coordinates")
    if source_ll is not None and target_ll is not None:
        if (not np.allclose(array[0], np.asarray(source_ll, dtype=np.float64), atol=1e-5, rtol=0.0)
                or not np.allclose(array[-1], np.asarray(target_ll, dtype=np.float64),
                                   atol=1e-5, rtol=0.0)):
            raise ValueError("PathTree.route_result returned a path disconnected from its endpoints")
    # WalkRouteResult points are [lat, lon].  Keep that contract explicit and
    # stable rather than swapping axes based on heuristics.
    return tuple((float(row[0]), float(row[1])) for row in array)


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _round_seconds(value: float) -> int:
    """Round one non-negative reference duration once, consistently (half-up)."""
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("transfer duration must be finite and non-negative")
    return int(math.floor(value + 0.5))


def _pack_direction(
    rows: list[list[_Candidate]],
    *,
    reverse: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat = [candidate for row in rows for candidate in row]
    off = np.zeros(len(rows) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(row) for row in rows], dtype=np.int64)
    to = np.asarray([item.source if reverse else item.target for item in flat], dtype=np.int32)
    walk = np.asarray([float(item.physical_seconds) for item in flat], dtype=np.float64)
    minimum = np.asarray([float(item.fixed_seconds) for item in flat], dtype=np.float64)
    effective = np.asarray([_round_seconds(max(item.physical_seconds, item.fixed_seconds))
                            for item in flat], dtype=np.int64)
    fallback = np.asarray([item.pathway_fallback for item in flat], dtype=bool)
    return (off, to, walk, minimum, effective, fallback)


def _pack_csr(
    candidates: Iterable[_Candidate],
    n_stops: int,
    scoped_rules: tuple[ScopedTransferRecord, ...],
    pathway_records: tuple[PathwayTransferRecord, ...],
) -> GraphTransferBake:
    ordered = sorted(candidates, key=lambda item: (item.source, item.target))
    # The caller constructs at most one candidate per pair.  Keep this assertion
    # close to packing so a future candidate source cannot silently create duplicate
    # CSR destinations.
    seen: set[tuple[int, int]] = set()
    rows: list[list[_Candidate]] = [[] for _ in range(n_stops)]
    for candidate in ordered:
        pair = (candidate.source, candidate.target)
        if pair in seen:
            raise ValueError(f"duplicate graph transfer pair: {pair}")
        seen.add(pair)
        rows[candidate.source].append(candidate)

    flat = [candidate for row in rows for candidate in row]
    forward = _pack_direction(rows)
    reverse_rows: list[list[_Candidate]] = [[] for _ in range(n_stops)]
    for candidate in flat:
        reverse_rows[candidate.target].append(candidate)
    reverse = _pack_direction(reverse_rows, reverse=True)
    tr_off, tr_to, tr_walk, tr_min, tr_time, tr_fallback = forward
    reverse_off, reverse_to, reverse_walk, reverse_min, reverse_time, reverse_fallback = reverse

    path_off = np.zeros(len(flat) + 1, dtype=np.int64)
    path_off[1:] = np.cumsum([len(item.points) for item in flat], dtype=np.int64)
    path_points = np.asarray(
        [point for item in flat for point in item.points],
        dtype=np.float64,
    ).reshape(-1, 2)
    return GraphTransferBake(
        tr_off=_readonly(tr_off),
        tr_to=_readonly(tr_to),
        tr_time=_readonly(tr_time),
        tr_walk_time=_readonly(tr_walk),
        tr_min_time=_readonly(tr_min),
        tr_path_off=_readonly(path_off),
        tr_path_points=_readonly(path_points),
        tr_path_fallback=_readonly(tr_fallback),
        tr_pathway_metadata=tuple(item.pathway_metadata for item in flat),
        tr_reverse_off=_readonly(reverse_off),
        tr_reverse_to=_readonly(reverse_to),
        tr_reverse_time=_readonly(reverse_time),
        tr_reverse_walk_time=_readonly(reverse_walk),
        tr_reverse_min_time=_readonly(reverse_min),
        tr_reverse_path_fallback=_readonly(reverse_fallback),
        scoped_rules=scoped_rules,
        pathway_records=pathway_records,
    )


def validate_transfer_views(
    n_stops: int,
    *,
    forward_off: np.ndarray,
    forward_to: np.ndarray,
    forward_walk: np.ndarray,
    forward_min: np.ndarray,
    forward_time: np.ndarray,
    forward_fallback: np.ndarray,
    reverse_off: np.ndarray,
    reverse_to: np.ndarray,
    reverse_walk: np.ndarray,
    reverse_min: np.ndarray,
    reverse_time: np.ndarray,
    reverse_fallback: np.ndarray,
    forward_pathway_off: np.ndarray | None = None,
    forward_pathway_time: np.ndarray | None = None,
) -> bool:
    """Prove that the persisted forward and reverse transfer views are identical.

    The two CSR views intentionally have different row orientation, but they must contain
    exactly one record for the same directed edge.  Comparing by ``(source, target)`` rather
    than relying on flat-array order catches reversed, missing, duplicated, and mismatched
    rows in an artifact.  Pathway metadata is attached to forward edges; a timed pathway must
    agree with the stored display fallback bit (untimed metadata never creates one).
    """
    try:
        n_stops = int(n_stops)
        if n_stops < 0:
            return False
        views = (
            (forward_off, forward_to, forward_walk, forward_min, forward_time, forward_fallback),
            (reverse_off, reverse_to, reverse_walk, reverse_min, reverse_time, reverse_fallback),
        )
        edge_maps: list[dict[tuple[int, int], int]] = []
        for is_reverse, (off, to, walk, minimum, effective, fallback) in enumerate(views):
            off = np.asarray(off); to = np.asarray(to); walk = np.asarray(walk)
            minimum = np.asarray(minimum); effective = np.asarray(effective)
            fallback = np.asarray(fallback)
            if (off.ndim != 1 or off.dtype.kind not in "iu" or len(off) != n_stops + 1
                    or off[0] != 0 or np.any(np.diff(off) < 0)
                    or int(off[-1]) != len(to)
                    or to.ndim != 1 or to.dtype.kind not in "iu"
                    or np.any(to < 0) or np.any(to >= n_stops)):
                return False
            edge_count = len(to)
            if any(a.ndim != 1 or len(a) != edge_count
                   for a in (walk, minimum, effective, fallback)):
                return False
            if (walk.dtype.kind not in "fiu" or minimum.dtype.kind not in "fiu"
                    or effective.dtype.kind not in "iu" or fallback.dtype.kind not in "biu"
                    or not np.isfinite(walk).all() or not np.isfinite(minimum).all()
                    or np.any(walk < 0) or np.any(minimum < 0) or np.any(effective < 0)
                    or not np.array_equal(
                        effective,
                        np.floor(np.maximum(walk, minimum) + 0.5).astype(effective.dtype),
                    )
                    or np.any((fallback != 0) & (fallback != 1))):
                return False
            edge_map: dict[tuple[int, int], int] = {}
            for source in range(n_stops):
                start, end = int(off[source]), int(off[source + 1])
                row = to[start:end]
                if len(row) > 1 and np.any(np.diff(row) <= 0):
                    return False
                for index in range(start, end):
                    # Reverse rows store target -> source.  Normalize both views to the
                    # same directed source -> target key before comparing records.
                    key = (source, int(to[index]))
                    if is_reverse:
                        key = (int(to[index]), source)
                    if key in edge_map:
                        return False
                    edge_map[key] = index
            edge_maps.append(edge_map)
        forward_map, reverse_map = edge_maps
        if set(forward_map) != set(reverse_map) or len(forward_map) != len(reverse_map):
            return False
        for key, forward_index in forward_map.items():
            reverse_index = reverse_map[key]
            f_values = (
                np.asarray(forward_walk)[forward_index],
                np.asarray(forward_min)[forward_index],
                np.asarray(forward_time)[forward_index],
                np.asarray(forward_fallback)[forward_index],
            )
            r_values = (
                np.asarray(reverse_walk)[reverse_index],
                np.asarray(reverse_min)[reverse_index],
                np.asarray(reverse_time)[reverse_index],
                np.asarray(reverse_fallback)[reverse_index],
            )
            if not (f_values[0] == r_values[0] and f_values[1] == r_values[1]
                    and f_values[2] == r_values[2] and f_values[3] == r_values[3]):
                return False
        if (forward_pathway_off is not None) != (forward_pathway_time is not None):
            return False
        if forward_pathway_off is not None:
            pathway_off = np.asarray(forward_pathway_off)
            pathway_time = np.asarray(forward_pathway_time)
            if (pathway_off.ndim != 1 or pathway_off.dtype.kind not in "iu"
                    or len(pathway_off) != len(np.asarray(forward_to)) + 1
                    or pathway_off[0] != 0 or np.any(np.diff(pathway_off) < 0)
                    or int(pathway_off[-1]) != len(pathway_time)
                    or pathway_time.ndim != 1 or pathway_time.dtype.kind not in "iu"
                    or np.any(pathway_time < -1)):
                return False
            for key, forward_index in forward_map.items():
                start, end = int(pathway_off[forward_index]), int(pathway_off[forward_index + 1])
                timed = pathway_time[start:end] >= 0
                if bool(timed.any()) != bool(np.asarray(forward_fallback)[forward_index]):
                    return False
                if timed.any():
                    # The bake chooses the shortest timed pathway as the authoritative
                    # physical duration.  A changed pathway row must not silently leave
                    # the CSR timing/pathway metadata disagreeing.
                    shortest = int(np.min(pathway_time[start:end][timed]))
                    physical = float(np.asarray(forward_walk)[forward_index])
                    if not math.isclose(physical, float(shortest), rel_tol=0.0, abs_tol=1e-9):
                        return False
        return True
    except (IndexError, TypeError, ValueError, OverflowError):
        return False


__all__ = [
    "DEFAULT_CAP_REF_SEC",
    "DEFAULT_RADIUS_M",
    "MAX_CANDIDATE_RADIUS_M",
    "GraphTransferBake",
    "PathwayTransferRecord",
    "ScopedTransferRecord",
    "bake_graph_transfers",
    "validate_transfer_views",
]
