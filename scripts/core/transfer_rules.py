"""Parse the optional GTFS transfer and station-pathway tables.

This module deliberately stops at a small, immutable intermediate representation.
The graph-backed access bake consumes it to calculate directed street-transfer
edges.  It does not calculate street paths, import pandas, or create cache
artifacts.

GTFS transfer rows have two different kinds of meaning that must not be conflated:

* An unscoped ``transfer_type=3`` is a prohibition for the stop pair.
* A prohibition scoped to routes or trips is retained as a scoped rule.  It is
  *never* promoted to a global prohibition because doing so would remove valid
  transfers for other route combinations.

Route/trip-scoped minimum times and prohibitions are retained with their complete
scope.  The graph bake preserves any defensible physical route/time as metadata,
but never promotes the rule to global RAPTOR: a future route-aware kernel must
apply it with context.  This is important while the current single-label RAPTOR
state cannot always recover both sides of a transfer.

GTFS transfer types 4 and 5 are linked-trip/in-seat semantics, not global physical
transfer rules. They are retained only when both trip IDs are present; malformed
unscoped rows are ignored.

Stop IDs are namespaced with the feed name.  Route and trip IDs are kept as their
feed-local strings; callers must use the enclosing rule set's feed when matching
those optional scopes.
"""

from __future__ import annotations

import csv
import io
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping, NamedTuple
import zipfile


class StopKey(NamedTuple):
    """A feed-namespaced GTFS stop identifier."""

    feed: str
    stop_id: str

    def __str__(self) -> str:
        return f"{self.feed}:{self.stop_id}"


StopPair = tuple[StopKey, StopKey]


@dataclass(frozen=True)
class TransferRule:
    """One explicit stop-pair rule.

    ``min_transfer_seconds`` is a lower bound, not an estimated street-walk
    duration.  A prohibited rule has ``prohibited=True``.  ``source`` is a stable
    diagnostic label (currently always ``transfers``).
    """

    from_stop: StopKey
    to_stop: StopKey
    min_transfer_seconds: int = 0
    prohibited: bool = False
    source: str = "transfers"

    @property
    def pair(self) -> StopPair:
        return self.from_stop, self.to_stop


@dataclass(frozen=True)
class ScopedTransferRule:
    """A rule whose application requires route/trip context.

    Trip IDs are retained for diagnostics and future route-aware consumers, but
    this module does not try to infer route labels from trips.  The current RAPTOR
    single-label state may not have enough context to apply these rules safely; that
    limitation belongs to the consumer and is not "solved" by globalizing them.
    """

    from_stop: StopKey
    to_stop: StopKey
    from_route_id: str | None = None
    to_route_id: str | None = None
    from_trip_id: str | None = None
    to_trip_id: str | None = None
    min_transfer_seconds: int | None = None
    prohibited: bool = False
    source: str = "transfers"
    transfer_type: str | None = None

    @property
    def pair(self) -> StopPair:
        return self.from_stop, self.to_stop


@dataclass(frozen=True)
class PathwayEdge:
    """One directed station-pathway edge from ``pathways.txt``."""

    pathway_id: str
    from_stop: StopKey
    to_stop: StopKey
    traversal_seconds: int | None
    pathway_mode: str | None = None
    length_meters: float | None = None
    reversed_from_bidirectional: bool = False

    @property
    def pair(self) -> StopPair:
        return self.from_stop, self.to_stop


@dataclass(frozen=True)
class TransferRuleSet:
    """Deterministic, immutable parser output for one GTFS feed.

    ``unconditional_rules`` contains one rule per directed stop pair, with the
    strongest fixed minimum time found in ``transfers.txt``.  A global prohibition
    wins over any minimum for that pair.  ``route_scoped_rules`` retains scoped
    minimums and prohibitions; callers must apply those with route/trip context.
    ``pathway_edges`` retains the explicit pathway rows and directionality for
    later graph baking.  Pathway traversal duration is deliberately not folded
    into ``min_transfer_seconds``: it is the physical edge cost and adding it to
    the fixed transfer constraint would double-count station traversal.
    """

    feed: str
    unconditional_rules: tuple[TransferRule, ...]
    route_scoped_rules: tuple[ScopedTransferRule, ...]
    pathway_edges: tuple[PathwayEdge, ...]
    min_transfer_seconds: Mapping[StopPair, int]
    prohibited_pairs: frozenset[StopPair]

    @property
    def rules(self) -> tuple[TransferRule, ...]:
        """Alias for consumers that want the explicit global stop-pair rules."""

        return self.unconditional_rules

    @property
    def scoped_prohibitions(self) -> tuple[ScopedTransferRule, ...]:
        """The subset of route/trip-scoped rules that prohibit a transfer."""

        return tuple(rule for rule in self.route_scoped_rules if rule.prohibited)


_TRANSFER_TYPES = frozenset({"0", "1", "2", "3", "4", "5"})


def _clean(value: object) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _integer(value: object) -> int | None:
    """Parse a GTFS integer, rejecting malformed and negative values."""

    value = _clean(value)
    if value is None:
        return None
    try:
        number = int(value, 10)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        return number if math.isfinite(number) and number >= 0 else None
    except OverflowError:
        return None


def _nonnegative_float(value: object) -> float | None:
    value = _clean(value)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _flag(value: object) -> bool | None:
    value = _clean(value)
    if value == "0":
        return False
    if value == "1":
        return True
    return None


def _norm_row(row: Mapping[str | None, object]) -> dict[str, object]:
    # utf-8-sig handles the BOM in the first header, while this also tolerates
    # harmless whitespace around headers from hand-authored synthetic feeds.
    return {str(k).strip().lstrip("\ufeff"): v for k, v in row.items() if k is not None}


@contextmanager
def _open_feed(feed_zip: os.PathLike[str] | str | zipfile.ZipFile):
    if isinstance(feed_zip, zipfile.ZipFile):
        yield feed_zip, None
        return
    z = zipfile.ZipFile(feed_zip)
    try:
        yield z, z
    finally:
        z.close()


def _feed_name(feed_zip: os.PathLike[str] | str | zipfile.ZipFile,
               feed_name: str | None) -> str:
    if feed_name is not None:
        value = _clean(feed_name)
        if value:
            return value
    if isinstance(feed_zip, zipfile.ZipFile):
        filename = feed_zip.filename or "feed"
    else:
        filename = os.fspath(feed_zip)
    return os.path.splitext(os.path.basename(filename))[0] or "feed"


def _rows(z: zipfile.ZipFile, filename: str) -> Iterator[dict[str, object]]:
    if filename not in z.namelist():
        return
    with z.open(filename, "r") as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        try:
            reader = csv.DictReader(text)
            if reader.fieldnames is None:
                return
            for row in reader:
                yield _norm_row(row)
        finally:
            text.detach()


def _stop(feed: str, row: Mapping[str, object], field: str) -> StopKey | None:
    value = _clean(row.get(field))
    return StopKey(feed, value) if value is not None else None


def parse_transfer_rules(
    feed_zip: os.PathLike[str] | str | zipfile.ZipFile,
    feed_name: str | None = None,
) -> TransferRuleSet:
    """Parse ``transfers.txt`` and ``pathways.txt`` from one GTFS ZIP.

    Missing optional files are valid and produce an empty rule set.  Individual
    malformed rows are ignored rather than making an otherwise usable feed
    unusable.  Duplicate global and same-scope minimums resolve by maximum;
    duplicate prohibitions remain one prohibition for their complete scope key.
    All output is sorted and immutable for repeatable cache builds.
    """

    feed = _feed_name(feed_zip, feed_name)
    pair_mins: dict[StopPair, int] = {}
    prohibited: set[StopPair] = set()
    scoped: dict[tuple[object, ...], ScopedTransferRule] = {}
    pathway_edges: dict[tuple[str, StopPair, bool], PathwayEdge] = {}

    with _open_feed(feed_zip) as (z, _owned):
        for row in _rows(z, "transfers.txt"):
            from_stop = _stop(feed, row, "from_stop_id")
            to_stop = _stop(feed, row, "to_stop_id")
            transfer_type = _clean(row.get("transfer_type"))
            if from_stop is None or to_stop is None or from_stop == to_stop:
                continue
            if transfer_type not in _TRANSFER_TYPES:
                continue

            from_route = _clean(row.get("from_route_id"))
            to_route = _clean(row.get("to_route_id"))
            from_trip = _clean(row.get("from_trip_id"))
            to_trip = _clean(row.get("to_trip_id"))
            has_scope = any(v is not None for v in (from_route, to_route, from_trip, to_trip))
            pair = (from_stop, to_stop)

            if transfer_type == "3":
                if has_scope:
                    key = (pair, from_route, to_route, from_trip, to_trip)
                    old = scoped.get(key)
                    scoped[key] = ScopedTransferRule(
                        from_stop=from_stop,
                        to_stop=to_stop,
                        from_route_id=from_route,
                        to_route_id=to_route,
                        from_trip_id=from_trip,
                        to_trip_id=to_trip,
                        min_transfer_seconds=(old.min_transfer_seconds if old else None),
                        prohibited=True,
                    )
                else:
                    prohibited.add(pair)
                continue

            # Types 4/5 describe linked-trip in-seat behavior. They are not a
            # station prohibition or walking minimum. Without both trip IDs the
            # row has no safe scope and is ignored rather than globalized.
            if transfer_type in {"4", "5"}:
                if from_trip is None or to_trip is None:
                    continue
                key = (pair, from_route, to_route, from_trip, to_trip, transfer_type)
                scoped[key] = ScopedTransferRule(
                    from_stop=from_stop,
                    to_stop=to_stop,
                    from_route_id=from_route,
                    to_route_id=to_route,
                    from_trip_id=from_trip,
                    to_trip_id=to_trip,
                    transfer_type=transfer_type,
                )
                continue

            # GTFS gives min_transfer_time meaning for type 2 and allows it on
            # other transfer rows.  Applying every valid non-prohibited value is
            # conservative and avoids silently discarding a feed's timing rule.
            seconds = _integer(row.get("min_transfer_time"))
            if seconds is not None:
                if has_scope:
                    key = (pair, from_route, to_route, from_trip, to_trip)
                    old = scoped.get(key)
                    scoped[key] = ScopedTransferRule(
                        from_stop=from_stop,
                        to_stop=to_stop,
                        from_route_id=from_route,
                        to_route_id=to_route,
                        from_trip_id=from_trip,
                        to_trip_id=to_trip,
                        min_transfer_seconds=max(
                            old.min_transfer_seconds if old and old.min_transfer_seconds is not None else 0,
                            seconds,
                        ),
                        prohibited=old.prohibited if old else False,
                    )
                else:
                    pair_mins[pair] = max(pair_mins.get(pair, 0), seconds)

        for row in _rows(z, "pathways.txt"):
            pathway_id = _clean(row.get("pathway_id"))
            from_stop = _stop(feed, row, "from_stop_id")
            to_stop = _stop(feed, row, "to_stop_id")
            seconds = _integer(row.get("traversal_time"))
            length = _nonnegative_float(row.get("length"))
            bidirectional = _flag(row.get("is_bidirectional"))
            if (pathway_id is None or from_stop is None or to_stop is None
                    or from_stop == to_stop or bidirectional is None):
                continue
            mode = _clean(row.get("pathway_mode"))
            edge = PathwayEdge(pathway_id, from_stop, to_stop, seconds, mode, length, False)
            key = (pathway_id, edge.pair, False)
            # If a duplicate pathway row is present, the shortest valid traversal
            # wins.  Equal-time ties use the lexicographically smallest mode so
            # output does not depend on ZIP row order.
            old = pathway_edges.get(key)
            if old is None or _pathway_candidate_key(edge) < _pathway_candidate_key(old):
                pathway_edges[key] = edge

            if bidirectional:
                reverse = PathwayEdge(pathway_id, to_stop, from_stop, seconds, mode, length, True)
                rkey = (pathway_id, reverse.pair, True)
                old = pathway_edges.get(rkey)
                if old is None or _pathway_candidate_key(reverse) < _pathway_candidate_key(old):
                    pathway_edges[rkey] = reverse
                # The reverse edge carries its own physical traversal duration;
                # neither direction is a fixed transfer minimum.

    rules: list[TransferRule] = []
    for pair in sorted(set(pair_mins) | prohibited, key=_pair_sort_key):
        rules.append(TransferRule(
            from_stop=pair[0],
            to_stop=pair[1],
            min_transfer_seconds=pair_mins.get(pair, 0),
            prohibited=pair in prohibited,
            source="transfers",
        ))

    scoped_rules = tuple(sorted(scoped.values(), key=_scoped_sort_key))
    edges = tuple(sorted(pathway_edges.values(), key=_pathway_sort_key))
    min_times = MappingProxyType({rule.pair: rule.min_transfer_seconds
                                  for rule in rules if rule.pair in pair_mins})
    return TransferRuleSet(
        feed=feed,
        unconditional_rules=tuple(rules),
        route_scoped_rules=scoped_rules,
        pathway_edges=edges,
        min_transfer_seconds=min_times,
        prohibited_pairs=frozenset(prohibited),
    )


def _stop_sort_key(stop: StopKey) -> tuple[str, str]:
    return stop.feed, stop.stop_id


def _pair_sort_key(pair: StopPair) -> tuple[tuple[str, str], tuple[str, str]]:
    return _stop_sort_key(pair[0]), _stop_sort_key(pair[1])


def _scoped_sort_key(rule: ScopedTransferRule):
    return (_pair_sort_key(rule.pair), rule.from_route_id or "", rule.to_route_id or "",
            rule.from_trip_id or "", rule.to_trip_id or "", rule.transfer_type or "")


def _pathway_sort_key(edge: PathwayEdge):
    return (_pair_sort_key(edge.pair), edge.pathway_id, edge.reversed_from_bidirectional)


def _pathway_candidate_key(edge: PathwayEdge):
    return (
        edge.traversal_seconds is None,
        edge.traversal_seconds if edge.traversal_seconds is not None else 0,
        edge.length_meters is None,
        edge.length_meters if edge.length_meters is not None else 0.0,
        edge.pathway_mode or "",
    )


def parse_transfer_rules_many(
    feeds: Iterable[os.PathLike[str] | str | zipfile.ZipFile],
) -> tuple[TransferRuleSet, ...]:
    """Parse multiple feeds independently, preserving each feed namespace."""

    return tuple(parse_transfer_rules(feed) for feed in feeds)


__all__ = [
    "PathwayEdge",
    "ScopedTransferRule",
    "StopKey",
    "StopPair",
    "TransferRule",
    "TransferRuleSet",
    "parse_transfer_rules",
    "parse_transfer_rules_many",
]
