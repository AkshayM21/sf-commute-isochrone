import os
import re
import sys
import math
import random
from pathlib import Path

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from core import server_raptor
from core.raptor_journey import _min_overshoot_alight
from core.raptor_journey_da import DepartAfterJourneyTree
from core.server_raptor import _select_diverse_alts


_HEAD = [[37.700, -122.450], [37.705, -122.440], [37.710, -122.430]]
_HEAD_REVERSE = [[37.700, -122.450], [37.695, -122.460], [37.690, -122.470]]
_MID = [[37.710, -122.430], [37.715, -122.420], [37.720, -122.410]]
_TAIL = [[37.720, -122.410], [37.725, -122.405], [37.730, -122.400]]
_TAIL_EARLY_BOARD = [[37.710, -122.430], [37.720, -122.410],
                     [37.725, -122.405], [37.730, -122.400]]
_OTHER_TAIL = [[37.720, -122.410], [37.730, -122.415], [37.740, -122.420]]


def _leg(name, pts, feed="fixture", mode="bus", minutes=8, route_id=None):
    leg = {"mode": "transit", "name": name, "feed": feed, "tmode": mode,
           "min": minutes, "pts": pts}
    if route_id is not None:
        leg["route_id"] = route_id
    return leg


def _route(names, total, paths=None, feeds=None, modes=None, route_ids=None):
    paths = paths or [_HEAD] + [_TAIL] * (len(names) - 1)
    feeds = feeds or ["fixture"] * len(names)
    modes = modes or ["bus"] * len(names)
    route_ids = route_ids or [None] * len(names)
    legs = [_leg(name, pts, feed, mode, route_id=route_id)
            for name, pts, feed, mode, route_id in zip(
                names, paths, feeds, modes, route_ids)]
    line = " > ".join(names)
    return {
        "line": line,
        "typical": {"total": total, "legs": legs},
        "best": {"total": total, "legs": legs},
    }


def test_same_branch_slower_simpler_route_does_not_prune_faster_tradeoff():
    out = _select_diverse_alts([
        _route(["A", "X", "T"], 26, [_HEAD, _MID, _TAIL]),
        _route(["A"], 23),
        _route(["A", "T"], 27, [_HEAD, _TAIL]),
        _route(["B", "U"], 24, [_MID, _OTHER_TAIL]),
    ], cap=6)
    labels = [a["line"] for a in out]

    assert "A" in labels
    assert "A > T" in labels
    assert "A > X > T" in labels


def test_same_branch_pareto_dominance_prunes_only_no_worse_route():
    out = _select_diverse_alts([
        _route(["A", "T"], 24, [_HEAD, _TAIL]),
        _route(["A", "X", "T"], 26, [_HEAD, _MID, _TAIL]),
    ], cap=3)

    assert [option["line"] for option in out] == ["A > T"]


def test_generic_dominance_keeps_materially_faster_detour():
    out = _select_diverse_alts([
        _route(["A", "X", "T"], 21, [_HEAD, _MID, _TAIL]),
        _route(["A"], 22),
        _route(["A", "T"], 27, [_HEAD, _TAIL]),
        _route(["B", "U"], 28, [_MID, _OTHER_TAIL]),
    ], cap=4)
    labels = [a["line"] for a in out]

    assert "A" in labels
    assert "A > T" in labels
    assert "A > X > T" in labels


def test_family_representative_never_uses_simplicity_to_outrank_faster_branch():
    out = _select_diverse_alts([
        _route(["A"], 22, [_HEAD]),
        _route(["A", "T"], 21, [_HEAD, _TAIL]),
    ], cap=1)

    assert [option["line"] for option in out] == ["A > T"]


def test_generic_dominance_keeps_indirect_tail_when_no_direct_version_exists():
    out = _select_diverse_alts([
        _route(["A", "X", "T"], 22, [_HEAD, _MID, _TAIL]),
        _route(["A"], 23),
        _route(["B", "U"], 24, [_MID, _OTHER_TAIL]),
    ], cap=4)
    labels = [a["line"] for a in out]

    assert "A" in labels
    assert "A > X > T" in labels


def test_indexed_dominance_matches_pairwise_reference_on_adversarial_routes():
    """The latency index may narrow comparisons only by necessary equivalence predicates."""
    def pairwise_reference(alts, context, family_keys):
        context = list(context or ())
        alts = list(alts)

        def rank(option):
            return server_raptor._alt_quality_rank(option)

        groups = []
        groups_by_bucket = {}
        for alt in sorted(alts, key=rank):
            bucket = server_raptor._alt_choice_bucket(alt)
            bucket_groups = groups_by_bucket.setdefault(bucket, [])
            compatible = [group for group in bucket_groups if all(
                server_raptor._alt_choice_equivalent(alt, member) for member in group)]
            if compatible:
                compatible[0].append(alt)
            else:
                group = [alt]
                groups.append(group)
                bucket_groups.append(group)

        deduped = []
        for options in groups:
            winner = min(options, key=rank)
            winner_bucket = server_raptor._alt_choice_bucket(winner)
            primary_matches = [option for option in context
                               if server_raptor._alt_choice_bucket(option) == winner_bucket
                               if server_raptor._alt_choice_equivalent(option, winner)]
            if (primary_matches
                    and min(server_raptor._alt_total(option)
                            for option in primary_matches)
                    <= server_raptor._alt_total(winner)):
                continue
            deduped.append(winner)

        comparisons = context + deduped
        return [option for option in deduped if not any(
            candidate is not option
            and server_raptor._alt_dominates(candidate, option, family_keys)
            for candidate in comparisons)]

    rng = random.Random(20260718)
    heads = [
        _HEAD,
        _HEAD_REVERSE,
        [[37.700, -122.450], [37.708, -122.443], [37.714, -122.435]],
        [[37.700, -122.450], [37.693, -122.442], [37.686, -122.435]],
    ]
    tails = [_TAIL, _OTHER_TAIL, _TAIL_EARLY_BOARD]
    names = ["ALPHA", "BETA", "GAMMA", "DELTA"]
    for _ in range(40):
        options = []
        for _j in range(rng.randint(8, 28)):
            count = rng.randint(1, 4)
            sequence = [rng.choice(names) for _k in range(count)]
            paths = [rng.choice(heads)] + [rng.choice(tails) for _k in range(count - 1)]
            route = _route(sequence, rng.randint(17, 38), paths)
            if rng.random() < 0.35:
                walk = {"mode": "walk", "min": rng.randint(1, 12), "pts": []}
                route["typical"]["legs"].insert(0, dict(walk))
                route["best"]["legs"].insert(0, dict(walk))
            options.append(route)
        context = options[:1] if rng.random() < 0.7 else []
        alts = options[1:] if context else options
        family_keys = server_raptor._discover_family_keys(context + alts)

        expected = pairwise_reference(alts, context, family_keys)
        actual = server_raptor._prune_dominated_alts(alts, context, family_keys)

        assert [id(option) for option in actual] == [id(option) for option in expected]


def test_distinct_family_gets_slot_before_duplicate_branch():
    out = _select_diverse_alts([
        _route(["A"], 20),
        _route(["A", "T"], 21, [_HEAD, _TAIL]),
        _route(["B"], 22, [_MID]),
    ], cap=2)
    labels = [a["line"] for a in out]

    assert labels == ["A", "B"]


def test_arbitrary_names_share_family_when_boarding_corridor_and_direction_match():
    a = _route(["ALPHA"], 20, [_HEAD])
    b = _route(["BETA"], 21, [_HEAD])
    keys = server_raptor._discover_family_keys([a, b])

    assert server_raptor._alt_family_key(a, keys) == server_raptor._alt_family_key(b, keys)


def test_partial_shared_prefix_with_different_stop_spacing_is_one_family():
    # The boarding points deliberately straddle the old 0.001-degree rounding boundary.
    sparse = [[37.69949, -122.450], [37.706, -122.439], [37.714, -122.425]]
    frequent = [[37.69951, -122.4498], [37.703, -122.444], [37.709, -122.432]]
    a = _route(["ALPHA"], 20, [sparse])
    b = _route(["BETA"], 21, [frequent])
    keys = server_raptor._discover_family_keys([a, b])

    assert server_raptor._alt_family_key(a, keys) == server_raptor._alt_family_key(b, keys)


def test_tiny_common_launch_then_divergence_is_not_a_shared_corridor():
    north = [[37.700, -122.450], [37.700, -122.4495], [37.710, -122.4495]]
    south = [[37.700, -122.450], [37.700, -122.4495], [37.690, -122.4495]]
    a = _route(["ALPHA"], 20, [north])
    b = _route(["BETA"], 21, [south])
    keys = server_raptor._discover_family_keys([a, b])

    assert server_raptor._alt_family_key(a, keys) != server_raptor._alt_family_key(b, keys)


def test_same_name_opposite_directions_are_distinct_families():
    forward = _route(["ALPHA"], 20, [_HEAD])
    reverse = _route(["ALPHA"], 21, [_HEAD_REVERSE])
    keys = server_raptor._discover_family_keys([forward, reverse])

    assert server_raptor._alt_family_key(forward, keys) != server_raptor._alt_family_key(reverse, keys)


def test_same_service_nearby_route_bend_remains_one_family():
    east = [[37.700, -122.450], [37.700, -122.440], [37.700, -122.430]]
    north = [[37.701, -122.450], [37.711, -122.450], [37.721, -122.450]]
    before_bend = _route(["ALPHA"], 20, [east])
    after_bend = _route(["ALPHA", "T"], 21, [north, _TAIL])
    keys = server_raptor._discover_family_keys([before_bend, after_bend])

    assert server_raptor._alt_family_key(before_bend, keys) == server_raptor._alt_family_key(
        after_bend, keys)


def test_complete_link_clustering_cannot_chain_east_through_north_to_west():
    east = _route(["ALPHA"], 20, [[[37.700, -122.450], [37.700, -122.440]]])
    north = _route(["ALPHA"], 21, [[[37.700, -122.450], [37.710, -122.450]]])
    west = _route(["ALPHA"], 22, [[[37.700, -122.450], [37.700, -122.460]]])
    keys = server_raptor._discover_family_keys([east, north, west])

    assert len(set(keys.values())) == 2
    assert server_raptor._alt_family_key(east, keys) != server_raptor._alt_family_key(west, keys)


def test_incompatible_components_cannot_collide_on_coarse_min_descriptor(monkeypatch):
    left_path = [[37.70001, -122.45001], [37.71001, -122.44001]]
    right_path = [[37.70002, -122.45002], [37.71002, -122.44002]]
    left = _route(["ALPHA"], 20, [left_path])
    right = _route(["BETA"], 21, [right_path])
    left_leg = server_raptor._alt_transit_legs(left)[0]
    right_leg = server_raptor._alt_transit_legs(right)[0]
    assert server_raptor._leg_corridor_sig(left_leg) == server_raptor._leg_corridor_sig(right_leg)
    assert server_raptor._leg_geom_sig(left_leg) == server_raptor._leg_geom_sig(right_leg)
    monkeypatch.setattr(server_raptor, "_same_boarding_corridor", lambda _a, _b: False)

    keys = server_raptor._discover_family_keys([left, right])

    assert server_raptor._alt_family_key(left, keys) != server_raptor._alt_family_key(right, keys)


def test_identical_sparse_component_digests_get_unique_stable_suffixes():
    left = _route(["ALPHA"], 20, [[]])
    right = _route(["ALPHA"], 21, [[]])
    forward = server_raptor._discover_family_keys([left, right])
    reverse = server_raptor._discover_family_keys([right, left])

    assert len(set(forward.values())) == 2
    assert server_raptor._alt_family_key(left, forward) == server_raptor._alt_family_key(left, reverse)
    assert server_raptor._alt_family_key(right, forward) == server_raptor._alt_family_key(right, reverse)

    # With no structural or ranking distinction at all, ordinal ownership is intentionally
    # interchangeable, but conservative separation and the emitted key set remain deterministic.
    clone_a = _route(["BETA"], 22, [[]])
    clone_b = _route(["BETA"], 22, [[]])
    clone_forward = server_raptor._discover_family_keys([clone_a, clone_b])
    clone_reverse = server_raptor._discover_family_keys([clone_b, clone_a])
    assert len(set(clone_forward.values())) == 2
    assert set(clone_forward.values()) == set(clone_reverse.values())


def test_same_service_bend_tolerances_have_explicit_inside_outside_edges():
    lat, lon = 37.700, -122.450
    base = _leg("ALPHA", [[lat, lon], [lat, lon + 0.010]])
    inside_board = _leg("ALPHA", [[lat + 299.0 / 111_320.0, lon],
                                  [lat + 299.0 / 111_320.0, lon + 0.010]])
    outside_board = _leg("ALPHA", [[lat + 301.0 / 111_320.0, lon],
                                   [lat + 301.0 / 111_320.0, lon + 0.010]])

    def angled(degrees):
        angle = math.radians(degrees)
        return _leg("ALPHA", [[lat, lon],
                              [lat + math.sin(angle) * 0.010,
                               lon + math.cos(angle) * 0.010 / math.cos(math.radians(lat))]])

    assert server_raptor._same_boarding_corridor(base, inside_board)
    assert not server_raptor._same_boarding_corridor(base, outside_board)
    assert server_raptor._same_boarding_corridor(base, angled(99.0))
    assert not server_raptor._same_boarding_corridor(base, angled(101.0))


def test_coincident_corridors_from_different_services_are_distinct_families():
    local = _route(["ALPHA"], 20, [_HEAD], feeds=["local"], modes=["bus"])
    regional = _route(["BETA"], 21, [_HEAD], feeds=["regional"], modes=["rail"])
    keys = server_raptor._discover_family_keys([local, regional])

    assert server_raptor._alt_family_key(local, keys) != server_raptor._alt_family_key(regional, keys)


def test_tail_branch_identity_ignores_intermediate_feeder_but_not_final_tail():
    direct = _route(["A", "T"], 20, [_HEAD, _TAIL])
    detour = _route(["A", "X", "T"], 21, [_HEAD, _MID, _TAIL])
    other = _route(["A", "U"], 22, [_HEAD, _OTHER_TAIL])
    fam = server_raptor._alt_family_key(direct)

    assert server_raptor._alt_branch_key(direct, fam) == server_raptor._alt_branch_key(detour, fam)
    assert server_raptor._alt_branch_key(direct, fam) != server_raptor._alt_branch_key(other, fam)


def test_tail_branch_identity_uses_arrival_approach_not_transfer_boarding_stop():
    direct = _route(["A", "T"], 23, [_HEAD, _TAIL_EARLY_BOARD])
    detour = _route(["A", "X", "T"], 24, [_HEAD, _MID, _TAIL])
    fam_keys = server_raptor._discover_family_keys([direct, detour])
    fam = server_raptor._alt_family_key(direct, fam_keys)

    assert server_raptor._alt_branch_key(direct, fam) == server_raptor._alt_branch_key(detour, fam)
    assert [a["line"] for a in _select_diverse_alts([detour, direct], cap=2)] == ["A > T"]


def test_one_seat_walk_branch_uses_terminal_approach_and_dedupes_boarding_variants():
    end = [37.775, -122.419]
    approach = [37.770, -122.424]
    near = _route(["F"], 15, [[[37.765, -122.431], approach, end]])
    far = _route(["F"], 15, [[[37.769, -122.427], approach, end]])
    near["typical"]["legs"].insert(0, {"mode": "walk", "min": 5})
    near["best"]["legs"].insert(0, {"mode": "walk", "min": 5})
    far["typical"]["legs"].insert(0, {"mode": "walk", "min": 7})
    far["best"]["legs"].insert(0, {"mode": "walk", "min": 7})
    family_keys = server_raptor._discover_family_keys([near, far])
    assert server_raptor._alt_family_key(near, family_keys) != server_raptor._alt_family_key(
        far, family_keys)
    assert server_raptor._alt_branch_key(near) == server_raptor._alt_branch_key(far)
    assert server_raptor._alt_dedupe_key(near, family_keys) == server_raptor._alt_dedupe_key(
        far, family_keys)

    out = _select_diverse_alts([far, near], cap=2)

    assert out == [near]
    assert _select_diverse_alts([near, far], cap=2) == [near]
    assert _select_diverse_alts([far], cap=2, primary=near) == []
    assert _select_diverse_alts([near], cap=2, primary=far) == []


def test_one_seat_walk_branches_preserve_different_endpoints_and_arrival_directions():
    endpoint_a = _route(["F"], 15, [[[37.765, -122.431], [37.770, -122.424],
                                    [37.775, -122.419]]])
    endpoint_b = _route(["F"], 16, [[[37.769, -122.427], [37.772, -122.422],
                                    [37.778, -122.416]]])
    opposite = _route(["F"], 17, [[[37.769, -122.427], [37.780, -122.414],
                                  [37.775, -122.419]]])

    keys = {server_raptor._alt_branch_key(option) for option in (endpoint_a, endpoint_b, opposite)}
    assert len(keys) == 3
    assert len(_select_diverse_alts([endpoint_a, endpoint_b, opposite], cap=3)) == 3


def test_cross_family_same_named_sequence_remains_service_qualified():
    path = [[37.765, -122.431], [37.770, -122.424], [37.775, -122.419]]
    local = _route(["F"], 15, [path], feeds=["local"], modes=["bus"])
    regional = _route(["F"], 15, [path], feeds=["regional"], modes=["rail"])

    assert server_raptor._alt_dedupe_key(local) != server_raptor._alt_dedupe_key(regional)
    assert len(_select_diverse_alts([local, regional], cap=2)) == 2


def test_preselection_dedupe_preserves_same_sequence_with_distinct_tail_endpoints():
    near = _route(["A", "T"], 23, [_HEAD, _TAIL])
    alternate_endpoint = _route(["A", "T"], 24, [_HEAD, _OTHER_TAIL])

    assert server_raptor._alt_dedupe_key(near) != server_raptor._alt_dedupe_key(alternate_endpoint)


def test_preselection_dedupe_collapses_cross_bin_same_sequence_same_terminal():
    sparse_head = [[37.69949, -122.450], [37.706, -122.439], [37.714, -122.425]]
    frequent_head = [[37.69951, -122.4498], [37.703, -122.444], [37.709, -122.432]]
    left = _route(["A", "T"], 23, [sparse_head, _TAIL])
    right = _route(["A", "T"], 24, [frequent_head, _TAIL])
    family_keys = server_raptor._discover_family_keys([left, right])

    assert server_raptor._alt_family_key(left, family_keys) == server_raptor._alt_family_key(
        right, family_keys)
    assert server_raptor._alt_dedupe_key(left, family_keys) == server_raptor._alt_dedupe_key(
        right, family_keys)


def test_preselection_dedupe_keeps_opposite_boarding_directions():
    join = [37.710, -122.430]
    before_end = [37.720, -122.420]
    end = [37.730, -122.410]
    east_first = [[37.700, -122.450], [37.700, -122.440], join, before_end, end]
    west_first = [[37.700, -122.450], [37.700, -122.460], join, before_end, end]
    east = _route(["ALPHA"], 20, [east_first], route_ids=["route-a"])
    west = _route(["ALPHA"], 21, [west_first], route_ids=["route-a"])

    assert server_raptor._alt_branch_key(east) == server_raptor._alt_branch_key(west)
    assert server_raptor._alt_dedupe_key(east) != server_raptor._alt_dedupe_key(west)
    assert len(_select_diverse_alts([east, west], cap=2)) == 2


def test_preselection_direction_guard_ignores_later_same_direction_bends():
    shared_start = [[37.700, -122.450], [37.700, -122.440]]
    before_end = [37.720, -122.420]
    end = [37.730, -122.410]
    north_bend = shared_start + [[37.710, -122.440], before_end, end]
    south_bend = shared_start + [[37.690, -122.440], before_end, end]
    north = _route(["ALPHA"], 20, [north_bend], route_ids=["route-a"])
    south = _route(["ALPHA"], 21, [south_bend], route_ids=["route-a"])

    assert server_raptor._alt_branch_key(north) == server_raptor._alt_branch_key(south)
    assert server_raptor._alt_dedupe_key(north) == server_raptor._alt_dedupe_key(south)


def test_final_choice_dedupe_is_not_broken_by_a_heading_bucket_boundary():
    """Same ride/tail from two access stops is one option even across a quantizer edge."""
    shared_finish = [[37.723004, -122.444951], [37.723031, -122.446932]]
    earlier_board = [[37.719728, -122.428676], [37.720112, -122.429467], *shared_finish]
    later_board = [[37.723147, -122.435845], [37.723670, -122.438670], *shared_finish]
    primary = _route(["ALPHA"], 20, [earlier_board], route_ids=["route-a"])
    duplicate = _route(["ALPHA"], 24, [later_board], route_ids=["route-a"])

    # The old eight-way key put these headings into adjacent bins despite only an 18-degree bend.
    assert server_raptor._alt_dedupe_key(primary) != server_raptor._alt_dedupe_key(duplicate)
    assert server_raptor._alt_choice_equivalent(primary, duplicate)
    assert _select_diverse_alts([duplicate], cap=2, primary=primary) == []
    assert _select_diverse_alts([duplicate, primary], cap=2) == [primary]


def _with_access_trace(route, points, marker):
    walk = {"mode": "walk", "min": 5, "pts": points}
    transit = list(route["typical"]["legs"])
    route["typical"]["legs"] = [dict(walk), *transit]
    route["best"]["legs"] = [dict(walk), *transit]
    route["marker"] = marker
    return route


def test_equal_total_semantic_duplicate_always_defers_to_primary():
    primary = _with_access_trace(
        _route(["ALPHA"], 20, [_HEAD], route_ids=["route-a"]),
        [[37.900, -122.500], [37.901, -122.499]], "primary")
    # This trace sorts before the primary's and therefore won the old pairwise rank despite being
    # the identical two-slot route choice and total.
    duplicate = _with_access_trace(
        _route(["ALPHA"], 20, [_HEAD], route_ids=["route-a"]),
        [[37.100, -122.500], [37.101, -122.499]], "duplicate")

    assert _select_diverse_alts([duplicate], cap=1, primary=primary) == []


def test_genuinely_faster_equivalent_alt_survives_primary_preference():
    primary = _with_access_trace(
        _route(["ALPHA"], 20, [_HEAD], route_ids=["route-a"]),
        [[37.900, -122.500], [37.901, -122.499]], "primary")
    faster = _with_access_trace(
        _route(["ALPHA"], 19, [_HEAD], route_ids=["route-a"]),
        [[37.100, -122.500], [37.101, -122.499]], "faster")

    selected = _select_diverse_alts([faster], cap=1, primary=primary)

    assert selected == [faster]


def test_equivalent_alts_without_primary_keep_one_deterministic_winner():
    low_trace = _with_access_trace(
        _route(["ALPHA"], 20, [_HEAD], route_ids=["route-a"]),
        [[37.100, -122.500], [37.101, -122.499]], "low")
    high_trace = _with_access_trace(
        _route(["ALPHA"], 20, [_HEAD], route_ids=["route-a"]),
        [[37.900, -122.500], [37.901, -122.499]], "high")

    forward = _select_diverse_alts([high_trace, low_trace], cap=2)
    reverse = _select_diverse_alts([low_trace, high_trace], cap=2)

    assert len(forward) == len(reverse) == 1
    assert forward[0]["marker"] == reverse[0]["marker"] == "low"


def test_exact_equivalent_prefers_less_physical_access_not_schedule_allowance():
    short_access = _route(["ALPHA"], 20, [_HEAD], route_ids=["route-a"])
    long_access = _route(["ALPHA"], 20, [_HEAD], route_ids=["route-a"])
    for route, physical, allowance in ((short_access, 3, 6), (long_access, 7, 2)):
        walk = {"mode": "walk", "min": physical + allowance,
                "physical_min": physical, "schedule_allowance_min": allowance}
        route["typical"]["legs"].insert(0, dict(walk))
        route["best"]["legs"].insert(0, dict(walk))

    selected = _select_diverse_alts([long_access, short_access], cap=2)

    assert selected == [short_access]


def _with_physical_walks(route, *walks):
    """Attach physical walking legs for recommendation-rank fixtures.

    ``min`` deliberately includes a variable schedule allowance; recommendation must compare the
    explicit ``physical_min`` values instead.
    """
    route["typical"]["legs"] = [*walks, *route["typical"]["legs"]]
    route["best"]["legs"] = [dict(walk) for walk in walks] + route["best"]["legs"]
    return route


def test_recommendation_same_displayed_minute_prefers_less_total_physical_walking():
    map_route = _with_physical_walks(_route(["ALPHA"], 22, [_HEAD]),
        {"mode": "walk", "min": 8, "physical_min": 5},
        {"mode": "walk", "min": 8, "physical_min": 5})
    less_walk = _with_physical_walks(_route(["BETA"], 22, [_MID]),
        {"mode": "walk", "min": 4, "physical_min": 2},
        {"mode": "walk", "min": 3, "physical_min": 1})

    assert server_raptor._recommend_route_choice(map_route, [less_walk]) is less_walk


def test_recommendation_never_trades_an_extra_displayed_minute_for_less_walking():
    map_route = _with_physical_walks(_route(["ALPHA"], 22, [_HEAD]),
        {"mode": "walk", "min": 12, "physical_min": 12})
    quicker = _with_physical_walks(_route(["BETA"], 23, [_MID]),
        {"mode": "walk", "min": 1, "physical_min": 1})

    assert server_raptor._recommend_route_choice(map_route, [quicker]) is map_route


def test_recommendation_is_ranked_independently_for_each_displayed_time_mode():
    best_case_winner = _route(["ALPHA"], 24, [_HEAD])
    best_case_winner["best"]["total"] = 20
    realistic_winner = _route(["BETA"], 22, [_MID])
    realistic_winner["best"]["total"] = 21

    recommendations = server_raptor._recommend_route_choices(
        best_case_winner, [realistic_winner])

    assert recommendations["r"] is realistic_winner
    assert recommendations["b"] is best_case_winner


def test_recommendation_excludes_schedule_allowance_from_physical_walking():
    # The map route's display leg has ten minutes of extra allowable boarding time but only two
    # minutes of actual walking. It must beat a three-minute physical walk at the same displayed
    # commute minute.
    map_route = _with_physical_walks(_route(["ALPHA"], 22, [_HEAD]),
        {"mode": "walk", "min": 12, "physical_min": 2,
         "schedule_allowance_min": 10})
    more_actual_walking = _with_physical_walks(_route(["BETA"], 22, [_MID]),
        {"mode": "walk", "min": 3, "physical_min": 3})

    assert server_raptor._recommend_route_choice(map_route, [more_actual_walking]) is map_route


def test_recommendation_uses_bad_day_impact_after_time_walk_and_transfers():
    stable = _route(["ALPHA"], 22, [_HEAD])
    fragile = _route(["BETA"], 22, [_MID])
    stable["frag"] = 3
    fragile["frag"] = 9

    assert server_raptor._recommend_route_choice(fragile, [stable]) is stable


def test_recommendation_uses_exact_seconds_then_later_boarding_as_tie_breaks():
    first = _route(["ALPHA"], 22, [_HEAD])
    second = _route(["BETA"], 22, [_MID])
    first["_branch"] = {"metric_sec": 1_300,
                        "raw": [("ride", 0, 500, 900, 0, 1, 2)]}
    second["_branch"] = {"metric_sec": 1_290,
                         "raw": [("ride", 0, 520, 900, 0, 1, 2)]}
    assert server_raptor._recommend_route_choice(first, [second]) is second

    # With equal real duration, a later feasible board leaves the user less early dead time.
    second["_branch"]["metric_sec"] = 1_300
    assert server_raptor._recommend_route_choice(first, [second]) is second


def test_recommendation_is_force_retained_when_breadth_cap_would_omit_it():
    first = _route(["ALPHA"], 20, [_HEAD])
    recommended = _route(["BETA"], 21, [_MID])

    selected = _select_diverse_alts([first, recommended], cap=1,
                                    force_include=recommended)

    assert selected == [first, recommended]


def test_recommendations_for_both_time_modes_are_force_retained():
    first = _route(["ALPHA"], 20, [_HEAD])
    realistic = _route(["BETA"], 21, [_MID])
    best = _route(["GAMMA"], 22, [_OTHER_TAIL])

    selected = _select_diverse_alts(
        [first, realistic, best], cap=1, force_include=[realistic, best])

    assert selected == [first, realistic, best]


def test_arriveby_pin_uses_full_alt_stop_universe_and_retains_recommendation(monkeypatch):
    """The pin must not inherit the hover chip cap when deciding/revealing its recommendation."""
    monkeypatch.setattr(server_raptor, "RAPTOR_ALT_CHIP_CAP", 1)
    mc = {
        "alt_geom": {},
        "alt_chips": {0: ["hover-only"]},
        "alt_bundle": {"alt_stop": [{"hover-only": 1, "less-walk": 2}]},
        "typ": {},
    }
    monkeypatch.setattr(server_raptor, "mc_peek", lambda *args: mc)
    calls = []

    def walk(minutes):
        return {"mode": "walk", "min": minutes, "physical_min": minutes, "pts": []}

    class FakeTree:
        # Omit _select deliberately: this is a geometry/cap fixture, so recommendation falls
        # back honestly without MC fragility rather than requiring a full simulation kernel.
        def itinerary_via_stop(self, ci, stop, geom_provider=None):
            calls.append(stop)
            if stop == 1:
                return {"total": 22, "geom": [walk(1), _leg("B", _MID), walk(10)]}
            return {"total": 22, "geom": [walk(5), _leg("C", _OTHER_TAIL), walk(1)]}

    entry = {"tree": FakeTree(), "geom": {
        0: {"total": 22, "geom": [walk(11), _leg("A", _HEAD), walk(11)]}
    }}
    monkeypatch.setattr(server_raptor, "raptor_tree", lambda *args: entry)

    out = server_raptor._itinerary_alts(
        0, 37.77, -122.40, 8, "med", provider=object(), pin=True)

    assert calls == [1, 2]  # pin enumerated the whole alt_stop universe, not just hover chips
    assert [route["line"] for route in out] == ["hover-only", "less-walk"]
    assert out[-1]["_recommendation_metrics"] == ["r", "b"]


def test_pinned_response_publishes_distinct_map_and_recommended_choice_keys():
    map_route = _route(["ALPHA"], 22, [_HEAD])
    recommended = _route(["BETA"], 22, [_MID])
    server_raptor._annotate_route_families(map_route, [recommended])
    response = {"choice_key": map_route["choice_key"], "alts": [recommended]}

    server_raptor._publish_choice_recommendation(
        response, map_route, pin=True,
        recommended_routes={"r": recommended, "b": map_route})

    assert response["map_choice_key"] == map_route["choice_key"]
    assert response["recommended_choice_key"] == recommended["choice_key"]
    assert response["recommended_choice_keys"] == {
        "r": recommended["choice_key"],
        "b": map_route["choice_key"],
    }


def test_route_label_preserves_real_same_service_reboarding_legs():
    legs = [
        _leg("ALPHA", _HEAD, route_id="route-a"),
        _leg("ALPHA", _MID, route_id="route-a"),
        _leg("BETA", _TAIL, route_id="route-b"),
    ]

    assert server_raptor._route_label(legs) == "ALPHA > ALPHA > BETA"


def test_server_metadata_is_authoritative_for_primary_and_alternatives():
    primary = _route(["A"], 20, [_HEAD])
    same_family = _route(["B", "T"], 21, [_HEAD, _TAIL])
    alternate = _route(["C"], 22, [_MID])

    server_raptor._annotate_route_families(primary, [same_family, alternate])

    assert primary["family"]["key"] == same_family["family"]["key"]
    assert primary["branch"]["kind"] == "walk"
    assert same_family["branch"]["kind"] == "transit"
    assert alternate["family"]["key"] != primary["family"]["key"]
    for option in (primary, same_family, alternate):
        assert set(option["family"]) == {
            "key", "name", "sub", "lines", "services", "tags"}
        assert set(option["branch"]) == {
            "key", "name", "kind", "lines", "services", "serviceKeys"}


def test_public_choice_keys_keep_non_dominated_same_family_branch_options_distinct():
    """Family/branch is intentionally too broad to identify every selectable route."""
    via_x = _route(["A", "X", "T"], 25, [_HEAD, _MID, _TAIL])
    via_y = _route(["A", "Y", "T"], 25, [_HEAD, _MID, _TAIL])

    assert not server_raptor._alt_dominates(via_x, via_y)
    assert not server_raptor._alt_dominates(via_y, via_x)
    server_raptor._annotate_route_families(None, [via_x, via_y])

    assert via_x["family"]["key"] == via_y["family"]["key"]
    assert via_x["branch"]["key"] == via_y["branch"]["key"]
    assert via_x["choice_key"] != via_y["choice_key"]
    assert via_x["choice_key"].startswith("choice:")
    assert not any(key.startswith("_") for key in via_x)


def test_public_choice_key_prefers_durable_route_id_over_display_name():
    original = _route(["Old label"], 20, [_HEAD], route_ids=["route-10"])
    renamed = _route(["New label"], 20, [_HEAD], route_ids=["route-10"])
    same_label_other_route = _route(["Old label"], 20, [_HEAD], route_ids=["route-11"])

    original_key = server_raptor._public_choice_key(original)
    assert original_key == server_raptor._public_choice_key(renamed)
    assert original_key != server_raptor._public_choice_key(same_label_other_route)
    assert "route=route-10" in original_key
    assert original_key.startswith("choice:")


def test_public_choice_key_uses_display_name_only_without_durable_route_id():
    old_label = _route(["Old label"], 20, [_HEAD])
    new_label = _route(["New label"], 20, [_HEAD])

    assert server_raptor._public_choice_key(old_label) != server_raptor._public_choice_key(new_label)
    assert "display=Old label" in server_raptor._public_choice_key(old_label)


def test_walk_only_primary_is_not_described_as_walk_after_transit():
    primary = _route([], 5)

    server_raptor._annotate_route_families(primary, [])

    assert primary["family"]["name"] == "Walk option"
    assert primary["branch"] == {
        "key": "walk:only",
        "name": "walk only",
        "kind": "walk",
        "lines": [],
        "services": [],
        "serviceKeys": [],
    }


def test_family_catalog_surfaces_proven_services_omitted_by_card_cap():
    routes = [
        _route(["ALPHA"], 20, [_HEAD], route_ids=["route-a"]),
        _route(["BETA"], 21, [_HEAD], route_ids=["route-b"]),
        _route(["GAMMA"], 22, [_HEAD], route_ids=["route-c"]),
    ]

    selected = _select_diverse_alts(routes, cap=1)
    server_raptor._annotate_route_families(None, selected)

    assert [option["line"] for option in selected] == ["ALPHA"]
    services = selected[0]["family"]["services"]
    assert [service["name"] for service in services] == ["ALPHA", "BETA", "GAMMA"]
    assert [service["shown"] for service in services] == [True, False, False]
    assert selected[0]["family"]["lines"] == ["ALPHA", "BETA", "GAMMA"]
    assert all(service["key"].startswith("service:") for service in services)
    assert not ({service["key"] for service in services}
                & {service["name"] for service in services})


def test_branch_service_catalog_is_exact_and_never_cross_advertises():
    walk_a = _route(["ALPHA"], 20, [_HEAD], route_ids=["route-a"])
    walk_b = _route(["BETA"], 21, [_HEAD], route_ids=["route-b"])
    tail_a = _route(["ALPHA", "TAIL"], 22, [_HEAD, _TAIL],
                    route_ids=["route-a", "route-tail"])
    tail_c = _route(["GAMMA", "TAIL"], 23, [_HEAD, _TAIL],
                    route_ids=["route-c", "route-tail"])

    selected = _select_diverse_alts([walk_a, walk_b, tail_a, tail_c], cap=4)
    server_raptor._annotate_route_families(None, selected)

    assert {service["name"] for service in selected[0]["family"]["services"]} == {
        "ALPHA", "BETA", "GAMMA"}
    rendered_branch_keys = {option["branch"]["key"] for option in selected}
    assert all(service["branchKeys"]
               and set(service["branchKeys"]) <= rendered_branch_keys
               for service in selected[0]["family"]["services"])
    walk = next(option["branch"] for option in selected
                if option["branch"]["kind"] == "walk")
    transit = next(option["branch"] for option in selected
                   if option["branch"]["kind"] == "transit")
    assert {service["name"] for service in walk["services"]} == {"ALPHA", "BETA"}
    assert {service["name"] for service in transit["services"]} == {"ALPHA", "GAMMA"}
    assert set(walk["serviceKeys"]) == {service["key"] for service in walk["services"]}
    assert set(transit["serviceKeys"]) == {service["key"] for service in transit["services"]}


def test_family_union_excludes_services_proven_only_on_a_suppressed_branch():
    visible_walk = _route(["ALPHA"], 20, [_HEAD], route_ids=["route-a"])
    hidden_tail = _route(["BETA", "TAIL"], 21, [_HEAD, _TAIL],
                         route_ids=["route-b", "route-tail"])

    selected = _select_diverse_alts([visible_walk, hidden_tail], cap=1)
    server_raptor._annotate_route_families(None, selected)

    assert [option["line"] for option in selected] == ["ALPHA"]
    assert [service["name"] for service in selected[0]["family"]["services"]] == ["ALPHA"]
    rendered = {option["branch"]["key"] for option in selected}
    assert all(set(service["branchKeys"]) <= rendered
               for service in selected[0]["family"]["services"])


def test_family_annotation_strips_all_private_selection_metadata():
    primary = _route(["ALPHA"], 20, [_HEAD], route_ids=["route-a"])
    alternatives = [_route(["BETA"], 21, [_HEAD], route_ids=["route-b"])]
    selected = _select_diverse_alts(alternatives, cap=1, primary=primary)

    server_raptor._annotate_route_families(primary, selected)

    for route in [primary, *selected]:
        assert not [key for key in route if key.startswith("_")]


def test_service_keys_are_stable_when_display_names_change():
    def scenario(first, second):
        routes = [
            _route([first], 20, [_HEAD], route_ids=["stable-a"]),
            _route([second], 21, [_HEAD], route_ids=["stable-b"]),
        ]
        selected = _select_diverse_alts(routes, cap=1)
        server_raptor._annotate_route_families(None, selected)
        return selected[0]["family"]

    left = scenario("ALPHA", "BETA")
    right = scenario("ORANGE", "PURPLE")

    assert [service["key"] for service in left["services"]] == [
        service["key"] for service in right["services"]]
    assert left["lines"] != right["lines"]


def test_service_key_is_readable_route_structure_not_an_opaque_digest():
    service = server_raptor._leg_service_meta(
        _leg("Renamable display", _HEAD, feed="fixture-feed", mode="rail",
             route_id="route-stable"))

    assert service["key"] == (
        "service:feed=fixture-feed;mode=rail;route=route-stable")


def test_family_keys_are_readable_structural_ordinals_and_reorder_invariant():
    east = _route(["ALPHA"], 20, [_HEAD])
    north = _route(["BETA"], 21, [_MID])

    forward = server_raptor._discover_family_keys([east, north])
    reverse = server_raptor._discover_family_keys([north, east])

    assert set(forward.values()) == {"corridor:1", "corridor:2"}
    assert server_raptor._alt_family_key(east, forward) == server_raptor._alt_family_key(
        east, reverse)
    assert server_raptor._alt_family_key(north, forward) == server_raptor._alt_family_key(
        north, reverse)


def test_family_ordinals_are_response_local_not_persistent_ids():
    """A changed discovered-component set may renumber concise corridor ordinals.

    The API/frontend must use them only to join family data within one response.  Persistent
    state must be keyed by the request/cell and response lifecycle, never by a family ordinal.
    """
    first = _route(["ALPHA"], 20, [_HEAD])
    second = _route(["BETA"], 21, [_MID])

    full = server_raptor._discover_family_keys([first, second])
    subset = server_raptor._discover_family_keys([second])

    assert server_raptor._alt_family_key(second, full) == "corridor:2"
    assert server_raptor._alt_family_key(second, subset) == "corridor:1"


def test_branch_tail_label_aggregates_all_proven_converging_tails_after_cap():
    first = _route(["ALPHA", "ECHO"], 20, [_HEAD, _TAIL],
                   route_ids=["route-a", "tail-e"])
    second = _route(["ALPHA", "FOXTROT"], 21, [_HEAD, _TAIL],
                    route_ids=["route-a", "tail-f"])
    assert server_raptor._alt_branch_key(first) == server_raptor._alt_branch_key(second)

    selected = _select_diverse_alts([first, second], cap=1)
    server_raptor._annotate_route_families(None, selected)

    branch = selected[0]["branch"]
    assert branch["lines"] == ["ECHO", "FOXTROT"]
    assert "ECHO" in branch["name"] and "FOXTROT" in branch["name"]


def test_branch_tail_label_preserves_a_real_same_service_reboard():
    repeated = _route(
        ["ALPHA", "ECHO", "ECHO", "TAIL"], 25,
        [_HEAD, _TAIL, _TAIL, _TAIL],
        route_ids=["route-a", "route-e", "route-e", "route-tail"],
    )

    selected = _select_diverse_alts([repeated], cap=1)
    server_raptor._annotate_route_families(None, selected)

    branch = selected[0]["branch"]
    assert "ECHO > ECHO" in branch["name"]
    assert branch["lines"] == ["ECHO", "TAIL"]


def test_plural_branch_tail_label_reads_as_distinct_choices():
    option = _route(["ALPHA", "ECHO"], 20, [_HEAD, _TAIL])

    meta = server_raptor._branch_display_meta(
        "ALPHA", "tail:fixture", [option],
        tail_sequences=[("ECHO",), ("CONNECTOR", "FOXTROT"), ("FOXTROT",)],
    )

    assert meta["name"] == "via ECHO, FOXTROT, or CONNECTOR > FOXTROT"


def test_primary_family_context_cannot_consume_a_missing_family_slot():
    primary = _route(["A"], 19, [_HEAD])
    same_family_branch = _route(["A", "T"], 20, [_HEAD, _TAIL])
    missing_family = _route(["B"], 21, [_MID])

    out = _select_diverse_alts([same_family_branch, missing_family], cap=1, primary=primary)

    assert [option["line"] for option in out] == ["B"]


def test_cap_pressure_reserves_one_primary_family_sibling_and_keeps_breadth():
    primary = _route(["A"], 19, [_HEAD])
    sibling = _route(["A", "T"], 20, [_HEAD, _TAIL])
    distinct = []
    for i, name in enumerate(["B", "C", "D", "E", "F", "G"]):
        lat = 37.760 + i * 0.004
        path = [[lat, -122.480], [lat, -122.470], [lat, -122.460]]
        distinct.append(_route([name], 21 + i, [path]))

    out = _select_diverse_alts([sibling, *distinct], cap=6, primary=primary)
    labels = [option["line"] for option in out]

    assert len(out) == 6
    assert "A > T" in labels
    assert labels[:6] == ["A > T", "B", "C", "D", "E", "F"]


def test_cap_pressure_prefers_alt_to_alt_siblings_over_primary_sibling():
    primary_path = [[37.800, -122.500], [37.800, -122.490], [37.800, -122.480]]
    primary = _route(["P"], 18, [primary_path])
    primary_sibling = _route(["P", "Q"], 19, [primary_path, _OTHER_TAIL])
    alt_walk = _route(["A"], 20, [_HEAD])
    alt_tail = _route(["A", "T"], 21, [_HEAD, _TAIL])
    distinct = []
    for i, name in enumerate(["B", "C", "D", "E", "F"]):
        lat = 37.760 + i * 0.004
        path = [[lat, -122.480], [lat, -122.470], [lat, -122.460]]
        distinct.append(_route([name], 22 + i, [path]))

    out = _select_diverse_alts(
        [primary_sibling, alt_walk, alt_tail, *distinct], cap=6, primary=primary)
    labels = [option["line"] for option in out]

    assert "A" in labels and "A > T" in labels
    assert "P > Q" not in labels
    assert len({option["_family_seed"] for option in out}) == 5


def test_slow_sibling_cannot_displace_much_faster_missing_corridor():
    primary_path = [[37.800, -122.500], [37.800, -122.490], [37.800, -122.480]]
    primary = _route(["P"], 18, [primary_path])
    alt_walk = _route(["A"], 20, [_HEAD])
    slow_sibling = _route(["A", "T"], 35, [_HEAD, _TAIL])
    distinct = []
    for i, name in enumerate(["B", "C", "D", "E", "F", "G"]):
        lat = 37.760 + i * 0.004
        path = [[lat, -122.480], [lat, -122.470], [lat, -122.460]]
        distinct.append(_route([name], 21 + i, [path]))

    out = _select_diverse_alts([alt_walk, slow_sibling, *distinct], cap=6, primary=primary)
    labels = [option["line"] for option in out]

    assert labels == ["A", "B", "C", "D", "E", "F"]
    assert "A > T" not in labels


def test_pinned_family_completion_is_branch_complete_bounded_and_rename_invariant():
    """The practical cap chooses corridors; an admitted family keeps each qualifying finish.

    The scenario is constructed twice with unrelated service labels.  One early family consumes
    the selector's single practical sibling reservation. A later admitted family has an equal-time
    walk finish and transit finish: strict selection keeps its simpler walk representative, while
    pinned completion must restore its other branch without admitting a new corridor. A third
    branch four minutes behind its family head remains outside the existing three-minute near-tie
    even though it is still inside the displaced-corridor ceiling, and stays suppressed.
    """
    def scenario(prefix):
        names = [f"{prefix}{i}" for i in range(11)]

        def corridor(lat):
            return [[lat, -122.490], [lat, -122.480], [lat, -122.470]]

        def tail(lat, north=True):
            delta = 0.008 if north else -0.008
            return [[lat, -122.470], [lat + delta / 2, -122.462],
                    [lat + delta, -122.454]]

        primary = _route([names[0]], 18, [corridor(37.800)])
        early_head = corridor(37.700)
        target_head = corridor(37.750)
        alternatives = [
            _route([names[1]], 20, [early_head]),
            _route([names[1], names[2]], 21, [early_head, tail(37.700)]),
            _route([names[3]], 21, [corridor(37.710)]),
            _route([names[4]], 22, [corridor(37.720)]),
            _route([names[5]], 23, [corridor(37.730)]),
            _route([names[6]], 24, [target_head]),
            _route([names[6], names[7]], 24, [target_head, tail(37.750)]),
            _route([names[6], names[8]], 28, [target_head, tail(37.750, north=False)]),
            _route([names[9]], 25, [corridor(37.770)]),
            _route([names[10]], 26, [corridor(37.780)]),
        ]
        return primary, alternatives, names

    shapes = []
    for prefix in ("LEFT_", "RENAMED_"):
        primary, alternatives, names = scenario(prefix)
        strict = _select_diverse_alts(alternatives, cap=6, primary=primary)
        complete = _select_diverse_alts(
            alternatives, cap=6, primary=primary, complete_selected_families=True)

        assert len(strict) == 6
        assert names[6] in {option["line"] for option in strict}
        assert f"{names[6]} > {names[7]}" not in {option["line"] for option in strict}
        assert f"{names[6]} > {names[7]}" in {option["line"] for option in complete}
        assert f"{names[6]} > {names[8]}" not in {option["line"] for option in complete}
        assert len(complete) == 7, "branch completion admitted a new corridor or slow branch"

        server_raptor._annotate_route_families(primary, complete)
        target = next(option for option in complete if option["line"] == names[6])
        target_family = target["family"]["key"]
        target_members = [option for option in complete
                          if option["family"]["key"] == target_family]
        assert {option["branch"]["kind"] for option in target_members} == {"walk", "transit"}
        shapes.append(sorted(
            (server_raptor._alt_total(option),
             len(server_raptor._alt_transit_legs(option)),
             option["branch"]["kind"])
            for option in complete
        ))

    assert shapes[0] == shapes[1]


def test_selection_family_seed_preserves_membership_after_bridge_removal():
    def shifted(name, lon, total):
        path = [[37.700, lon], [37.700, lon + 0.010], [37.700, lon + 0.020]]
        return _route([name], total, [path])

    primary = shifted("A", -122.450, 19)
    bridge = shifted("B", -122.449, 30)
    far_member = shifted("C", -122.4495, 20)
    full_keys = server_raptor._discover_family_keys([primary, bridge, far_member])
    subset_keys = server_raptor._discover_family_keys([primary, far_member])
    expected_seed = server_raptor._alt_family_key(far_member, full_keys)
    # Readable corridor ordinals are response-local.  This remains corridor:1 in both one-family
    # responses; the load-bearing property is that selection carries the full-universe family
    # membership through pruning, not that an opaque digest changes with every member set.
    assert expected_seed == server_raptor._alt_family_key(far_member, subset_keys)

    selected = _select_diverse_alts([bridge, far_member], cap=1, primary=primary)
    assert selected == [far_member]
    assert far_member["_family_seed"] == expected_seed
    server_raptor._annotate_route_families(primary, selected)

    assert primary["family"]["key"] == far_member["family"]["key"] == expected_seed


def test_family_selection_and_membership_are_invariant_under_route_renaming():
    def scenario(names):
        return [
            _route([names[0]], 20, [_HEAD]),
            _route([names[0], names[2]], 21, [_HEAD, _TAIL]),
            _route([names[0], names[1], names[2]], 22, [_HEAD, _MID, _TAIL]),
            _route([names[3]], 23, [_MID]),
        ]

    left = scenario(["A", "X", "T", "B"])
    right = scenario(["P", "Q", "R", "S"])
    left_selected = _select_diverse_alts(left, cap=3)
    right_selected = _select_diverse_alts(right, cap=3)
    server_raptor._annotate_route_families(left[0], left[1:])
    server_raptor._annotate_route_families(right[0], right[1:])

    assert [a["typical"]["total"] for a in left_selected] == [
        a["typical"]["total"] for a in right_selected]
    assert [a["family"]["key"] for a in left] == [a["family"]["key"] for a in right]
    assert [a["branch"]["key"] for a in left] == [a["branch"]["key"] for a in right]


def test_server_route_family_source_contains_no_concrete_line_heuristics():
    source = Path(server_raptor.__file__).read_text()
    banned_symbols = (
        "_family_is_22", "_is_shared_corridor_feeder_to_19", "55_49",
        "_is_protected_branch_alt", "_direct_family_branches",
    )
    assert not [symbol for symbol in banned_symbols if symbol in source]
    quoted_line = re.compile(r"(['\"])(?:22|19|55|49|K|L|M|N)\1")
    assert quoted_line.search(source) is None
    assert "hashlib.sha1" not in source


def test_departafter_window_alt_label_uses_traced_route_sequence(monkeypatch):
    class FakeTree:
        def itinerary_via_stop(self, ci, stop, geom_provider=None, percentile=None):
            assert ci == 0
            assert stop == 12
            assert percentile == "planned"
            return {
                "total": 27,
                "geom": [
                    {"mode": "transit", "name": "22", "min": 12},
                    {"mode": "transit", "name": "19", "min": 3},
                ],
            }

    monkeypatch.setattr(
        server_raptor,
        "_alt_route_preamble",
        lambda *args, **kwargs: ("ready", ({}, {"22": 12}, ["22"])),
    )
    entry = {"planned": True, "tree": FakeTree(), "geom": {0: {"geom": []}}, "geom5": {}}

    out = server_raptor._itinerary_alts_departafter(
        0, entry, 37.77, -122.40, 8, "med", prov50=object(), prov5=object())

    assert [a["line"] for a in out] == ["22 > 19"]
    assert out[0]["chip_line"] == "22"


def test_final_alight_tie_prefers_shorter_egress_walk():
    # p=1 and p=2 both arrive at W at t=1260, but p=2 stays on the vehicle longer and walks less.
    pat_arr = [0, 600, 720]
    pat_stops = [0, 1, 2]
    eg_sec = [9999, 660, 540]

    apos, walk = _min_overshoot_alight(
        pat_arr, pat_stops, eg_sec, trow=0, sbase=0, bpos=0, ns=3, apos=1, nd_egress=660)

    assert apos == 2
    assert walk == 540


def test_planned_branch_tie_prefers_later_alight_less_final_walk():
    early = {
        "line": "22",
        "total": 25,
        "sig": (("early",),),
        "raw": [("ride", 1, 0, 600, 0, 3, 10), ("egress", 660, 10)],
    }
    late = {
        "line": "22",
        "total": 25,
        "sig": (("late",),),
        "raw": [("ride", 1, 0, 720, 0, 4, 11), ("egress", 540, 11)],
    }

    assert DepartAfterJourneyTree._planned_candidate_better(late, early)
    assert not DepartAfterJourneyTree._planned_candidate_better(early, late)


def test_tail_alight_tie_prefers_shorter_egress_walk():
    # Same finish time, but the later alight walks less. This is the tail-branch equivalent of
    # the final-ride no-overshoot tie-break.
    best = (1260, 3, 10, 660)
    assert DepartAfterJourneyTree._alight_tail_better(1260, 4, 540, best)
    assert not DepartAfterJourneyTree._alight_tail_better(1260, 3, 660, (1260, 4, 11, 540))


# ---- A1: primary-dedupe signature must compare (best, typical) pairs, not (typical, typical) ----

_PRIM_LEGS = [
    {"mode": "walk", "min": 5},
    {"mode": "transit", "name": "J", "min": 10, "wait": 2,
     "pts": [[37.76, -122.43], [37.77, -122.42]]},
    {"mode": "walk", "min": 3},
]
_ALT_BEST_LEGS = [
    {"mode": "walk", "min": 4},
    {"mode": "transit", "name": "B-alt", "min": 9, "wait": 1,
     "pts": [[37.75, -122.44], [37.76, -122.43]]},
    {"mode": "walk", "min": 2},
]
_ALT_TYP_LEGS = [
    {"mode": "walk", "min": 6},
    {"mode": "transit", "name": "T-alt", "min": 11, "wait": 3,
     "pts": [[37.74, -122.41], [37.75, -122.40]]},
    {"mode": "walk", "min": 2},
]


def test_primary_dedupe_keeps_alt_whose_typical_matches_but_best_differs(monkeypatch):
    """Regression for the (sig(typical), sig(typical)) primary-dedupe bug: an alt whose TYPICAL
    journey duplicates the primary's typical but whose BEST-CASE is a genuinely different route
    must SURVIVE (it informs the best-case toggle); the old tuple fabricated both slots from the
    typical legs, matched psig whenever the primary's p5==p50, and over-filtered it. The vice-versa
    alt (best matches the primary, typical differs) must survive too. An alt duplicating the
    primary in BOTH slots is still dropped."""

    class FakeTree:
        def itinerary_via_stop(self, ci, stop, geom_provider=None, percentile=None):
            table = {
                # alt A: typical == the primary's typical, best is a distinct route -> KEEP
                (1, 50): {"total": 18, "geom": _PRIM_LEGS},
                (1, 5): {"total": 15, "geom": _ALT_BEST_LEGS},
                # alt B (vice versa): best == the primary's best, typical differs -> KEEP
                (2, 50): {"total": 19, "geom": _ALT_TYP_LEGS},
                (2, 5): {"total": 15, "geom": _PRIM_LEGS},
                # alt D: exact duplicate of the primary pair (faster total) -> DROP
                (3, 50): {"total": 17, "geom": _PRIM_LEGS},
                (3, 5): {"total": 17, "geom": _PRIM_LEGS},
            }
            return table[(stop, percentile)]

    geom_cache = {}
    monkeypatch.setattr(
        server_raptor,
        "_alt_route_preamble",
        lambda *args, **kwargs: ("ready", (geom_cache, {"A": 1, "B": 2, "D": 3},
                                           ["A", "B", "D"])),
    )
    primary = {"total": 18, "geom": _PRIM_LEGS}
    entry = {"planned": False, "tree": FakeTree(),
             "geom": {0: primary}, "geom5": {0: primary}}   # primary p5 == p50

    out = server_raptor._itinerary_alts_departafter(
        0, entry, 37.77, -122.40, 8, "med", prov50=object(), prov5=object())
    labels = [a["line"] for a in out]

    assert labels == ["J", "T-alt"], labels
    strip_j = out[0]
    # The duplicate D (total 17) must not have hijacked the "J" strip via the dedupe-by-label:
    assert strip_j["typical"]["total"] == 18
    assert [l["name"] for l in strip_j["best"]["legs"]
            if l.get("mode") == "transit"] == ["B-alt"]
    assert geom_cache[0] is out              # assembled alts cached for the cell


# ---- A3: the per-pin typ cache is keyed by the alts list it was computed for ----

def test_typ_cache_recomputes_on_alts_signature_mismatch(monkeypatch):
    """The typ/frag result is POSITIONALLY aligned to the request's alts list; a pin that raced
    the MC build (alts=[]) must not poison the cache for later pins that see the full list."""
    mc = {"typ": {}}
    monkeypatch.setattr(server_raptor, "mc_peek", lambda *a, **k: mc)

    alts_empty = []
    status, payload = server_raptor._alt_typicals_preamble(0, 37.77, -122.40, 8, "med",
                                                           alts_empty)
    assert status == "ready"
    typ_cache, got_mc, sig_empty = payload
    assert got_mc is mc and typ_cache is mc["typ"]
    result_empty = {"prim_frag": 2, "alt_frags": []}
    typ_cache[0] = {"sig": sig_empty, "out": result_empty}

    # Same alts list -> served from the cache.
    status, payload = server_raptor._alt_typicals_preamble(0, 37.77, -122.40, 8, "med",
                                                           alts_empty)
    assert status == "cached" and payload is result_empty

    # A different alts list (the MC landed; branches expanded) -> signature mismatch -> recompute.
    alts_full = [{"line": "22", "via_stop": 7,
                  "typical": {"total": 21, "legs": [
                      {"mode": "transit", "name": "22", "min": 12}]}}]
    status, payload = server_raptor._alt_typicals_preamble(0, 37.77, -122.40, 8, "med",
                                                           alts_full)
    assert status == "ready", "stale typ entry served for a different alts list"
    _tc, _mc2, sig_full = payload
    assert sig_full != sig_empty
    typ_cache[0] = {"sig": sig_full, "out": {"prim_frag": 2, "alt_frags": [3]}}
    status, payload = server_raptor._alt_typicals_preamble(0, 37.77, -122.40, 8, "med",
                                                           alts_full)
    assert status == "cached" and payload == {"prim_frag": 2, "alt_frags": [3]}


# ---- A5: /itinerary responses never alias (or annotate) the cached alt dicts ----

def test_pin_response_alts_are_copies_departafter(monkeypatch):
    cached_alt = {"line": "22", "via_stop": 7,
                  "typical": {"total": 21, "legs": []}, "best": {"total": 21, "legs": []}}
    cached_alts = [cached_alt]
    typ_j = {"total": 20, "xfers": 0, "legs": [], "geom": []}
    entry = {"planned": True, "tree": object(), "geom": {0: typ_j}, "geom5": {},
             "cells": {"c0": [20, 20]}}

    class FakeRaptor:
        cell_index = {"c0": 0}
        cell_ids = ["c0"]

    monkeypatch.setattr(server_raptor, "_RAPTOR", FakeRaptor())
    monkeypatch.setattr(server_raptor, "raptor_tree", lambda *a, **k: entry)
    monkeypatch.setattr(server_raptor, "_itinerary_alts_departafter",
                        lambda *a, **k: cached_alts)
    monkeypatch.setattr(server_raptor, "_itinerary_alt_typicals_departafter",
                        lambda *a, **k: {"prim_frag": 2, "alt_frags": [3]})

    res = server_raptor.itinerary_departafter("c0", None, None, 37.77, -122.40, 8, "med", 1.0,
                                              pin=True)

    assert res["frag"] == 2
    assert res["alts"][0]["frag"] == 3
    assert res["alts"][0] is not cached_alt, "response alt aliases the cached dict"
    assert "frag" not in cached_alt, "pin-only field leaked into the cached alt (hover leak)"
    assert "alts" not in typ_j and "frag" not in typ_j


def test_pin_response_alts_are_copies_arriveby(monkeypatch):
    cached_alt = {"line": "22", "min": 21, "legs": []}
    res_cached = {"total": 20, "xfers": 0, "legs": [], "geom": []}
    entry = {"tree": object(), "geom": {0: res_cached}, "cells": {"c0": [20, 20]}}

    class FakeRaptor:
        cell_index = {"c0": 0}
        cell_ids = ["c0"]

    monkeypatch.setattr(server_raptor, "_RAPTOR", FakeRaptor())
    monkeypatch.setattr(server_raptor, "raptor_tree", lambda *a, **k: entry)
    monkeypatch.setattr(server_raptor, "_itinerary_alts", lambda *a, **k: [cached_alt])
    monkeypatch.setattr(server_raptor, "_itinerary_alt_typicals",
                        lambda *a, **k: {"prim": (25, 4), "alts": [(30, 5)]})

    res = server_raptor.itinerary_arriveby("c0", None, None, 37.77, -122.40, 8, "med", 1.0,
                                           pin=True)

    assert (res["real"], res["frag"]) == (25, 4)
    assert (res["alts"][0]["real"], res["alts"][0]["frag"]) == (30, 5)
    assert res["alts"][0] is not cached_alt, "response alt aliases the cached dict"
    assert "real" not in cached_alt and "frag" not in cached_alt
    assert "alts" not in res_cached and "real" not in res_cached


# ---- A4 + A6 + A7: chipless pins cache their branch alts; enumeration is geometry-free ----

def test_chipless_pin_branch_alts_cached_and_enumeration_geometry_free(monkeypatch):
    """A pinned cell whose MC entry exists but has NO window chips must still run the branch
    expansion, cache the result under (ci, "branches") in the MC entry's alt_geom (A4 — the old
    order returned before the cache and recomputed per click), construct branch alts with an
    unconditional via_stop (A6), and enumerate geometry-free with hydration through the real
    provider only for survivors (A7)."""
    mc = {"alt_geom": {}, "alt_chips": {}, "alt_bundle": None, "typ": {}}
    monkeypatch.setattr(server_raptor, "mc_peek", lambda *a, **k: mc)
    calls = {"enum": 0, "hydrate": 0}

    class FakeTree:
        def planned_branch_itineraries(self, ci, base, window, geom_provider=None):
            calls["enum"] += 1
            assert geom_provider is None, "enumeration must run geometry-free (A7)"
            return [{"line": "22", "total": 21, "stop": 7, "sig": (),
                     "raw": [], "home": 0, "jt": None, "target_sec": 1260, "it": {}}]

        def _format_planned_raw(self, ci, s, raw, home, jt, geom_provider=None,
                                planned_total=None, planned_target_sec=None):
            calls["hydrate"] += 1
            assert geom_provider is not None, "survivor hydration must use the real provider"
            assert planned_target_sec == 1260
            return {"total": planned_total,
                    "geom": [{"mode": "transit", "name": "22", "min": 12,
                              "pts": [[37.76, -122.43], [37.77, -122.42]]},
                             {"mode": "walk", "min": 4}]}

    entry = {"planned": True, "tree": FakeTree(),
             "geom": {0: {"total": 20, "geom": []}}, "geom5": {}}

    out = server_raptor._itinerary_alts_departafter(
        0, entry, 37.77, -122.40, 8, "med", prov50=object(), prov5=object(),
        expand_branches=True)

    assert [a["line"] for a in out] == ["22"]
    assert out[0]["via_stop"] == 7, "branch alt must carry its access stop unconditionally (A6)"
    assert (0, "branches") in mc["alt_geom"], "chipless pin result must be cached (A4)"
    assert "_branch" not in out[0]

    out2 = server_raptor._itinerary_alts_departafter(
        0, entry, 37.77, -122.40, 8, "med", prov50=object(), prov5=object(),
        expand_branches=True)
    assert out2 == out
    assert calls["enum"] == 1, "second chipless pin re-ran the branch enumeration (A4)"


def test_planned_proxy_tie_uses_schedule_structure_before_hydration(monkeypatch):
    """Equivalent branch proxies choose one readable scheduled representative before walks.

    These candidates have the same route/pattern, boarding/alighting stops, displayed total and
    destination-facing branch.  They differ only in their scheduled occurrence.  The exact street
    access/egress polyline is semantically interchangeable, so it must not force both candidates
    through geometry hydration merely to obtain a deterministic winner.
    """
    mc = {"alt_geom": {}, "alt_chips": {}, "alt_bundle": None, "typ": {}}
    monkeypatch.setattr(server_raptor, "mc_peek", lambda *a, **k: mc)
    calls = {"hydrate": 0, "departures": []}
    head = [[37.76, -122.43], [37.77, -122.42]]
    route_shape = ((("route", "fixture", "service-r", ("stops", (4, 7, 9))), 1, 3),)

    def proxy(branch):
        legs = [{"mode": "transit", "name": "Fixture service", "feed": "fixture",
                 "tmode": "bus", "route_id": "service-r", "min": 12, "pts": head}]
        return {"line": "Fixture service", "source": "branch", "_branch": branch,
                "_needs_hydration": True, "via_stop": int(branch["stop"]),
                "typical": {"total": int(branch["total"]), "legs": legs},
                "best": {"total": int(branch["total"]), "legs": legs}}

    monkeypatch.setattr(server_raptor, "_planned_branch_proxy_option", proxy)

    class FakeTree:
        def planned_branch_itineraries(self, ci, base, window, geom_provider=None):
            assert geom_provider is None
            # Both are valid, equal-quality variants. The later viable boarding anchor wins the
            # canonical deterministic rank; it should be the only one hydrated.
            return [
                {"line": "Fixture service", "total": 21, "stop": 4,
                 "raw": [("access", 120), ("ride", 8, 540, 900, 1, 3, 9),
                         ("egress", 300, 9)], "route_key": route_shape,
                 "home": 420, "jt": object(), "it": {}},
                {"line": "Fixture service", "total": 21, "stop": 4,
                 "raw": [("access", 120), ("ride", 8, 510, 870, 1, 3, 9),
                         ("egress", 300, 9)], "route_key": route_shape,
                 "home": 390, "jt": object(), "it": {}},
            ]

        def _format_planned_raw(self, ci, s, raw, home, jt, geom_provider=None,
                                planned_total=None, planned_target_sec=None):
            calls["hydrate"] += 1
            calls["departures"].append(int(raw[1][2]))
            assert geom_provider is not None
            return {"total": planned_total,
                    "geom": [{"mode": "walk", "min": 2},
                             {"mode": "transit", "name": "Fixture service", "min": 12,
                              "pts": head},
                             {"mode": "walk", "min": 5}]}

    entry = {"planned": True, "tree": FakeTree(),
             "geom": {0: {"total": 20, "geom": []}}, "geom5": {}}
    out = server_raptor._itinerary_alts_departafter(
        0, entry, 37.77, -122.40, 8, "med", prov50=object(), prov5=object(),
        expand_branches=True)

    assert [a["line"] for a in out] == ["Fixture service"]
    assert calls == {"hydrate": 1, "departures": [540]}


def test_planned_closure_cannot_surface_candidate_faster_than_map_primary(monkeypatch):
    """A closure-discovered branch is checked against the canonical painted route, not reranked."""
    monkeypatch.setattr(server_raptor, "mc_peek", lambda *a, **k: None)
    head = [[37.76, -122.43], [37.77, -122.42]]

    class FakeTree:
        def planned_branch_access_stops(self, ci, base, window):
            return {7}

        def planned_branch_itineraries(self, ci, base, window, geom_provider=None, access_stops=None):
            assert geom_provider is None and access_stops == {7}
            return [{"line": "ALPHA", "total": 20, "stop": 7,
                     "raw": [("access", 120), ("ride", 8, 540, 900, 1, 3, 9),
                             ("egress", 180, 9)],
                     "route_key": (("fixture",),), "home": 420, "jt": self, "it": {}}]

        def _format_planned_raw(self, ci, s, raw, home, jt, geom_provider=None,
                                planned_total=None, planned_target_sec=None):
            return {"total": planned_total,
                    "geom": [{"mode": "walk", "min": 2},
                             {"mode": "transit", "name": "ALPHA", "feed": "fixture",
                              "tmode": "bus", "route_id": "route-a", "min": 6, "pts": head},
                             {"mode": "walk", "min": 3}]}

    entry = {"planned": True, "tree": FakeTree(),
             "geom": {0: {"total": 21, "geom": []}}, "geom5": {}, "branch_geom": {}}
    with pytest.raises(AssertionError, match="faster than canonical primary"):
        server_raptor._itinerary_alts_departafter(
            0, entry, 37.77, -122.40, 8, "med", prov50=object(), prov5=object(),
            expand_branches=True)


def test_pre_variance_pin_serves_branch_alts_and_caches_once(monkeypatch):
    """PRODUCT SPEC (test_itinerary_equals_map_departafter pins WITHOUT /variance priming): a
    pinned planned cell must serve its deterministic branch alternatives even before the MC
    entry exists — they come from the cached tree, not the MC. The result is cached on the TREE
    entry ("branch_geom") so a repeat pre-variance pin does NOT re-enumerate; once the MC lands,
    the (ci, "branches") MC-entry cache takes over. A plain HOVER still returns [] pre-variance
    (the alt chips come from the MC)."""
    mc_holder = {"mc": None}                      # starts pre-variance; flipped below
    monkeypatch.setattr(server_raptor, "mc_peek", lambda *a, **k: mc_holder["mc"])
    calls = {"enum": 0}

    class FakeTree:
        def planned_branch_itineraries(self, ci, base, window, geom_provider=None):
            calls["enum"] += 1
            assert geom_provider is None, "enumeration must run geometry-free (A7)"
            return [{"line": "22", "total": 21, "stop": 7, "sig": (),
                     "raw": [], "home": 0, "jt": None, "it": {}}]

        def _format_planned_raw(self, ci, s, raw, home, jt, geom_provider=None,
                                planned_total=None, planned_target_sec=None):
            assert geom_provider is not None, "survivor hydration must use the real provider"
            return {"total": planned_total,
                    "geom": [{"mode": "transit", "name": "22", "min": 12,
                              "pts": [[37.76, -122.43], [37.77, -122.42]]},
                             {"mode": "walk", "min": 4}]}

    entry = {"planned": True, "tree": FakeTree(),
             "geom": {0: {"total": 20, "geom": []}}, "geom5": {}, "branch_geom": {}}

    # Pre-variance PIN: branch alts served (not []), enumerated once, cached on the tree entry.
    out = server_raptor._itinerary_alts_departafter(
        0, entry, 37.77, -122.40, 8, "med", prov50=object(), prov5=object(),
        expand_branches=True)
    assert [a["line"] for a in out] == ["22"], "pre-variance pin must serve branch alts"
    assert out[0]["via_stop"] == 7
    assert calls["enum"] == 1
    assert entry["branch_geom"][0] is out, "pre-variance pin result must be cached on the entry"

    # Repeat pre-variance PIN: served from the tree-entry cache, no re-enumeration.
    out2 = server_raptor._itinerary_alts_departafter(
        0, entry, 37.77, -122.40, 8, "med", prov50=object(), prov5=object(),
        expand_branches=True)
    assert out2 is out
    assert calls["enum"] == 1, "second pre-variance pin re-ran the branch enumeration"

    # Plain HOVER pre-variance: still [] (chips come from the MC), and no enumeration.
    hov = server_raptor._itinerary_alts_departafter(
        0, entry, 37.77, -122.40, 8, "med", prov50=object(), prov5=object())
    assert hov == [] and calls["enum"] == 1

    # /variance lands: the MC entry's (ci, "branches") cache takes over seamlessly — one fresh
    # build cached there, and the pre-variance tree-entry cache stops being consulted.
    mc_holder["mc"] = {"alt_geom": {}, "alt_chips": {}, "alt_bundle": None, "typ": {}}
    out3 = server_raptor._itinerary_alts_departafter(
        0, entry, 37.77, -122.40, 8, "med", prov50=object(), prov5=object(),
        expand_branches=True)
    assert [a["line"] for a in out3] == ["22"]
    assert calls["enum"] == 2, "post-variance pin must rebuild against the MC entry"
    assert (0, "branches") in mc_holder["mc"]["alt_geom"]
    out4 = server_raptor._itinerary_alts_departafter(
        0, entry, 37.77, -122.40, 8, "med", prov50=object(), prov5=object(),
        expand_branches=True)
    assert out4 is mc_holder["mc"]["alt_geom"][(0, "branches")]
    assert calls["enum"] == 2
