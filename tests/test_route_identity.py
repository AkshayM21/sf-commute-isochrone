"""Direct tests for server-free structural route identity primitives."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

from core import route_identity


def _leg(name="A", *, feed="feed", route_id="route", points=None):
    return {
        "mode": "transit", "name": name, "feed": feed, "tmode": "bus",
        "route_id": route_id, "pts": points or [(37.70, -122.40), (37.71, -122.39)],
        "min": 10,
    }


def _alt(leg):
    return {"best": {"legs": [leg], "total": 20},
            "typical": {"legs": [leg], "total": 20}}


def test_identity_is_service_qualified_and_directional():
    east = _leg()
    west = _leg(points=[(37.70, -122.40), (37.69, -122.41)])
    assert route_identity.leg_service_sig(east) == "feed|bus"
    assert route_identity.leg_route_id(east) == "route"
    assert route_identity.leg_dir_sig(east) != route_identity.leg_dir_sig(west)
    assert route_identity.alt_dedupe_key(_alt(east)) != route_identity.alt_dedupe_key(_alt(west))


def test_family_discovery_uses_injected_corridor_callback():
    left = _alt(_leg(points=[(37.70, -122.40), (37.71, -122.39)]))
    right = _alt(_leg(points=[(37.70, -122.40), (37.71, -122.39)]))
    raw_calls = []

    def raw_override(a, b):
        raw_calls.append((a, b))
        return False

    keys = route_identity.discover_family_keys([left, right], same_corridor=raw_override)
    assert keys[id(left)] != keys[id(right)]
    assert raw_calls and all(call[0] in (left["best"]["legs"]) and
                             call[1] in (right["best"]["legs"]) for call in raw_calls)


def test_family_discovery_fast_path_precomputes_profiles_and_preserves_keys():
    options = [_alt(_leg(points=[(37.70, -122.40), (37.71, -122.39)])) for _ in range(12)]
    baseline = route_identity.discover_family_keys(options)
    profile_calls = []

    def profile_predicate(left, right, left_name, right_name):
        profile_calls.append((left, right, left_name, right_name))
        return route_identity.same_boarding_profiles(left, right, left_name, right_name)

    fast = route_identity.discover_family_keys(
        options,
        same_corridor=route_identity.same_boarding_corridor,
        same_corridor_default=route_identity.same_boarding_corridor,
        same_profile=profile_predicate,
    )
    assert fast == baseline
    assert len(profile_calls) > 0
    assert all(call[0] is not None and call[1] is not None for call in profile_calls)


def test_family_discovery_profile_work_is_linear_in_option_count():
    count = 400
    options = [_alt(_leg(points=[(37.70, -122.40), (37.71, -122.39)]))
               for _ in range(count)]
    profile_calls = 0
    profile_count = 0

    original_profile = route_identity.leg_boarding_profile

    def counted_profile(leg):
        nonlocal profile_count
        profile_count += 1
        return original_profile(leg)

    def counted_predicate(left, right, left_name, right_name):
        nonlocal profile_calls
        profile_calls += 1
        return route_identity.same_boarding_profiles(left, right, left_name, right_name)

    route_identity.leg_boarding_profile = counted_profile
    try:
        route_identity.discover_family_keys(
            options,
            same_corridor=route_identity.same_boarding_corridor,
            same_corridor_default=route_identity.same_boarding_corridor,
            same_profile=counted_predicate,
        )
    finally:
        route_identity.leg_boarding_profile = original_profile

    assert profile_count == count
    assert profile_calls <= count * (count - 1) // 2
