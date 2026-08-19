"""Pure shadow-memory accounting tests (no server boot, routing, or JIT)."""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from core import server_raptor as sr  # noqa: E402


def test_owned_payload_aliases_are_counted_once():
    shared = {"legs": [{"mode": "walk", "min": 3}]}
    shared_bytes = sr._owned_payload_nbytes(shared)
    root = (shared, shared)

    assert sr._owned_payload_nbytes(root) == sys.getsizeof(root) + shared_bytes


def test_owned_payload_arrays_and_views_count_one_base_allocation():
    array = np.arange(32, dtype=np.int64)
    view = array[4:20]

    array_bytes = sr._owned_payload_nbytes(array)
    view_bytes = sr._owned_payload_nbytes(view)
    assert array_bytes >= array.nbytes + 128
    assert view_bytes == sys.getsizeof(view) + array_bytes

    aliased = (array, array)
    assert sr._owned_payload_nbytes(aliased) == sys.getsizeof(aliased) + array_bytes

    other_view = array[8:24]
    roots = (view, other_view)
    assert sr._owned_payload_nbytes(roots) == (
        sys.getsizeof(roots) + sys.getsizeof(view) + sys.getsizeof(other_view) + array_bytes)


def test_owned_payload_borrowed_ndarray_base_keeps_only_owned_view_headers():
    array = np.arange(32, dtype=np.int64)
    first_view = array[2:18]
    second_view = array[10:28]
    roots = (first_view, second_view)

    assert sr._owned_payload_nbytes(roots, borrowed_root_ids={id(array)}) == (
        sys.getsizeof(roots) + sys.getsizeof(first_view) + sys.getsizeof(second_view))


def test_owned_payload_excludes_explicit_borrowed_root_and_descendants():
    borrowed = {"feed": np.arange(128, dtype=np.int32), "labels": ["A", "B"]}
    owned = {"cells": {"1": [21, 21]}}
    root = {"borrowed": borrowed, "owned": owned}

    full = sr._owned_payload_nbytes(root)
    excluded = sr._owned_payload_nbytes(root, borrowed_root_ids={id(borrowed)})
    assert 0 <= excluded < full
    assert excluded >= sr._owned_payload_nbytes(owned)


def test_nested_geometry_and_mc_payloads_include_shared_structures_once():
    geometry = [{
        "mode": "ride",
        "name": "example service",
        "pts": [[37.76, -122.43], [37.77, -122.42]],
    }]
    geom_cache = sr.BoundedCellCache(8)
    geom_cache[1] = {"total": 24, "geom": geometry}
    alt_cache = sr.BoundedCellCache(8)
    alt_cache[1] = [{"total": 25, "geom": geometry}]
    typical_cache = sr.BoundedCellCache(8)
    typical_cache[1] = {"sig": ("example",), "out": [{"typical": 25, "frag": 4}]}

    tree_entry = {"cells": {"1": [24, 24]}, "geom": geom_cache}
    mc_entry = {
        "realistic": {"1": 24},
        "variance": {"1": {"frag": 4, "std": 2, "stuck": 0.0}},
        "alt_geom": alt_cache,
        "typ": typical_cache,
    }
    combined = sr._owned_payload_nbytes((tree_entry, mc_entry))
    separate = (sr._owned_payload_nbytes(tree_entry)
                + sr._owned_payload_nbytes(mc_entry))

    assert combined > 0
    assert combined < separate       # the aliased geometry list is de-duplicated across roots
    assert sr._owned_payload_nbytes(geom_cache) > 0
    assert sr._owned_payload_nbytes(alt_cache) > 0
    assert sr._owned_payload_nbytes(typical_cache) > 0


def test_combined_request_owned_roots_dedupe_across_cache_categories():
    """A single retained-memory total must not add aliases from separate cache roots twice."""
    shared = {"points": np.arange(48, dtype=np.float64)}
    tree_root = {"tree": shared, "cells": {"7": [24, 25]}}
    weighted_root = {"workplace_path": shared}
    result_root = {"result": {"7": {"total": 25}}}
    legacy_root = {"itinerary": {"7": {"legs": [shared]}}}

    combined = sr._owned_payload_nbytes(
        (tree_root, weighted_root, result_root, legacy_root))
    separately_measured = sum(sr._owned_payload_nbytes(root) for root in (
        tree_root, weighted_root, result_root, legacy_root))

    assert combined > 0
    assert combined < separately_measured


def test_slotted_payload_and_invalid_size_never_produce_negative_totals():
    class ScenarioLike:
        __slots__ = ("profile", "data")

        def __init__(self):
            self.profile = np.arange(12, dtype=np.float64)
            self.data = {"static": np.arange(64, dtype=np.int16)}

    class InvalidSize:
        def __sizeof__(self):
            return -1

    scenario = ScenarioLike()
    measured = sr._owned_payload_nbytes(
        scenario, borrowed_root_ids={id(scenario.data)})
    assert measured >= sr._owned_payload_nbytes(scenario.profile)
    assert sr._owned_payload_nbytes(InvalidSize()) >= 0
    assert sr._owned_payload_nbytes(None) >= 0
