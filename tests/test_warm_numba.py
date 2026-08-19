"""Topology-independent coverage for deployment's Numba warmup gate."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _warm_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "warm_numba.py"
    spec = importlib.util.spec_from_file_location("warm_numba_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_one_tail_warmup_fixture_needs_no_service_or_route_topology():
    """Its ABI fixture stays valid even when the public warmup pin has no transfers."""
    warm = _warm_module()
    calls = []

    def dispatcher(*args):
        calls.append(args)
        return np.empty((0, 12), dtype=np.int64)

    warm._warm_planned_one_tail_discovery(dispatcher)

    assert len(calls) == 1
    args = calls[0]
    assert len(args) == 19
    assert all(arg.flags.c_contiguous for arg in args[:14])
    assert [arg.dtype for arg in args[:14]] == [
        np.dtype(np.int32), np.dtype(np.int32), np.dtype(np.int64), np.dtype(np.int64),
        np.dtype(np.int32), np.dtype(np.int32), np.dtype(np.int32), np.dtype(np.int64),
        np.dtype(np.int32), np.dtype(np.int32), np.dtype(np.int64), np.dtype(np.int32),
        np.dtype(np.int32), np.dtype(np.int64),
    ]
    # There is a scheduled first leg but no route occurrence at its only alight, which proves
    # this is a synthetic ABI fixture rather than a discovered service topology.
    np.testing.assert_array_equal(args[7], np.array([0, 1, 1], dtype=np.int64))
    np.testing.assert_array_equal(args[10], np.array([0, 0, 0], dtype=np.int64))


def test_one_tail_warmup_compiles_dispatcher_without_booting_server():
    """The required dispatcher's signature is populated from the synthetic fixture alone."""
    warm = _warm_module()
    from core.raptor_planned_numba import discover_one_tail_variants

    warm._warm_planned_one_tail_discovery(discover_one_tail_variants)

    assert discover_one_tail_variants.signatures
