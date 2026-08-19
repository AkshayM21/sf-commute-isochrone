"""Exact guards for the one-sort Monte-Carlo summary seam.

These are deliberately array-only tests: they do not boot GTFS or compile numba, so failures
identify a percentile/floor contract drift rather than a routing-data issue.
"""
import numpy as np
import pytest

from core.raptor_engine import _mc_summary_from_draws


def _old_summary_oracle(draws, perfect, max_min):
    """Literal pre-optimization implementation; keep independent of the new helper."""
    realistic_raw = np.ceil(np.percentile(draws, 50, axis=1)).astype(np.int32)
    floored = np.maximum(
        draws, np.where(np.asarray(perfect) >= 0, perfect, 0).astype(np.float64)[:, None])
    p50 = np.percentile(floored, 50, axis=1)
    p90 = np.percentile(floored, 90, axis=1)
    return (realistic_raw,
            np.ceil(p50).astype(np.int32),
            np.maximum(0, np.round(p90 - p50)).astype(np.int32),
            np.round(p90).astype(np.int32),
            np.round(np.std(floored, axis=1)).astype(np.int32),
            np.mean(floored >= max_min - 1e-9, axis=1))


def _assert_summary_equal(actual, expected):
    for got, want in zip(actual, expected):
        if np.issubdtype(np.asarray(got).dtype, np.floating):
            np.testing.assert_allclose(got, want, rtol=0, atol=0, equal_nan=True)
        else:
            np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64, np.float16,
                                  np.float32, np.float64])
def test_mc_summary_one_sort_matches_literal_old_oracle_for_24_draws(dtype):
    """The usual even 24-draw case exercises both .5 median and .7 p90 interpolation."""
    rng = np.random.default_rng(20260807)
    base = rng.integers(-15, 115, size=(17, 24))
    draws = base.astype(dtype)
    if np.issubdtype(dtype, np.floating):
        draws = draws + rng.normal(0, 0.37, size=draws.shape).astype(dtype)
    # Includes the -1 unknown sentinel, a zero floor, exact cap, a high floor, and repeated floors.
    perfect = np.array([-1, 0, 1, 5, 17, 33, 75, 90, -1, 12, 3, 0, 25, 60, -1, 8, 42],
                       dtype=np.int32)
    _assert_summary_equal(_mc_summary_from_draws(draws, perfect, 75),
                          _old_summary_oracle(draws, perfect, 75))


def test_mc_summary_adversarial_floors_sentinels_and_nan_match_old_oracle():
    """Regression cases for perfect floors, cap/stuck, repeated values, and NaN propagation."""
    draws = np.array([
        [1.0] * 24,                         # a perfect floor lifts every draw
        [75.0] * 24,                        # every draw is stuck
        list(range(24)),                    # even-draw interpolation exactly between ranks
        [74.999999999] * 12 + [75.0] * 12,  # cap epsilon boundary
        [-4.0, -1.0, 0.0, 1.0] * 6,         # sentinel floor becomes zero
        [np.nan] + [11.0] * 23,             # percentile must not hide a tail-sorted NaN
        [10.0] * 23 + [91.0],               # p90 uses the upper linear rank
    ], dtype=np.float64)
    perfect = np.array([20, -1, -1, 75, -1, 0, 12], dtype=np.int32)
    with np.errstate(invalid="ignore"):
        _assert_summary_equal(_mc_summary_from_draws(draws, perfect, 75),
                              _old_summary_oracle(draws, perfect, 75))
