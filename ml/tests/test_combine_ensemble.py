"""Tests for combine_ensemble.py — pure function: percentile_normalize."""

import numpy as np
import pytest
from ml.combine_ensemble import percentile_normalize


class TestPercentileNormalize:
    """percentile_normalize maps raw scores to [0, 1] via rank-based normalization."""

    def test_output_range_is_zero_to_one(self):
        x = np.array([100.0, 1.0, 50.0, 75.0, 25.0])
        result = percentile_normalize(x)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_monotonic_with_sorted_input(self):
        """Larger raw values should map to larger normalized values."""
        x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = percentile_normalize(x)
        assert all(result[i] < result[i + 1] for i in range(len(result) - 1))

    def test_monotonic_with_unsorted_input(self):
        x = np.array([50.0, 10.0, 40.0, 20.0, 30.0])
        result = percentile_normalize(x)
        order = np.argsort(x)
        sorted_result = result[order]
        assert all(sorted_result[i] < sorted_result[i + 1] for i in range(len(sorted_result) - 1))

    def test_tied_values_get_same_rank(self):
        x = np.array([1.0, 2.0, 2.0, 3.0])
        result = percentile_normalize(x)
        assert result[1] == pytest.approx(result[2])

    def test_two_elements(self):
        x = np.array([5.0, 10.0])
        result = percentile_normalize(x)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(1.0)

    def test_robust_to_outliers(self):
        """A single extreme outlier shouldn't compress the rest of the range
        (unlike min-max normalization) — that's the point of rank-based."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 1e9])
        result = percentile_normalize(x)
        # The four non-outlier values should be spread out, not compressed near 0
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.25)
        assert result[2] == pytest.approx(0.5)
        assert result[3] == pytest.approx(0.75)
        assert result[4] == pytest.approx(1.0)

    def test_negative_values_handled(self):
        x = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])
        result = percentile_normalize(x)
        assert result[0] == pytest.approx(0.0)
        assert result[4] == pytest.approx(1.0)

    def test_identical_values(self):
        """All same value → all same rank → all same normalized score."""
        x = np.array([7.0, 7.0, 7.0, 7.0])
        result = percentile_normalize(x)
        assert len(set(result)) == 1  # all identical
