"""Tests for monitor_student.py — pure functions: fit_bin_edges, bin_proportions, psi, sigmoid."""

import numpy as np
import pytest
from ml.monitor_student import fit_bin_edges, bin_proportions, psi, sigmoid


# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------

class TestSigmoid:

    def test_zero_gives_half(self):
        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive_saturates_to_one(self):
        assert sigmoid(100.0) == pytest.approx(1.0, abs=1e-10)

    def test_large_negative_saturates_to_zero(self):
        assert sigmoid(-100.0) == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self):
        """sigmoid(-x) = 1 - sigmoid(x)"""
        for val in [0.5, 1.0, 2.5, 5.0]:
            assert sigmoid(-val) == pytest.approx(1.0 - sigmoid(val))

    def test_vectorized(self):
        x = np.array([-1.0, 0.0, 1.0])
        result = sigmoid(x)
        assert result.shape == (3,)
        assert result[1] == pytest.approx(0.5)
        assert result[0] < 0.5 < result[2]


# ---------------------------------------------------------------------------
# fit_bin_edges
# ---------------------------------------------------------------------------

class TestFitBinEdges:

    def test_returns_inf_endpoints(self):
        vals = np.random.randn(100)
        edges = fit_bin_edges(vals, n_bins=10)
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf

    def test_enough_edges_for_binning(self):
        """Need at least 2 edges (1 bin) to form a valid histogram."""
        vals = np.random.randn(200)
        edges = fit_bin_edges(vals, n_bins=10)
        assert len(edges) >= 2

    def test_degenerate_constant_input(self):
        """A constant array should still produce usable edges (the docstring
        says it pads to avoid crashing histogram)."""
        vals = np.full(50, 7.0)
        edges = fit_bin_edges(vals, n_bins=10)
        assert len(edges) >= 2
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf

    def test_edges_are_sorted(self):
        vals = np.random.randn(500)
        edges = fit_bin_edges(vals, n_bins=5)
        assert all(edges[i] <= edges[i + 1] for i in range(len(edges) - 1))


# ---------------------------------------------------------------------------
# bin_proportions
# ---------------------------------------------------------------------------

class TestBinProportions:

    def test_proportions_sum_to_one(self):
        vals = np.random.randn(1000)
        edges = fit_bin_edges(vals, n_bins=10)
        props = bin_proportions(vals, edges)
        assert props.sum() == pytest.approx(1.0, abs=1e-3)

    def test_no_zeros_in_output(self):
        """bin_proportions clips to 1e-6 to avoid log(0) in PSI."""
        vals = np.array([1.0, 1.0, 1.0])
        # edges that create empty bins
        edges = np.array([-np.inf, 0.5, 0.9, 1.1, np.inf])
        props = bin_proportions(vals, edges)
        assert (props > 0).all()

    def test_correct_number_of_bins(self):
        vals = np.random.randn(500)
        edges = fit_bin_edges(vals, n_bins=10)
        props = bin_proportions(vals, edges)
        assert len(props) == len(edges) - 1

    def test_empty_input_does_not_crash(self):
        """Edge case: 0 samples — bin_proportions divides by max(len, 1)."""
        edges = np.array([-np.inf, 0.0, np.inf])
        props = bin_proportions(np.array([]), edges)
        assert len(props) == 2
        assert (props > 0).all()  # clipped to 1e-6


# ---------------------------------------------------------------------------
# psi
# ---------------------------------------------------------------------------

class TestPSI:

    def test_identical_distributions_give_zero(self):
        props = np.array([0.1, 0.2, 0.3, 0.4])
        assert psi(props, props) == pytest.approx(0.0)

    def test_psi_is_non_negative(self):
        ref = np.array([0.25, 0.25, 0.25, 0.25])
        cur = np.array([0.1, 0.3, 0.4, 0.2])
        assert psi(ref, cur) >= 0.0

    def test_shifted_distribution_gives_positive_psi(self):
        ref = np.array([0.5, 0.3, 0.2])
        cur = np.array([0.1, 0.3, 0.6])
        result = psi(ref, cur)
        assert result > 0.0

    def test_symmetric_property(self):
        """PSI is NOT perfectly symmetric, but swapping ref/cur should still
        produce a very similar value (both positive). Checking both are > 0."""
        ref = np.array([0.4, 0.3, 0.2, 0.1])
        cur = np.array([0.1, 0.2, 0.3, 0.4])
        assert psi(ref, cur) > 0
        assert psi(cur, ref) > 0

    def test_large_shift_exceeds_alert_threshold(self):
        """Dramatically different distributions should give PSI well above 0.25."""
        ref = np.array([0.01, 0.01, 0.01, 0.97])  # clipped versions of extreme shift
        cur = np.array([0.97, 0.01, 0.01, 0.01])
        result = psi(ref, cur)
        assert result > 0.25

    def test_end_to_end_with_numpy_data(self):
        """Generate two distributions, bin them, compute PSI — the full flow
        that main() uses, minus disk I/O."""
        rng = np.random.default_rng(42)
        ref_vals = rng.normal(0, 1, 10000)
        cur_vals = rng.normal(0, 1, 10000)  # same distribution

        edges = fit_bin_edges(ref_vals, n_bins=10)
        ref_props = bin_proportions(ref_vals, edges)
        cur_props = bin_proportions(cur_vals, edges)
        result = psi(ref_props, cur_props)
        assert result < 0.10  # same distribution → should be well below watch threshold

    def test_end_to_end_drifted(self):
        """Shifted mean should produce meaningful PSI."""
        rng = np.random.default_rng(42)
        ref_vals = rng.normal(0, 1, 10000)
        cur_vals = rng.normal(3, 1, 10000)  # shifted by 3 standard deviations

        edges = fit_bin_edges(ref_vals, n_bins=10)
        ref_props = bin_proportions(ref_vals, edges)
        cur_props = bin_proportions(cur_vals, edges)
        result = psi(ref_props, cur_props)
        assert result > 0.25  # massive shift → alert
