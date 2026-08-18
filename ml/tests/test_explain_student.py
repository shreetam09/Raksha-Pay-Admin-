"""Tests for explain_student.py — sigmoid, compute_baseline, integrated_gradients,
and occlusion_attributions.

The attribution methods need an ONNX session, so we mock one with a simple
linear model (f(x) = w·x) whose true attributions are known analytically,
letting us verify correctness without loading a real model or hitting disk.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from ml.explain_student import (
    sigmoid,
    compute_baseline,
    integrated_gradients,
    occlusion_attributions,
    forward_batch,
)


# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------

class TestSigmoid:

    def test_zero(self):
        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_positive(self):
        assert sigmoid(10.0) == pytest.approx(1.0, abs=1e-4)

    def test_negative(self):
        assert sigmoid(-10.0) == pytest.approx(0.0, abs=1e-4)

    def test_array(self):
        result = sigmoid(np.array([-1, 0, 1]))
        assert result.shape == (3,)
        assert result[0] < 0.5 < result[2]


# ---------------------------------------------------------------------------
# compute_baseline
# ---------------------------------------------------------------------------

class TestComputeBaseline:

    def test_baseline_is_mean_of_normals(self):
        X_full = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 100.0],
            "f2": [10.0, 20.0, 30.0, 999.0],
        })
        meta = pd.DataFrame({"label": [0, 0, 0, 1]})  # row 3 is fraud

        baseline = compute_baseline(X_full, meta)

        # Mean of rows 0-2 only (label==0)
        assert baseline[0] == pytest.approx(2.0)  # mean(1,2,3)
        assert baseline[1] == pytest.approx(20.0)  # mean(10,20,30)

    def test_baseline_dtype_is_float32(self):
        X_full = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        meta = pd.DataFrame({"label": [0, 0]})
        baseline = compute_baseline(X_full, meta)
        assert baseline.dtype == np.float32

    def test_baseline_excludes_fraud(self):
        """If all rows are fraud, baseline should be empty-mean (NaN or 0),
        but the more realistic check: fraud rows should NOT influence the baseline."""
        X_full = pd.DataFrame({"f1": [0.0, 0.0, 100.0]})
        meta = pd.DataFrame({"label": [0, 0, 1]})
        baseline = compute_baseline(X_full, meta)
        assert baseline[0] == pytest.approx(0.0)  # 100.0 excluded


# ---------------------------------------------------------------------------
# Mock ONNX session: f(x) = w·x  (linear model, known closed-form solution)
# ---------------------------------------------------------------------------

def _make_linear_session(weights: np.ndarray):
    """Returns a (mock_session, input_name) pair that computes f(x) = x @ w."""
    w = weights.astype(np.float32)

    def _run(output_names, input_dict):
        x = list(input_dict.values())[0].astype(np.float32)
        logits = x @ w
        return [logits.reshape(-1, 1)]

    sess = MagicMock()
    sess.run = _run
    return sess, "features"


# ---------------------------------------------------------------------------
# integrated_gradients
# ---------------------------------------------------------------------------

class TestIntegratedGradients:

    def test_completeness_property(self):
        """For IG: sum(attributions) ≈ f(x) - f(baseline).
        With a linear model this should be exact (up to finite-difference noise)."""
        w = np.array([1.0, -2.0, 0.5], dtype=np.float32)
        sess, input_name = _make_linear_session(w)

        x = np.array([3.0, 1.0, 4.0], dtype=np.float32)
        baseline = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        attr = integrated_gradients(sess, input_name, x, baseline, steps=50, eps=0.01)

        f_x = forward_batch(sess, input_name, x[None, :])[0]
        f_b = forward_batch(sess, input_name, baseline[None, :])[0]
        expected_diff = f_x - f_b

        assert attr.sum() == pytest.approx(expected_diff, abs=0.05)

    def test_attribution_sign_matches_weight_sign(self):
        """For a linear model with baseline=0, feature with positive weight
        and positive input value should get positive attribution."""
        w = np.array([2.0, -3.0], dtype=np.float32)
        sess, input_name = _make_linear_session(w)

        x = np.array([1.0, 1.0], dtype=np.float32)
        baseline = np.zeros(2, dtype=np.float32)

        attr = integrated_gradients(sess, input_name, x, baseline, steps=50, eps=0.01)

        assert attr[0] > 0  # positive weight, positive input → positive attribution
        assert attr[1] < 0  # negative weight, positive input → negative attribution

    def test_zero_diff_feature_gets_zero_attribution(self):
        """If x[i] == baseline[i], the attribution for feature i should be ~0."""
        w = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        sess, input_name = _make_linear_session(w)

        x = np.array([5.0, 0.0, 3.0], dtype=np.float32)
        baseline = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Feature 1 has x==baseline==0 → attribution should be 0
        # But let's set baseline[1] = x[1] to test the general case
        baseline_same = np.array([0.0, 5.0, 0.0], dtype=np.float32)
        x_same = np.array([5.0, 5.0, 3.0], dtype=np.float32)
        attr = integrated_gradients(sess, input_name, x_same, baseline_same, steps=50, eps=0.01)
        assert attr[1] == pytest.approx(0.0, abs=0.05)

    def test_output_shape(self):
        w = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        sess, input_name = _make_linear_session(w)
        x = np.ones(4, dtype=np.float32)
        baseline = np.zeros(4, dtype=np.float32)
        attr = integrated_gradients(sess, input_name, x, baseline, steps=10, eps=0.01)
        assert attr.shape == (4,)


# ---------------------------------------------------------------------------
# occlusion_attributions
# ---------------------------------------------------------------------------

class TestOcclusionAttributions:

    def test_output_shape(self):
        w = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sess, input_name = _make_linear_session(w)
        x = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        baseline = np.zeros(3, dtype=np.float32)
        attr = occlusion_attributions(sess, input_name, x, baseline)
        assert attr.shape == (3,)

    def test_linear_model_attributions_match_weights(self):
        """For f(x) = w·x with baseline=0 and x=[1,1,...,1],
        occluding feature i gives f(x) - f(x with x_i=0) = w_i * 1 = w_i."""
        w = np.array([1.0, -2.0, 0.5], dtype=np.float32)
        sess, input_name = _make_linear_session(w)
        x = np.ones(3, dtype=np.float32)
        baseline = np.zeros(3, dtype=np.float32)

        attr = occlusion_attributions(sess, input_name, x, baseline)
        np.testing.assert_allclose(attr, w, atol=1e-5)

    def test_zero_weight_feature_has_zero_attribution(self):
        w = np.array([3.0, 0.0, -1.0], dtype=np.float32)
        sess, input_name = _make_linear_session(w)
        x = np.ones(3, dtype=np.float32)
        baseline = np.zeros(3, dtype=np.float32)

        attr = occlusion_attributions(sess, input_name, x, baseline)
        assert attr[1] == pytest.approx(0.0, abs=1e-5)

    def test_attribution_sign(self):
        """Positive weight → positive attribution (removing the feature drops the score)."""
        w = np.array([5.0, -3.0], dtype=np.float32)
        sess, input_name = _make_linear_session(w)
        x = np.array([1.0, 1.0], dtype=np.float32)
        baseline = np.zeros(2, dtype=np.float32)

        attr = occlusion_attributions(sess, input_name, x, baseline)
        assert attr[0] > 0
        assert attr[1] < 0
