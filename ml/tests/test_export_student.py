"""Tests for export_student.py — StudentMLP architecture.

export_student.py is mostly I/O orchestration (load checkpoint, write ONNX,
write TFLite, verify), but the StudentMLP class is a pure, testable unit:
we can instantiate it with known dimensions and verify forward-pass behavior
without any saved model files.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed — skipping StudentMLP tests")

from ml.export_student import StudentMLP


class TestStudentMLP:

    @pytest.fixture
    def small_model(self):
        """A tiny StudentMLP for fast tests."""
        return StudentMLP(input_dim=10, hidden_dims=[16, 8])

    def test_forward_output_shape_single(self, small_model):
        """Forward pass on a single sample should return shape (1,) after squeeze."""
        small_model.eval()
        x = torch.randn(1, 10)
        with torch.no_grad():
            out = small_model(x)
        assert out.shape == (1,)

    def test_forward_output_shape_batch(self, small_model):
        """Forward pass on a batch should return shape (batch_size,)."""
        small_model.eval()
        x = torch.randn(32, 10)
        with torch.no_grad():
            out = small_model(x)
        assert out.shape == (32,)

    def test_output_is_raw_logit(self, small_model):
        """Output should be an unbounded logit (no sigmoid applied), so it
        can be negative or > 1."""
        small_model.eval()
        torch.manual_seed(0)
        x = torch.randn(100, 10) * 10  # wide range input
        with torch.no_grad():
            out = small_model(x)
        # With random weights and large inputs, logits should span negative and positive
        assert out.min().item() < 0 or out.max().item() > 1  # at least one of these should hold

    def test_deterministic_in_eval_mode(self, small_model):
        """In eval mode, Dropout is off, BatchNorm uses running stats →
        same input should give identical output across calls."""
        small_model.eval()
        x = torch.randn(5, 10)
        with torch.no_grad():
            out1 = small_model(x)
            out2 = small_model(x)
        torch.testing.assert_close(out1, out2)

    def test_different_hidden_dims(self):
        """Verify that various hidden_dims configurations instantiate and run."""
        for dims in [[32], [64, 32], [128, 64, 32], [256, 128, 64, 32]]:
            model = StudentMLP(input_dim=20, hidden_dims=dims)
            model.eval()
            x = torch.randn(4, 20)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (4,), f"Failed for hidden_dims={dims}"

    def test_parameter_count_scales_with_architecture(self):
        """A deeper/wider model should have strictly more parameters."""
        small = StudentMLP(input_dim=10, hidden_dims=[8])
        large = StudentMLP(input_dim=10, hidden_dims=[64, 32])
        n_small = sum(p.numel() for p in small.parameters())
        n_large = sum(p.numel() for p in large.parameters())
        assert n_large > n_small

    def test_gradient_flows(self):
        """Verify backprop works (no dead layers, no accidental detach)."""
        model = StudentMLP(input_dim=5, hidden_dims=[8, 4])
        model.train()
        x = torch.randn(8, 5)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_batchnorm_layers_present(self, small_model):
        """StudentMLP should contain BatchNorm1d layers per hidden dim."""
        bn_layers = [m for m in small_model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        assert len(bn_layers) == 2  # one per hidden dim [16, 8]

    def test_dropout_layers_present(self, small_model):
        """StudentMLP should contain Dropout layers per hidden dim."""
        do_layers = [m for m in small_model.modules() if isinstance(m, torch.nn.Dropout)]
        assert len(do_layers) == 2
