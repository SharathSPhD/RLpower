"""Unit tests for FNO1d surrogate model -- written FIRST (TDD RED phase)."""

import pytest
import torch

from sco2rl.surrogate.fno_model import FNO1d, FNOBlock, SpectralConv1d


MODES = 16
WIDTH = 64
N_LAYERS = 4
INPUT_DIM = 25
OUTPUT_DIM = 20


@pytest.fixture
def default_fno():
    return FNO1d(
        modes=MODES,
        width=WIDTH,
        n_layers=N_LAYERS,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
    )


def test_spectral_conv_output_shape():
    conv = SpectralConv1d(in_channels=16, out_channels=16, modes=8)
    x = torch.randn(2, 16, 50)
    out = conv(x)
    assert out.shape == (2, 16, 50), f"Expected (2, 16, 50), got {out.shape}"


def test_fno_block_output_shape():
    block = FNOBlock(width=64, modes=MODES)
    x = torch.randn(2, 64, 50)
    out = block(x)
    assert out.shape == (2, 64, 50), f"Expected (2, 64, 50), got {out.shape}"


def test_fno1d_forward_shape(default_fno):
    x = torch.randn(2, INPUT_DIM, 50)
    out = default_fno(x)
    assert out.shape == (2, OUTPUT_DIM, 50), f"Expected (2, 20, 50), got {out.shape}"


def test_fno1d_forward_no_nan(default_fno):
    x = torch.randn(4, INPUT_DIM, 50)
    out = default_fno(x)
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"


def test_different_inputs_different_outputs(default_fno):
    torch.manual_seed(0)
    x1 = torch.randn(2, INPUT_DIM, 50)
    x2 = torch.randn(2, INPUT_DIM, 50)
    out1 = default_fno(x1)
    out2 = default_fno(x2)
    assert not torch.allclose(out1, out2), "Different inputs produced identical outputs"


def test_fno1d_trainable(default_fno):
    x = torch.randn(2, INPUT_DIM, 50)
    out = default_fno(x)
    loss = out.sum()
    loss.backward()
    has_grad = any(p.grad is not None for p in default_fno.parameters())
    assert has_grad, "No gradients computed after backward pass"


def test_predict_next_state_shape(default_fno):
    state = torch.randn(4, OUTPUT_DIM)
    action = torch.randn(4, 5)
    next_state = default_fno.predict_next_state(state, action)
    assert next_state.shape == (4, OUTPUT_DIM), f"Expected (4, 20), got {next_state.shape}"


def test_predict_next_state_no_nan(default_fno):
    state = torch.randn(4, OUTPUT_DIM)
    action = torch.randn(4, 5)
    next_state = default_fno.predict_next_state(state, action)
    assert not torch.isnan(next_state).any(), "predict_next_state returned NaN"
    assert not torch.isinf(next_state).any(), "predict_next_state returned Inf"


def test_fno_parameter_count(default_fno):
    total = sum(p.numel() for p in default_fno.parameters() if p.requires_grad)
    assert total < 1_000_000, f"Too many parameters: {total:,} >= 1,000,000"
