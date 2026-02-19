"""Unit tests for SCO2SurrogateFNO (PhysicsNeMo-backed FNO).

Tests cover:
  - PhysicsNeMo import succeeds (requires physicsnemo or modulus package)
  - FNO construction with valid config
  - forward() tensor shapes
  - predict_next_state() interface
  - Config-driven construction using fno_surrogate.yaml values

These tests are skipped automatically if physicsnemo/modulus or torch is not
installed. They must pass in the full Docker image where both are installed.
"""
from __future__ import annotations

import importlib
import pytest

try:
    import torch as _torch
    _torch_available = True
except ImportError:
    _torch_available = False


def _physicsnemo_available():
    candidates = [
        "physicsnemo.models.fno",
        "modulus.models.fno.fno",
        "modulus.models.fno",
    ]
    for pkg in candidates:
        try:
            importlib.import_module(pkg)
            return True
        except (ImportError, AttributeError):
            pass
    return False


requires_physicsnemo = pytest.mark.skipif(
    not (_torch_available and _physicsnemo_available()),
    reason="physicsnemo (or modulus) and torch are required",
)

SMALL_CONFIG = {
    "input_dim":   6,
    "output_dim":  4,
    "modes":       4,
    "width":      16,
    "n_layers":    2,
    "padding":     4,
}

DEFAULT_CONFIG = {
    "input_dim":  18,
    "output_dim": 14,
    "modes":      16,
    "width":      64,
    "n_layers":    4,
    "padding":     8,
}


class TestPhysicsNeMoImport:
    @requires_physicsnemo
    def test_import_succeeds(self):
        from sco2rl.surrogate.fno_model import SCO2SurrogateFNO
        assert SCO2SurrogateFNO is not None

    @requires_physicsnemo
    def test_construction_succeeds(self):
        import torch
        from sco2rl.surrogate.fno_model import SCO2SurrogateFNO
        model = SCO2SurrogateFNO(config=SMALL_CONFIG)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    @requires_physicsnemo
    def test_fno_attribute_exists(self):
        import torch
        from sco2rl.surrogate.fno_model import SCO2SurrogateFNO
        model = SCO2SurrogateFNO(config=SMALL_CONFIG)
        assert hasattr(model, "fno")
        assert isinstance(model.fno, torch.nn.Module)


class TestForwardPass:
    @requires_physicsnemo
    def test_forward_output_shape(self):
        import torch
        from sco2rl.surrogate.fno_model import SCO2SurrogateFNO
        model = SCO2SurrogateFNO(config=SMALL_CONFIG)
        model.eval()
        B, T = 4, 200
        x = torch.randn(B, SMALL_CONFIG["input_dim"], T)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, SMALL_CONFIG["output_dim"], T)

    @requires_physicsnemo
    def test_forward_T1_single_step(self):
        import torch
        from sco2rl.surrogate.fno_model import SCO2SurrogateFNO
        model = SCO2SurrogateFNO(config=SMALL_CONFIG)
        model.eval()
        x = torch.randn(4, SMALL_CONFIG["input_dim"], 1)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, SMALL_CONFIG["output_dim"], 1)


class TestPredictNextState:
    @requires_physicsnemo
    def test_predict_next_state_shape(self):
        import torch
        from sco2rl.surrogate.fno_model import SCO2SurrogateFNO
        model = SCO2SurrogateFNO(config=SMALL_CONFIG)
        model.eval()
        n_act = SMALL_CONFIG["input_dim"] - SMALL_CONFIG["output_dim"]
        B = 8
        state = torch.randn(B, SMALL_CONFIG["output_dim"])
        action = torch.randn(B, n_act)
        with torch.no_grad():
            next_state = model.predict_next_state(state, action)
        assert next_state.shape == (B, SMALL_CONFIG["output_dim"])

    @requires_physicsnemo
    def test_predict_next_state_deterministic(self):
        import torch
        from sco2rl.surrogate.fno_model import SCO2SurrogateFNO
        model = SCO2SurrogateFNO(config=SMALL_CONFIG)
        model.eval()
        n_act = SMALL_CONFIG["input_dim"] - SMALL_CONFIG["output_dim"]
        state = torch.randn(4, SMALL_CONFIG["output_dim"])
        action = torch.randn(4, n_act)
        with torch.no_grad():
            out1 = model.predict_next_state(state, action)
            out2 = model.predict_next_state(state, action)
        torch.testing.assert_close(out1, out2)

    @requires_physicsnemo
    def test_predict_next_state_gradients_flow(self):
        import torch
        from sco2rl.surrogate.fno_model import SCO2SurrogateFNO
        model = SCO2SurrogateFNO(config=SMALL_CONFIG)
        model.train()
        n_act = SMALL_CONFIG["input_dim"] - SMALL_CONFIG["output_dim"]
        state = torch.randn(4, SMALL_CONFIG["output_dim"])
        action = torch.randn(4, n_act)
        out = model.predict_next_state(state, action)
        out.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


class TestProductionConfig:
    @requires_physicsnemo
    def test_production_config_construction(self):
        from sco2rl.surrogate.fno_model import SCO2SurrogateFNO
        model = SCO2SurrogateFNO(config=DEFAULT_CONFIG)
        assert model is not None

    @requires_physicsnemo
    def test_production_forward_shape(self):
        import torch
        from sco2rl.surrogate.fno_model import SCO2SurrogateFNO
        model = SCO2SurrogateFNO(config=DEFAULT_CONFIG)
        model.eval()
        x = torch.randn(2, 18, 200)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 14, 200)

    @requires_physicsnemo
    def test_latent_channels_alias(self):
        from sco2rl.surrogate.fno_model import SCO2SurrogateFNO
        cfg = {
            "input_dim": 6, "output_dim": 4, "modes": 4,
            "latent_channels": 16, "n_layers": 2, "padding": 4,
        }
        model = SCO2SurrogateFNO(config=cfg)
        assert model is not None


class TestBackwardCompatibility:
    @requires_physicsnemo
    def test_fno1d_alias_deprecated(self):
        import torch
        from sco2rl.surrogate.fno_model import FNO1d
        with pytest.warns(DeprecationWarning):
            model = FNO1d(modes=4, width=16, n_layers=2, input_dim=6, output_dim=4, padding=4)
        assert isinstance(model, torch.nn.Module)

    def test_fnoblock_removed(self):
        from sco2rl.surrogate.fno_model import FNOBlock
        with pytest.raises(ImportError):
            FNOBlock()

    def test_spectralconv1d_removed(self):
        from sco2rl.surrogate.fno_model import SpectralConv1d
        with pytest.raises(ImportError):
            SpectralConv1d(4, 4, 4)
