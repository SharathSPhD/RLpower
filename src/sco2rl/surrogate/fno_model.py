"""SCO2SurrogateFNO -- PhysicsNeMo (NVIDIA Modulus) FNO surrogate model.

Wraps the PhysicsNeMo FNO class with the predict_next_state(state, action)
interface required by SurrogateTrainer.

Architecture
------------
  Input  : (B, input_dim, T)   -- state-action sequence (T=1 for one-step prediction)
  FNO    : PhysicsNeMo FNO1d   -- Fourier spectral convolutions on the time axis
  Output : (B, output_dim, T)  -- predicted next states

For autoregressive one-step prediction:
  T=1, input_dim = n_state + n_action, output_dim = n_state
"""
from __future__ import annotations

import importlib
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


def _get_torch():
    import torch
    return torch


def _get_nn():
    import torch.nn as nn
    return nn


def _import_physicsnemo_fno():
    """Try to import the PhysicsNeMo FNO class.

    PhysicsNeMo has gone through several package renames:
      - nvidia-modulus < 0.9  -> modulus.models.fno.fno.FNO
      - physicsnemo >= 0.1    -> physicsnemo.models.fno.FNO
      - modulus alias maintained for backward compatibility in some builds

    Returns the FNO class or raises ImportError with diagnostic message.
    """
    candidates = [
        ("physicsnemo.models.fno", "FNO"),
        ("modulus.models.fno.fno", "FNO"),
        ("modulus.models.fno", "FNO"),
    ]
    for module_path, class_name in candidates:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            return cls
        except (ImportError, AttributeError):
            continue

    tried = ", ".join("{}.{}".format(m, c) for m, c in candidates)
    raise ImportError(
        "PhysicsNeMo FNO not found. Install with:\n"
        "  pip install nvidia-physicsnemo\n"
        "or inside the Docker image:\n"
        "  pip install --no-cache-dir nvidia-physicsnemo\n\n"
        "Tried: " + tried
    )


class SCO2SurrogateFNO:
    """PhysicsNeMo FNO surrogate for sCO2 state prediction.

    Parameters
    ----------
    config:
        Dict with keys matching fno_surrogate.yaml:
        - input_dim (int): state_dim + action_dim (e.g. 14 + 4 = 18)
        - output_dim (int): number of state variables predicted (e.g. 14)
        - modes (int): Fourier modes retained (default 16)
        - width or latent_channels (int): spectral latent dimension (default 64)
        - n_layers (int): number of FNO spectral layers (default 4)
        - padding (int): spectral padding (default 8)
    """

    def __init__(self, config: dict) -> None:
        torch = _get_torch()
        nn = _get_nn()
        # Call nn.Module.__init__ through the superclass mechanism
        # (we inherit dynamically to avoid top-level torch import)
        _ModuleBase = nn.Module
        _ModuleBase.__init__(self)

        ModulusFNO = _import_physicsnemo_fno()

        in_ch = int(config["input_dim"])
        out_ch = int(config["output_dim"])
        latent = int(config.get("width", config.get("latent_channels", 64)))
        modes = int(config.get("modes", 16))
        n_layers = int(config.get("n_layers", 4))
        padding = int(config.get("padding", 8))

        self.fno = ModulusFNO(
            in_channels=in_ch,
            out_channels=out_ch,
            dimension=1,
            latent_channels=latent,
            num_fno_layers=n_layers,
            num_fno_modes=modes,
            padding=padding,
            activation_fn="gelu",
            coord_features=False,
        )

        self._input_dim = in_ch
        self._output_dim = out_ch
        self._modes = modes  # needed for predict_next_state context length

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x:
            Shape (B, input_dim, T).

        Returns
        -------
        Tensor, shape (B, output_dim, T).
        """
        return self.fno(x)

    def predict_next_state(self, state, action):
        """One-step state prediction from the FNO surrogate.

        The FNO requires T >= num_fno_modes for its spectral convolution.
        When called step-by-step (T=1), we replicate the state-action pair
        across T_ctx timesteps, run the FNO, and return the last output.

        Parameters
        ----------
        state:
            Shape (B, n_state) -- current normalized observation.
        action:
            Shape (B, n_action) -- current action.

        Returns
        -------
        Tensor, shape (B, n_state) -- predicted next state.
        """
        torch = _get_torch()
        x_single = torch.cat([state, action], dim=-1).unsqueeze(-1)  # (B, C, 1)
        T_ctx = max(self._modes * 2, 32)
        x = x_single.expand(-1, -1, T_ctx).contiguous()  # (B, C, T_ctx)
        out = self.fno(x)  # (B, output_dim, T_ctx)
        return out[:, :, -1]  # (B, output_dim)

    # Make SCO2SurrogateFNO behave as nn.Module at runtime (duck-typing via __init_subclass__)
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def to(self, *args, **kwargs):
        self.fno = self.fno.to(*args, **kwargs)
        return self

    def train(self, mode=True):
        self.fno.train(mode)
        return self

    def eval(self):
        self.fno.eval()
        return self

    def parameters(self):
        return self.fno.parameters()

    def state_dict(self, *args, **kwargs):
        return self.fno.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        return self.fno.load_state_dict(state_dict, strict=strict, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _make_surrogate_fno(config: dict):
    """Factory that returns a proper nn.Module subclass wrapping PhysicsNeMo FNO."""
    import torch.nn as nn
    ModulusFNO = _import_physicsnemo_fno()

    in_ch = int(config["input_dim"])
    out_ch = int(config["output_dim"])
    latent = int(config.get("width", config.get("latent_channels", 64)))
    modes = int(config.get("modes", 16))
    n_layers = int(config.get("n_layers", 4))
    padding = int(config.get("padding", 8))

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.fno = ModulusFNO(
                in_channels=in_ch,
                out_channels=out_ch,
                dimension=1,
                latent_channels=latent,
                num_fno_layers=n_layers,
                num_fno_modes=modes,
                padding=padding,
                activation_fn="gelu",
                coord_features=False,
            )
            self._input_dim = in_ch
            self._output_dim = out_ch

        def forward(self, x):
            return self.fno(x)

        def predict_next_state(self, state, action):
            """One-step surrogate prediction.

            The FNO requires T >= num_fno_modes (default 16) for its spectral
            convolution layers.  When called step-by-step from SurrogateEnv,
            the input is (B, C, 1) which is too short.  We repeat the
            current state-action pair across T_ctx timesteps to provide the
            required context, then return the last output timestep.

            This is equivalent to assuming the system was at the current
            operating point for the last T_ctx steps — a conservative warm-up
            that avoids transient artefacts from arbitrary initial history.
            """
            import torch
            x_single = torch.cat([state, action], dim=-1).unsqueeze(-1)  # (B, C, 1)
            T_ctx = max(modes * 2, 32)  # context length >= 2*modes for spectral validity
            x = x_single.expand(-1, -1, T_ctx).contiguous()  # (B, C, T_ctx)
            out = self.fno(x)  # (B, output_dim, T_ctx)
            return out[:, :, -1]  # last predicted timestep -> (B, output_dim)

    return _Wrapper()


# Override SCO2SurrogateFNO with a proper nn.Module factory
_OrigSCO2SurrogateFNO = SCO2SurrogateFNO


class SCO2SurrogateFNO:  # noqa: F811 - intentional redefinition
    """PhysicsNeMo FNO surrogate for sCO2 state prediction (nn.Module subclass).

    Parameters
    ----------
    config:
        Dict with keys matching fno_surrogate.yaml:
        - input_dim, output_dim, modes, width or latent_channels, n_layers, padding
    """

    def __new__(cls, config: dict):
        return _make_surrogate_fno(config)


class _LegacyFNOAlias:
    """Backward-compatible alias: FNO1d redirects to SCO2SurrogateFNO."""

    def __new__(cls, modes=16, width=64, n_layers=4,
                input_dim=18, output_dim=14, activation="gelu", padding=8):
        warnings.warn(
            "FNO1d is deprecated. Use SCO2SurrogateFNO instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return SCO2SurrogateFNO(config={
            "input_dim": input_dim,
            "output_dim": output_dim,
            "modes": modes,
            "width": width,
            "n_layers": n_layers,
            "padding": padding,
        })


FNO1d = _LegacyFNOAlias


class _RemovedClass:
    def __init__(self, *args, **kwargs):
        raise ImportError(
            self.__class__.__name__ + " has been removed. "
            "Use SCO2SurrogateFNO (nvidia-physicsnemo-backed) instead."
        )


class FNOBlock(_RemovedClass):
    pass


class SpectralConv1d(_RemovedClass):
    pass
