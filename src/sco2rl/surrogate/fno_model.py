"""FNO1d surrogate model implementation.

Implements 1D Fourier Neural Operator for time-series state prediction.
Architecture: SpectralConv1d -> FNOBlock -> FNO1d (full network).
Reference: Li et al., "Fourier Neural Operator for Parametric PDEs" (2021).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """Complex multiplication in Fourier space -- core FNO building block.

    Performs the Fourier layer operation:
    1. FFT the input along the time axis
    2. Multiply the lowest `modes` Fourier coefficients by learnable complex weights
    3. Inverse FFT back to physical space
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Complex weights stored as real tensor of shape (in_c, out_c, modes, 2)
        # using torch.view_as_complex / view_as_real for compatibility
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def _compl_mul1d(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Batched complex matmul: (B, in_c, modes) x (in_c, out_c, modes) -> (B, out_c, modes)."""
        return torch.einsum("bim,iom->bom", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C_in, T) -> (B, C_out, T)."""
        T = x.shape[-1]

        # FFT along time axis
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C_in, T//2 + 1)

        # Retain only `modes` lowest frequencies
        modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(
            x.shape[0], self.out_channels, x_ft.shape[-1],
            dtype=torch.cfloat, device=x.device,
        )
        out_ft[:, :, :modes] = self._compl_mul1d(x_ft[:, :, :modes], self.weights[:, :, :modes])

        # Inverse FFT
        return torch.fft.irfft(out_ft, n=T, dim=-1)  # (B, C_out, T)


class FNOBlock(nn.Module):
    """One FNO layer: spectral conv + bypass pointwise conv + GELU activation."""

    def __init__(self, width: int, modes: int, activation: str = "gelu") -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.bypass = nn.Conv1d(width, width, kernel_size=1)

        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, width, T) -> (B, width, T)."""
        return self.act(self.spectral(x) + self.bypass(x))


class FNO1d(nn.Module):
    """Full FNO1d network for 1D time-series surrogate modeling.

    Maps (batch, input_dim, T) -> (batch, output_dim, T).

    Architecture:
      input_projection: input_dim -> width (linear, no activation)
      n_layers x FNOBlock(width, modes)
      output_projection: width -> output_dim (2-layer MLP)

    Used autoregressively via predict_next_state() for one-step simulation.
    """

    def __init__(
        self,
        modes: int,
        width: int,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        activation: str = "gelu",
        padding: int = 8,
    ) -> None:
        super().__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.padding = padding

        # Input projection: pointwise conv (equivalent to linear per time-step)
        self.input_proj = nn.Conv1d(input_dim, width, kernel_size=1)

        # FNO blocks
        self.blocks = nn.ModuleList(
            [FNOBlock(width, modes, activation) for _ in range(n_layers)]
        )

        # Output projection: width -> width//2 -> output_dim
        self.output_proj = nn.Sequential(
            nn.Conv1d(width, width // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width // 2, output_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, input_dim, T).

        Returns
        -------
        torch.Tensor
            Shape (B, output_dim, T).
        """
        # Project input channels to latent width
        x = self.input_proj(x)  # (B, width, T)

        # Pad to avoid aliasing at boundaries
        if self.padding > 0:
            x = F.pad(x, [0, self.padding])

        # FNO blocks
        for block in self.blocks:
            x = block(x)

        # Remove padding
        if self.padding > 0:
            x = x[..., : -self.padding]

        # Project to output
        return self.output_proj(x)  # (B, output_dim, T)

    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressive one-step prediction.

        Parameters
        ----------
        state : torch.Tensor
            Shape (B, n_obs) -- current normalized observation.
        action : torch.Tensor
            Shape (B, n_act) -- current action in physical / normalized space.

        Returns
        -------
        torch.Tensor
            Shape (B, n_obs) -- predicted next state.
        """
        # Concatenate state and action along feature axis -> (B, input_dim)
        x = torch.cat([state, action], dim=-1)  # (B, input_dim)

        # Add time dimension T=1 -> (B, input_dim, 1)
        x = x.unsqueeze(-1)

        # Forward pass -> (B, output_dim, 1)
        out = self.forward(x)

        # Remove time dimension -> (B, output_dim)
        return out.squeeze(-1)
