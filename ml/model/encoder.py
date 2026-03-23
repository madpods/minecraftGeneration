"""
3D convolutional encoder for the ConditionalChunkVAE.

Takes a block state feature volume [B, 33, 16, 16, 16] and a condition vector
[B, COND_DIM], and outputs (mu, logvar) each of shape [B, LATENT_DIM].

Downsampling path:
    16³ → 8³ → 4³ → 2³  (three stride-2 convolutions)
    Flatten 128 × 2 × 2 × 2 = 1024
    Concat condition → Linear → mu / logvar
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .block_state import COND_DIM, LATENT_DIM, VOXEL_DIM


class Encoder(nn.Module):
    """
    Maps a voxel feature volume + condition vector to VAE latent parameters.

    Args:
        voxel_dim: Number of input channels (default VOXEL_DIM=33).
        cond_dim:  Dimension of the condition vector (default COND_DIM=209).
        latent_dim: Dimension of the latent space (default LATENT_DIM=128).
    """

    def __init__(
        self,
        voxel_dim: int = VOXEL_DIM,
        cond_dim: int = COND_DIM,
        latent_dim: int = LATENT_DIM,
    ) -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(
            # 16³ → 8³
            nn.Conv3d(voxel_dim, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            # 8³ → 4³
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            # 4³ → 2³
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        # After conv: [B, 128, 2, 2, 2] → flatten → [B, 1024]
        flat_dim: int = 128 * 2 * 2 * 2  # 1024

        self.mu_head     = nn.Linear(flat_dim + cond_dim, latent_dim)
        self.logvar_head = nn.Linear(flat_dim + cond_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,  # [B, VOXEL_DIM, 16, 16, 16]
        c: torch.Tensor,  # [B, COND_DIM]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mu     — [B, LATENT_DIM]
            logvar — [B, LATENT_DIM]
        """
        h = self.conv_layers(x)          # [B, 128, 2, 2, 2]
        h = h.flatten(start_dim=1)       # [B, 1024]
        h = torch.cat([h, c], dim=-1)    # [B, 1024 + COND_DIM]

        mu     = self.mu_head(h)         # [B, LATENT_DIM]
        logvar = self.logvar_head(h)     # [B, LATENT_DIM]

        return mu, logvar
