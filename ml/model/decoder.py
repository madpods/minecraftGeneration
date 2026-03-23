"""
3D convolutional decoder for the ConditionalChunkVAE.

Takes a concatenated latent + condition vector [B, LATENT_DIM + COND_DIM] and
produces per-voxel logits for each block state component over a 16×16×16 grid.

Upsampling path:
    Linear → reshape [B, 128, 2, 2, 2]
    2³ → 4³ → 8³ → 16³  (three stride-2 transposed convolutions)
    Five parallel 1×1 output heads (no activation)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .block_state import (
    COND_DIM,
    LATENT_DIM,
    NUM_AXIS,
    NUM_BLOCK_TYPES,
    NUM_FACING,
    NUM_HALF,
    NUM_SHAPE,
)


class Decoder(nn.Module):
    """
    Maps a latent + condition vector back to block state logits.

    Args:
        latent_dim: Dimension of the VAE latent vector (default LATENT_DIM=128).
        cond_dim:   Dimension of the condition vector (default COND_DIM=209).
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        cond_dim: int = COND_DIM,
    ) -> None:
        super().__init__()

        # Project z_c to spatial seed
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 1024),
            nn.ReLU(inplace=True),
        )

        # Upsample: 2³ → 4³ → 8³ → 16³
        self.deconv_layers = nn.Sequential(
            # 2³ → 4³
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            # 4³ → 8³
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            # 8³ → 16³
            nn.ConvTranspose3d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        # Per-component output heads (1×1×1 conv, no activation)
        self.head_block_id = nn.Conv3d(32, NUM_BLOCK_TYPES, kernel_size=1)  # → [B, 256, 16, 16, 16]
        self.head_facing   = nn.Conv3d(32, NUM_FACING,      kernel_size=1)  # → [B, 7,   16, 16, 16]
        self.head_axis     = nn.Conv3d(32, NUM_AXIS,        kernel_size=1)  # → [B, 4,   16, 16, 16]
        self.head_half     = nn.Conv3d(32, NUM_HALF,        kernel_size=1)  # → [B, 3,   16, 16, 16]
        self.head_shape    = nn.Conv3d(32, NUM_SHAPE,       kernel_size=1)  # → [B, 7,   16, 16, 16]

    def forward(
        self,
        z_c: torch.Tensor,  # [B, LATENT_DIM + COND_DIM]
    ) -> dict[str, torch.Tensor]:
        """
        Returns a dict of raw logits (no softmax / argmax applied):
            block_id — [B, NUM_BLOCK_TYPES, 16, 16, 16]
            facing   — [B, NUM_FACING,      16, 16, 16]
            axis     — [B, NUM_AXIS,        16, 16, 16]
            half     — [B, NUM_HALF,        16, 16, 16]
            shape    — [B, NUM_SHAPE,       16, 16, 16]
        """
        h = self.fc(z_c)                        # [B, 1024]
        h = h.view(h.size(0), 128, 2, 2, 2)    # [B, 128, 2, 2, 2]
        h = self.deconv_layers(h)               # [B, 32, 16, 16, 16]

        return {
            "block_id": self.head_block_id(h),
            "facing":   self.head_facing(h),
            "axis":     self.head_axis(h),
            "half":     self.head_half(h),
            "shape":    self.head_shape(h),
        }
