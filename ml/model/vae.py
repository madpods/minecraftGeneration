"""
ConditionalChunkVAE — top-level model composing all sub-modules.

Also contains:
  - ConditionBuilder: converts a building request into a condition vector
  - vae_loss: multi-component ELBO loss with weighted reconstruction terms
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block_state import (
    BLOCK_EMBED_DIM,
    CHUNK_SIZE,
    COND_DIM,
    LATENT_DIM,
    BlockStateEncoder,
)
from .decoder import Decoder
from .encoder import Encoder


# ---------------------------------------------------------------------------
# Face / purpose / tile-type index counts
# ---------------------------------------------------------------------------

NUM_TILE_TYPES: int = 20
NUM_PURPOSES:   int = 30
NUM_FACE_TYPES: int = 4   # OPEN, DOOR, WALL, NONE  (FLOOR handled as WALL variant)


# ---------------------------------------------------------------------------
# ConditionBuilder
# ---------------------------------------------------------------------------

class ConditionBuilder(nn.Module):
    """
    Converts a structured building request into a fixed-size condition vector
    of dimension COND_DIM=209.

    Raw component dimensions (before projection):
        tile_type   → 32
        purpose     → 32
        face_n/s/e/w/u/d → 8 each × 6 = 48
        size        → 16
        variance    → 8
        faction_vec → 128  (passed through directly)
        ─────────────────
        total raw   → 264  → Linear(264, 209)
    """

    _RAW_DIM: int = 32 + 32 + 8 * 6 + 16 + 8 + LATENT_DIM  # 264

    def __init__(self, cond_dim: int = COND_DIM) -> None:
        super().__init__()

        self.tile_type_embed = nn.Embedding(NUM_TILE_TYPES, 32)
        self.purpose_embed   = nn.Embedding(NUM_PURPOSES,   32)

        # Six face directions: north, south, east, west, up, down
        self.face_n_embed = nn.Embedding(NUM_FACE_TYPES, 8)
        self.face_s_embed = nn.Embedding(NUM_FACE_TYPES, 8)
        self.face_e_embed = nn.Embedding(NUM_FACE_TYPES, 8)
        self.face_w_embed = nn.Embedding(NUM_FACE_TYPES, 8)
        self.face_u_embed = nn.Embedding(NUM_FACE_TYPES, 8)
        self.face_d_embed = nn.Embedding(NUM_FACE_TYPES, 8)

        self.size_proj     = nn.Linear(1, 16)
        self.variance_proj = nn.Linear(1, 8)

        # Project raw concatenation down to COND_DIM
        self.proj = nn.Linear(self._RAW_DIM, cond_dim)

    def build_condition(
        self,
        tile_type:      torch.Tensor,   # [B]  int
        purpose:        torch.Tensor,   # [B]  int
        face_n:         torch.Tensor,   # [B]  int
        face_s:         torch.Tensor,   # [B]  int
        face_e:         torch.Tensor,   # [B]  int
        face_w:         torch.Tensor,   # [B]  int
        face_u:         torch.Tensor,   # [B]  int
        face_d:         torch.Tensor,   # [B]  int
        size:           torch.Tensor,   # [B]  float
        variance:       torch.Tensor,   # [B]  float
        faction_vector: torch.Tensor,   # [B, LATENT_DIM]
    ) -> torch.Tensor:
        """Returns condition vector of shape [B, COND_DIM]."""

        parts = [
            self.tile_type_embed(tile_type),                # [B, 32]
            self.purpose_embed(purpose),                    # [B, 32]
            self.face_n_embed(face_n),                      # [B, 8]
            self.face_s_embed(face_s),
            self.face_e_embed(face_e),
            self.face_w_embed(face_w),
            self.face_u_embed(face_u),
            self.face_d_embed(face_d),
            self.size_proj(size.unsqueeze(-1)),             # [B, 16]
            self.variance_proj(variance.unsqueeze(-1)),     # [B, 8]
            faction_vector,                                 # [B, 128]
        ]

        raw = torch.cat(parts, dim=-1)   # [B, 264]
        return self.proj(raw)            # [B, COND_DIM]

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Alias for build_condition to allow use as a Module."""
        return self.build_condition(*args, **kwargs)


# ---------------------------------------------------------------------------
# ConditionalChunkVAE
# ---------------------------------------------------------------------------

class ConditionalChunkVAE(nn.Module):
    """
    Full conditional VAE for 16×16×16 Minecraft chunk generation.

    Composed of:
        BlockStateEncoder  — integer block states → dense feature volume
        Encoder            — feature volume + condition → (mu, logvar)
        Decoder            — latent + condition → per-component logits
        ConditionBuilder   — building request → condition vector
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        cond_dim:   int = COND_DIM,
    ) -> None:
        super().__init__()

        self.state_encoder    = BlockStateEncoder()
        self.encoder          = Encoder(latent_dim=latent_dim, cond_dim=cond_dim)
        self.decoder          = Decoder(latent_dim=latent_dim, cond_dim=cond_dim)
        self.condition_builder = ConditionBuilder(cond_dim=cond_dim)

        self.latent_dim = latent_dim
        self.cond_dim   = cond_dim

    def reparameterize(
        self,
        mu:     torch.Tensor,   # [B, LATENT_DIM]
        logvar: torch.Tensor,   # [B, LATENT_DIM]
    ) -> torch.Tensor:
        """Returns a sample from N(mu, exp(0.5*logvar)) during training, or mu during eval."""
        if not self.training:
            return mu
        sigma = torch.exp(0.5 * logvar)
        eps   = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(
        self,
        block_ids:  torch.Tensor,   # [B, 16, 16, 16]  int
        facing:     torch.Tensor,   # [B, 16, 16, 16]  int
        axis:       torch.Tensor,   # [B, 16, 16, 16]  int
        half:       torch.Tensor,   # [B, 16, 16, 16]  int
        shape:      torch.Tensor,   # [B, 16, 16, 16]  int
        condition:  torch.Tensor,   # [B, COND_DIM]    float
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Returns:
            outputs — dict of logit tensors {block_id, facing, axis, half, shape}
            mu      — [B, LATENT_DIM]
            logvar  — [B, LATENT_DIM]
        """
        x          = self.state_encoder(block_ids, facing, axis, half, shape)
        mu, logvar = self.encoder(x, condition)
        z          = self.reparameterize(mu, logvar)
        z_c        = torch.cat([z, condition], dim=-1)
        outputs    = self.decoder(z_c)
        return outputs, mu, logvar

    @torch.no_grad()
    def generate(
        self,
        condition:  torch.Tensor,   # [B, COND_DIM]
        faction_z:  torch.Tensor,   # [B, LATENT_DIM]  — sampled build vector from FactionIdentity
    ) -> dict[str, torch.Tensor]:
        """
        Generate a chunk schematic from a faction latent vector + condition.
        Uses the faction_z directly (no encoder forward pass).

        Returns dict of argmax-decoded integer tensors [B, 16, 16, 16].
        """
        self.eval()
        z_c     = torch.cat([faction_z, condition], dim=-1)
        logits  = self.decoder(z_c)

        return {k: v.argmax(dim=1) for k, v in logits.items()}


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def vae_loss(
    outputs:   dict[str, torch.Tensor],   # raw logits from decoder
    targets:   dict[str, torch.Tensor],   # integer ground truth [B, 16, 16, 16]
    mu:        torch.Tensor,              # [B, LATENT_DIM]
    logvar:    torch.Tensor,              # [B, LATENT_DIM]
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Multi-component ELBO loss.

    Reconstruction terms (cross-entropy per component, weighted):
        block_id × 1.0  — wrong block type is worst
        facing   × 0.5
        axis     × 0.4
        half     × 0.3
        shape    × 0.3

    KL term: normalized by batch size, scaled by kl_weight (annealed during training).

    Returns:
        total_loss  — scalar
        recon_loss  — scalar (unweighted sum for logging)
        kl_loss     — scalar (unnormalized for logging)
    """
    recon_block  = F.cross_entropy(outputs["block_id"], targets["block_id"])
    recon_facing = F.cross_entropy(outputs["facing"],   targets["facing"])
    recon_axis   = F.cross_entropy(outputs["axis"],     targets["axis"])
    recon_half   = F.cross_entropy(outputs["half"],     targets["half"])
    recon_shape  = F.cross_entropy(outputs["shape"],    targets["shape"])

    recon = (
        recon_block  * 1.0
        + recon_facing * 0.5
        + recon_axis   * 0.4
        + recon_half   * 0.3
        + recon_shape  * 0.3
    )

    # KL divergence: -0.5 * sum(1 + logvar - mu² - exp(logvar))
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]

    total = recon + kl_weight * kl
    return total, recon, kl
