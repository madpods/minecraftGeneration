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
    NUM_STYLES,
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
        style_embed        — learned style vector per style id (e.g. village
                             biome), fed to the ConditionBuilder's faction
                             slot for style-conditioned training
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
        self.style_embed      = nn.Embedding(NUM_STYLES, latent_dim)

        self.latent_dim = latent_dim
        self.cond_dim   = cond_dim

    # ------------------------------------------------------------------
    # Construction / conditioning helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        ckpt: dict[str, Any],
        map_location: str | torch.device = "cpu",
    ) -> "ConditionalChunkVAE":
        """
        Build a model from a checkpoint dict (as saved by ml.train).

        Checkpoints from before style conditioning have no style_embed
        weights; those are left at their initial values (they are unused by
        an unconditional model).
        """
        model = cls().to(map_location)
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        if unexpected or any(k != "style_embed.weight" for k in missing):
            raise RuntimeError(f"Checkpoint mismatch: missing={missing} unexpected={unexpected}")
        return model

    def style_condition(self, style_vectors: torch.Tensor) -> torch.Tensor:
        """
        Condition vector for style-conditioned generation.

        style_vectors: [B, LATENT_DIM] — a row of style_embed for a known
        style, or any mix of them (e.g. 0.5 * desert + 0.5 * taiga).
        The other ConditionBuilder inputs are held constant because the
        training data has no labels for them yet.
        """
        B, dev = style_vectors.shape[0], style_vectors.device
        zeros = torch.zeros(B, dtype=torch.long, device=dev)
        return self.condition_builder.build_condition(
            tile_type=zeros, purpose=zeros,
            face_n=zeros, face_s=zeros, face_e=zeros,
            face_w=zeros, face_u=zeros, face_d=zeros,
            size=torch.full((B,), 0.5, device=dev),
            variance=torch.zeros(B, device=dev),
            faction_vector=style_vectors,
        )

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
        occ_threshold: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Generate a chunk schematic from a faction latent vector + condition.
        Uses the faction_z directly (no encoder forward pass).

        occ_threshold: if given, a voxel is solid when P(not air) exceeds it
        (see decode_block_ids); None uses plain argmax.

        Returns dict of decoded integer tensors [B, 16, 16, 16].
        """
        self.eval()
        z_c     = torch.cat([faction_z, condition], dim=-1)
        logits  = self.decoder(z_c)

        out = {k: v.argmax(dim=1) for k, v in logits.items()}
        out["block_id"] = decode_block_ids(logits["block_id"], occ_threshold)
        return out


# ---------------------------------------------------------------------------
# Occupancy view of the block logits
# ---------------------------------------------------------------------------

AIR_ID: int = 0


def occupancy_logit(block_logits: torch.Tensor) -> torch.Tensor:
    """
    Log-odds that a voxel is NOT air, derived from the block_id logits.

    P(solid) = sum_{k != air} softmax_k, so
    logit(P(solid)) = logsumexp(logits[non-air]) - logits[air].
    [B, K, X, Y, Z] -> [B, X, Y, Z]
    """
    lf = block_logits.float()
    return torch.logsumexp(lf[:, AIR_ID + 1:], dim=1) - lf[:, AIR_ID]


def decode_block_ids(
    block_logits: torch.Tensor,
    occ_threshold: float | None = None,
) -> torch.Tensor:
    """
    Turn block logits into block ids.

    occ_threshold=None: plain argmax over all classes.
    occ_threshold=t:    two-step decode — a voxel is solid if P(solid) > t,
                        and solid voxels take the most likely non-air block.
                        t is a density dial: lower = more blocks.
    """
    if occ_threshold is None:
        return block_logits.argmax(dim=1)
    solid = torch.sigmoid(occupancy_logit(block_logits)) > occ_threshold
    types = block_logits[:, AIR_ID + 1:].argmax(dim=1) + AIR_ID + 1
    return torch.where(solid, types, torch.full_like(types, AIR_ID))


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def _ce_per_sample(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Cross-entropy summed over all voxels, averaged over the batch.

    Summing over voxels (instead of F.cross_entropy's default mean over
    B*16^3 positions) keeps the reconstruction term on the same per-sample
    scale as the KL term, which is a sum over latent dims per sample. With
    the mean, KL was ~4096x too strong and pushed toward posterior collapse.

    `weight` is an optional per-class weight vector (e.g. to down-weight air).
    """
    return F.cross_entropy(
        logits.float(), target.long(), weight=weight, reduction="sum"
    ) / logits.shape[0]


def _ce_masked(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy summed over voxels where mask is True, averaged over the batch."""
    ce = F.cross_entropy(logits.float(), target.long(), reduction="none")
    return (ce * mask).sum() / logits.shape[0]


def _occupancy_loss(
    block_logits: torch.Tensor,
    solid: torch.Tensor,
    pos_weight: float,
    focal_gamma: float,
) -> torch.Tensor:
    """Binary block-vs-air loss on the occupancy logit, summed over voxels / batch."""
    logit = occupancy_logit(block_logits)
    target = solid.float()
    bce = F.binary_cross_entropy_with_logits(
        logit, target,
        pos_weight=torch.tensor(pos_weight, device=logit.device),
        reduction="none",
    )
    if focal_gamma > 0:
        # Focal loss: scale each voxel by (1 - p_correct)^gamma so the easy,
        # confidently-empty space far from the building stops dominating.
        p = torch.sigmoid(logit)
        p_correct = torch.where(solid, p, 1 - p)
        bce = bce * (1 - p_correct).pow(focal_gamma)
    return bce.sum() / block_logits.shape[0]


def vae_loss(
    outputs:   dict[str, torch.Tensor],   # raw logits from decoder
    targets:   dict[str, torch.Tensor],   # integer ground truth [B, 16, 16, 16]
    mu:        torch.Tensor,              # [B, LATENT_DIM]
    logvar:    torch.Tensor,              # [B, LATENT_DIM]
    kl_weight: float = 1.0,
    block_class_weights: torch.Tensor | None = None,   # [NUM_BLOCK_TYPES]
    mode:           str   = "joint",
    occ_pos_weight: float = 1.0,
    focal_gamma:    float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Multi-component ELBO loss.

    Reconstruction terms (cross-entropy per component, weighted):
        block_id × 1.0  — wrong block type is worst
        facing   × 0.5
        axis     × 0.4
        half     × 0.3
        shape    × 0.3

    Each reconstruction term is summed over the 16^3 voxels and averaged over
    the batch (per-sample nats), matching the KL term's scale.

    KL term: summed over latent dims, averaged over the batch, scaled by
    kl_weight (annealed during training).

    block_class_weights optionally re-weights the block_id cross-entropy per
    class. Village houses are ~95% air, so down-weighting air (index 0) keeps
    the model from settling on "predict air everywhere".

    mode="split" factorises the block term into two questions:
        occupancy — is this voxel a block or air? Binary cross-entropy on
                    occupancy_logit(), with `occ_pos_weight` on solid voxels
                    and optional focal scaling (`focal_gamma`).
        type      — which block? Cross-entropy over non-air classes, counted
                    only on voxels that really are solid.
    The facing/axis/half/shape terms are also counted only on solid voxels
    (on air they would just be learning "none"). block_class_weights is
    ignored in split mode.

    Returns:
        total_loss  — scalar
        recon_loss  — scalar (unweighted sum for logging)
        kl_loss     — scalar (unnormalized for logging)
    """
    if mode == "joint":
        recon_block  = _ce_per_sample(outputs["block_id"], targets["block_id"], block_class_weights)
        recon_facing = _ce_per_sample(outputs["facing"], targets["facing"])
        recon_axis   = _ce_per_sample(outputs["axis"], targets["axis"])
        recon_half   = _ce_per_sample(outputs["half"], targets["half"])
        recon_shape  = _ce_per_sample(outputs["shape"], targets["shape"])
    elif mode == "split":
        block_t = targets["block_id"].long()
        solid   = block_t != AIR_ID
        recon_block = (
            _occupancy_loss(outputs["block_id"], solid, occ_pos_weight, focal_gamma)
            + _ce_masked(outputs["block_id"][:, AIR_ID + 1:], (block_t - AIR_ID - 1).clamp(min=0), solid)
        )
        recon_facing = _ce_masked(outputs["facing"], targets["facing"], solid)
        recon_axis   = _ce_masked(outputs["axis"],   targets["axis"],   solid)
        recon_half   = _ce_masked(outputs["half"],   targets["half"],   solid)
        recon_shape  = _ce_masked(outputs["shape"],  targets["shape"],  solid)
    else:
        raise ValueError(f"mode must be 'joint' or 'split', got {mode!r}")

    recon = (
        recon_block  * 1.0
        + recon_facing * 0.5
        + recon_axis   * 0.4
        + recon_half   * 0.3
        + recon_shape  * 0.3
    )

    # KL divergence: -0.5 * sum(1 + logvar - mu² - exp(logvar))
    mu, logvar = mu.float(), logvar.float()
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]

    total = recon + kl_weight * kl
    return total, recon, kl
