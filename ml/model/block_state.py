"""
Block state constants and BlockStateEncoder for the ConditionalChunkVAE.

Each Minecraft block is represented as a combination of:
  - block_id:  integer index into a vocabulary of block types
  - facing:    cardinal/vertical direction the block faces
  - axis:      orientation axis (for logs, pillars, etc.)
  - half:      top/bottom half (for slabs, stairs, etc.)
  - shape:     stair shape variant

All components are encoded separately and concatenated to form a per-voxel
representation of dimension 33 (16 + 3 + 4 + 2 + 8).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Vocabulary sizes
# ---------------------------------------------------------------------------

NUM_BLOCK_TYPES: int = 256   # base block IDs mapped to Minecraft namespaced IDs externally
NUM_FACING: int = 7          # north, south, east, west, up, down, none
NUM_AXIS: int = 4            # x, y, z, none
NUM_HALF: int = 3            # top, bottom, none
NUM_SHAPE: int = 7           # straight, inner_left, inner_right, outer_left, outer_right, none (×2 spare)

# ---------------------------------------------------------------------------
# Embedding / projection dimensions
# ---------------------------------------------------------------------------

BLOCK_EMBED_DIM: int = 16
# Facing uses a raw 3D unit vector — no learnable embedding, dim=3
FACING_DIM: int = 3
AXIS_EMBED_DIM: int = 4
HALF_EMBED_DIM: int = 2
SHAPE_EMBED_DIM: int = 8

# Combined per-voxel feature dim: 16 + 3 + 4 + 2 + 8 = 33
VOXEL_DIM: int = BLOCK_EMBED_DIM + FACING_DIM + AXIS_EMBED_DIM + HALF_EMBED_DIM + SHAPE_EMBED_DIM

# ---------------------------------------------------------------------------
# Spatial / latent dimensions
# ---------------------------------------------------------------------------

CHUNK_SIZE: int = 16
LATENT_DIM: int = 128
COND_DIM: int = 209

# ---------------------------------------------------------------------------
# Facing index mapping
# ---------------------------------------------------------------------------

# Integer indices for facing values (must match data pipeline)
FACING_INDEX: dict[str, int] = {
    "north": 0,
    "south": 1,
    "east":  2,
    "west":  3,
    "up":    4,
    "down":  5,
    "none":  6,
}

# Raw 3D unit vectors for each facing direction (no learnable parameters)
FACING_VECTORS: dict[str, list[float]] = {
    "north": [ 0.0,  0.0, -1.0],
    "south": [ 0.0,  0.0,  1.0],
    "east":  [ 1.0,  0.0,  0.0],
    "west":  [-1.0,  0.0,  0.0],
    "up":    [ 0.0,  1.0,  0.0],
    "down":  [ 0.0, -1.0,  0.0],
    "none":  [ 0.0,  0.0,  0.0],
}

# Pre-built tensor [NUM_FACING, 3] of unit vectors, indexed by FACING_INDEX
_FACING_TENSOR: torch.Tensor = torch.tensor(
    [FACING_VECTORS[k] for k in sorted(FACING_INDEX, key=lambda k: FACING_INDEX[k])],
    dtype=torch.float32,
)

# ---------------------------------------------------------------------------
# Axis index mapping
# ---------------------------------------------------------------------------

AXIS_INDEX: dict[str, int] = {
    "x":    0,
    "y":    1,
    "z":    2,
    "none": 3,
}

# ---------------------------------------------------------------------------
# Half index mapping
# ---------------------------------------------------------------------------

HALF_INDEX: dict[str, int] = {
    "top":    0,
    "bottom": 1,
    "none":   2,
}

# ---------------------------------------------------------------------------
# Shape index mapping
# ---------------------------------------------------------------------------

SHAPE_INDEX: dict[str, int] = {
    "straight":    0,
    "inner_left":  1,
    "inner_right": 2,
    "outer_left":  3,
    "outer_right": 4,
    "none":        5,
    "spare":       6,  # unused slot to reach NUM_SHAPE=7
}


# ---------------------------------------------------------------------------
# BlockStateEncoder
# ---------------------------------------------------------------------------

class BlockStateEncoder(nn.Module):
    """
    Encodes a batch of 16×16×16 block state grids into a dense feature volume.

    Inputs (all integer tensors of shape [B, 16, 16, 16]):
        block_ids  — indices in [0, NUM_BLOCK_TYPES)
        facing_ids — indices in [0, NUM_FACING)
        axis_ids   — indices in [0, NUM_AXIS)
        half_ids   — indices in [0, NUM_HALF)
        shape_ids  — indices in [0, NUM_SHAPE)

    Output:
        [B, VOXEL_DIM, 16, 16, 16]  (channels first, float32)
    """

    def __init__(self) -> None:
        super().__init__()

        self.block_embed = nn.Embedding(NUM_BLOCK_TYPES, BLOCK_EMBED_DIM)
        self.axis_embed  = nn.Embedding(NUM_AXIS,        AXIS_EMBED_DIM)
        self.half_embed  = nn.Embedding(NUM_HALF,        HALF_EMBED_DIM)
        self.shape_embed = nn.Embedding(NUM_SHAPE,       SHAPE_EMBED_DIM)

        # Register the facing lookup table as a non-parameter buffer so it
        # moves with the module to the correct device automatically.
        self.register_buffer("facing_vectors", _FACING_TENSOR)  # [7, 3]

    def forward(
        self,
        block_ids:  torch.Tensor,   # [B, 16, 16, 16]  int
        facing_ids: torch.Tensor,   # [B, 16, 16, 16]  int
        axis_ids:   torch.Tensor,   # [B, 16, 16, 16]  int
        half_ids:   torch.Tensor,   # [B, 16, 16, 16]  int
        shape_ids:  torch.Tensor,   # [B, 16, 16, 16]  int
    ) -> torch.Tensor:
        B = block_ids.shape[0]

        # Each embedding lookup: [B, 16, 16, 16] → [B, 16, 16, 16, embed_dim]
        e_block  = self.block_embed(block_ids)   # [B, 16, 16, 16, 16]
        e_axis   = self.axis_embed(axis_ids)     # [B, 16, 16, 16, 4]
        e_half   = self.half_embed(half_ids)     # [B, 16, 16, 16, 2]
        e_shape  = self.shape_embed(shape_ids)   # [B, 16, 16, 16, 8]

        # Facing: raw vector lookup (no learnable params)
        # facing_vectors: [7, 3] → index with facing_ids → [B, 16, 16, 16, 3]
        e_facing = self.facing_vectors[facing_ids]  # [B, 16, 16, 16, 3]

        # Concatenate along last dim → [B, 16, 16, 16, VOXEL_DIM=33]
        combined = torch.cat([e_block, e_facing, e_axis, e_half, e_shape], dim=-1)

        # Permute to channels-first: [B, VOXEL_DIM, 16, 16, 16]
        return combined.permute(0, 4, 1, 2, 3).contiguous()
