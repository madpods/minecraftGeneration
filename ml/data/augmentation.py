"""
Spatial augmentation for 16×16×16 chunk tensors.

Supports:
  - Four 90° rotations around the vertical (Y) axis, rotating in the XZ plane.
  - Mirror flip on the X axis.

After spatial rotation/flip, state values are remapped so that block
orientations remain physically correct:
  - facing: a block facing north faces east after a 90° clockwise rotation
  - axis:   x <-> z swap on 90°/270° rotations (logs, pillars)
  - shape:  left <-> right swap on mirror (stair corners, door hinges, chests)

Tensor layout: all spatial tensors are assumed to have shape [16, 16, 16]
with dimensions ordered (X, Y, Z), using Minecraft's axes: +X = east,
+Z = south, +Y = up. Seen from above with north at the top, +X points right
and +Z points down, so torch.rot90(k=1, dims=(0, 2)) — which rotates +X
toward +Z — is a 90° CLOCKWISE turn (east -> south).
"""

from __future__ import annotations

import torch

from ..model.block_state import AXIS_INDEX, FACING_INDEX, SHAPE_INDEX

# ---------------------------------------------------------------------------
# Facing remap tables
# ---------------------------------------------------------------------------

# Maps each facing name to the new facing name after clockwise rotation
# (when viewed from above). 90° CW: north→east, east→south, south→west, west→north.
ROTATION_MAP: dict[int, dict[str, str]] = {
    90:  {
        "north": "east",
        "east":  "south",
        "south": "west",
        "west":  "north",
    },
    180: {
        "north": "south",
        "east":  "west",
        "south": "north",
        "west":  "east",
    },
    270: {
        "north": "west",
        "west":  "south",
        "south": "east",
        "east":  "north",
    },
}

# Mirror on X axis swaps east ↔ west
MIRROR_FACING_MAP: dict[str, str] = {
    "east": "west",
    "west": "east",
}

# Directions unaffected by horizontal rotation / mirror
_VERTICAL_DIRECTIONS: set[str] = {"up", "down", "none"}


# 90°/270° rotations turn X-aligned blocks into Z-aligned ones and vice versa
AXIS_SWAP_MAP: dict[str, str] = {"x": "z", "z": "x"}

# A mirror reverses handedness: every left becomes right
MIRROR_SHAPE_MAP: dict[str, str] = {
    "inner_left":  "inner_right",
    "inner_right": "inner_left",
    "outer_left":  "outer_right",
    "outer_right": "outer_left",
    "left":        "right",
    "right":       "left",
}


def _build_remap(index: dict[str, int], value_map: dict[str, str]) -> torch.Tensor:
    """
    Build a 1D integer remap tensor of shape [len(index)] such that
    remap[old_idx] = new_idx.
    """
    remap = list(range(len(index)))   # identity by default
    for src_name, dst_name in value_map.items():
        remap[index[src_name]] = index[dst_name]
    return torch.tensor(remap, dtype=torch.long)


# Pre-built remap tensors for fast index substitution
_ROTATION_REMAP: dict[int, torch.Tensor] = {
    rot: _build_remap(FACING_INDEX, facing_map)
    for rot, facing_map in ROTATION_MAP.items()
}
_MIRROR_REMAP: torch.Tensor = _build_remap(FACING_INDEX, MIRROR_FACING_MAP)
_AXIS_SWAP_REMAP: torch.Tensor = _build_remap(AXIS_INDEX, AXIS_SWAP_MAP)
_MIRROR_SHAPE_REMAP: torch.Tensor = _build_remap(SHAPE_INDEX, MIRROR_SHAPE_MAP)


# ---------------------------------------------------------------------------
# Public augmentation functions
# ---------------------------------------------------------------------------

def augment_chunk(
    tensors: dict[str, torch.Tensor],
    rotation: int,
) -> dict[str, torch.Tensor]:
    """
    Rotate a chunk 90/180/270° clockwise (viewed from above) around the Y axis.

    Spatial tensors are rotated using torch.rot90 on the XZ plane (dims 0 and 2).
    Facing values are remapped so block orientations stay physically correct.

    Args:
        tensors:  Dict with keys block_ids, facing, axis, half, shape.
                  All tensors must have shape [16, 16, 16] and dtype torch.int16
                  (or any integer dtype).
        rotation: Must be 0, 90, 180, or 270.  0 is a no-op.

    Returns:
        New dict with augmented tensors (same dtype as input).
    """
    if rotation not in (0, 90, 180, 270):
        raise ValueError(f"rotation must be 0, 90, 180, or 270; got {rotation}")

    if rotation == 0:
        return {k: v.clone() for k, v in tensors.items()}

    # torch.rot90(k=1, dims=(0, 2)) is one clockwise quarter turn in Minecraft
    # coordinates (see module docstring), so k is simply rotation / 90.
    k = rotation // 90

    result: dict[str, torch.Tensor] = {}
    remap = _ROTATION_REMAP[rotation]

    for key, t in tensors.items():
        rotated = torch.rot90(t, k=k, dims=(0, 2))  # spatial rotation

        if key == "facing":
            # Remap facing indices: apply remap[old_idx] = new_idx element-wise
            rotated = remap[rotated.long()].to(t.dtype)
        elif key == "axis" and rotation in (90, 270):
            rotated = _AXIS_SWAP_REMAP[rotated.long()].to(t.dtype)

        result[key] = rotated

    return result


def mirror_chunk(
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Mirror a chunk by flipping the X axis, swapping east ↔ west facing values,
    and swapping left ↔ right shape values (a mirror reverses handedness).

    Args:
        tensors: Dict with keys block_ids, facing, axis, half, shape.
                 All tensors shape [16, 16, 16].

    Returns:
        New dict with mirrored tensors.
    """
    result: dict[str, torch.Tensor] = {}

    for key, t in tensors.items():
        flipped = torch.flip(t, dims=(0,))   # flip X axis

        if key == "facing":
            flipped = _MIRROR_REMAP[flipped.long()].to(t.dtype)
        elif key == "shape":
            flipped = _MIRROR_SHAPE_REMAP[flipped.long()].to(t.dtype)

        result[key] = flipped

    return result
