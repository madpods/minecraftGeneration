"""
Spatial augmentation for 16×16×16 chunk tensors.

Supports:
  - Four 90° rotations around the vertical (Y) axis, rotating in the XZ plane.
  - Mirror flip on the X axis.

After spatial rotation/flip, facing state values are remapped so that block
orientations remain physically correct — e.g. a block facing north in the
original chunk should face east after a 90° clockwise rotation.

Tensor layout: all spatial tensors are assumed to have shape [16, 16, 16]
with dimensions ordered (X, Y, Z). Rotations use torch.rot90 on dims (0, 2),
i.e. the XZ plane, leaving Y (vertical) unchanged.
"""

from __future__ import annotations

import torch

from ..model.block_state import FACING_INDEX

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


def _build_facing_remap(
    facing_map: dict[str, str],
) -> torch.Tensor:
    """
    Build a 1D integer remap tensor of shape [NUM_FACING] such that
    remap[old_facing_idx] = new_facing_idx.
    """
    remap = list(range(len(FACING_INDEX)))   # identity by default
    for src_name, dst_name in facing_map.items():
        remap[FACING_INDEX[src_name]] = FACING_INDEX[dst_name]
    return torch.tensor(remap, dtype=torch.long)


# Pre-built remap tensors for fast index substitution
_ROTATION_REMAP: dict[int, torch.Tensor] = {
    rot: _build_facing_remap(facing_map)
    for rot, facing_map in ROTATION_MAP.items()
}
_MIRROR_REMAP: torch.Tensor = _build_facing_remap(MIRROR_FACING_MAP)


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

    # Number of 90° CCW steps for torch.rot90.
    # torch.rot90 with k=1 rotates CCW; we want CW, so k = (4 - steps) % 4.
    steps_ccw = {90: 3, 180: 2, 270: 1}[rotation]

    result: dict[str, torch.Tensor] = {}
    remap = _ROTATION_REMAP[rotation]

    for key, t in tensors.items():
        rotated = torch.rot90(t, k=steps_ccw, dims=(0, 2))  # spatial rotation

        if key == "facing":
            # Remap facing indices: apply remap[old_idx] = new_idx element-wise
            rotated = remap[rotated.long()].to(t.dtype)

        result[key] = rotated

    return result


def mirror_chunk(
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Mirror a chunk by flipping the X axis and swapping east ↔ west facing values.

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

        result[key] = flipped

    return result
