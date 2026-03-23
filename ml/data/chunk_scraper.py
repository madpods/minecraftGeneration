"""
Amulet-based .mca region file parser for extracting training chunks.

Walks all region files in a Minecraft world directory, filters chunks by a
quality heuristic, encodes block states as integer tensors, and saves each
accepted chunk as a .pt file for the ChunkDataset.

Saved file format (torch.save):
    {
        "block_ids": torch.int16 [16, 16, 16],
        "facing":    torch.int16 [16, 16, 16],
        "axis":      torch.int16 [16, 16, 16],
        "half":      torch.int16 [16, 16, 16],
        "shape":     torch.int16 [16, 16, 16],
    }

A shared vocabulary JSON file "vocab.json" is written next to the .pt files,
mapping Minecraft namespaced block IDs → integer indices.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
import tqdm

from ..model.block_state import (
    AXIS_INDEX,
    CHUNK_SIZE,
    FACING_INDEX,
    HALF_INDEX,
    NUM_BLOCK_TYPES,
    SHAPE_INDEX,
)

# ---------------------------------------------------------------------------
# Block vocabulary (built dynamically during scraping)
# ---------------------------------------------------------------------------

# Blocks assigned IDs in order of first encounter, capped at NUM_BLOCK_TYPES.
# Index 0 is reserved for minecraft:air.
_vocab: dict[str, int] = {"minecraft:air": 0}


def _get_block_id(namespace_id: str) -> int:
    """
    Map a Minecraft namespaced block ID to a vocabulary integer.
    New IDs are assigned sequentially until NUM_BLOCK_TYPES is exhausted,
    after which unknown blocks map to index NUM_BLOCK_TYPES-1 (sentinel).
    """
    if namespace_id not in _vocab:
        if len(_vocab) < NUM_BLOCK_TYPES:
            _vocab[namespace_id] = len(_vocab)
        else:
            return NUM_BLOCK_TYPES - 1   # overflow sentinel
    return _vocab[namespace_id]


def save_vocab(output_dir: str | Path) -> None:
    """Write the current block vocabulary to output_dir/vocab.json."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "vocab.json").write_text(
        json.dumps(_vocab, indent=2, sort_keys=True)
    )


def load_vocab(vocab_path: str | Path) -> None:
    """Load a previously saved vocabulary into the module-level _vocab dict."""
    global _vocab
    _vocab = json.loads(Path(vocab_path).read_text())


# ---------------------------------------------------------------------------
# Block state extraction helpers
# ---------------------------------------------------------------------------

def _extract_facing(props: dict[str, str]) -> int:
    return FACING_INDEX.get(props.get("facing", "none"), FACING_INDEX["none"])


def _extract_axis(props: dict[str, str]) -> int:
    return AXIS_INDEX.get(props.get("axis", "none"), AXIS_INDEX["none"])


def _extract_half(props: dict[str, str]) -> int:
    return HALF_INDEX.get(props.get("half", "none"), HALF_INDEX["none"])


def _extract_shape(props: dict[str, str]) -> int:
    return SHAPE_INDEX.get(props.get("shape", "none"), SHAPE_INDEX["none"])


# ---------------------------------------------------------------------------
# Interest heuristic
# ---------------------------------------------------------------------------

def is_interesting(
    block_ids: list[list[list[int]]],
    air_id: int = 0,
    air_threshold: float = 0.6,
    min_unique_types: int = 8,
) -> bool:
    """
    Return True if a chunk is worth including in the training set.

    Rejects chunks that are:
      - Mostly air (air_ratio > air_threshold)
      - Too homogeneous (fewer than min_unique_types distinct block types)
    """
    flat: list[int] = [
        block_ids[x][y][z]
        for x in range(CHUNK_SIZE)
        for y in range(CHUNK_SIZE)
        for z in range(CHUNK_SIZE)
    ]

    total      = len(flat)
    air_count  = flat.count(air_id)
    air_ratio  = air_count / total

    if air_ratio > air_threshold:
        return False

    unique_types = len(set(flat))
    if unique_types < min_unique_types:
        return False

    return True


# ---------------------------------------------------------------------------
# Single chunk loader
# ---------------------------------------------------------------------------

def load_chunk(
    world_path: str,
    cx: int,
    cz: int,
    y_offset: int = 64,
) -> dict[str, torch.Tensor] | None:
    """
    Load a single 16×16×16 sub-chunk and return encoded block state tensors.

    Uses the amulet-core library to open the world and read block data.
    The sub-chunk at y_offset is extracted (default y=64, typical overworld).

    Args:
        world_path: Path to the Minecraft world folder.
        cx, cz:     Chunk coordinates.
        y_offset:   Y-level of the bottom of the 16-block tall slice to extract.

    Returns:
        Dict of int16 tensors {block_ids, facing, axis, half, shape}, or None if
        the chunk is not interesting / could not be loaded.
    """
    try:
        import amulet  # type: ignore[import-untyped]
        from amulet.api.errors import ChunkDoesNotExist, ChunkLoadError  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "amulet-core is required for chunk scraping. "
            "Install it with: pip install amulet-core"
        ) from e

    try:
        level = amulet.load_level(world_path)
        chunk = level.get_chunk(cx, cz, "minecraft:overworld")
    except Exception:
        return None

    # Build 3D integer arrays [16, 16, 16]
    block_id_grid: list[list[list[int]]] = [
        [[0] * CHUNK_SIZE for _ in range(CHUNK_SIZE)] for _ in range(CHUNK_SIZE)
    ]
    facing_grid: list[list[list[int]]] = [
        [[FACING_INDEX["none"]] * CHUNK_SIZE for _ in range(CHUNK_SIZE)] for _ in range(CHUNK_SIZE)
    ]
    axis_grid: list[list[list[int]]] = [
        [[AXIS_INDEX["none"]] * CHUNK_SIZE for _ in range(CHUNK_SIZE)] for _ in range(CHUNK_SIZE)
    ]
    half_grid: list[list[list[int]]] = [
        [[HALF_INDEX["none"]] * CHUNK_SIZE for _ in range(CHUNK_SIZE)] for _ in range(CHUNK_SIZE)
    ]
    shape_grid: list[list[list[int]]] = [
        [[SHAPE_INDEX["none"]] * CHUNK_SIZE for _ in range(CHUNK_SIZE)] for _ in range(CHUNK_SIZE)
    ]

    for x in range(CHUNK_SIZE):
        for y in range(CHUNK_SIZE):
            for z in range(CHUNK_SIZE):
                world_y = y_offset + y
                try:
                    block, _ = chunk.get_block(x, world_y, z)
                except Exception:
                    continue

                # Namespaced block ID, e.g. "minecraft:stone_bricks"
                ns_id = f"{block.namespace}:{block.base_name}"
                block_id_grid[x][y][z] = _get_block_id(ns_id)

                props: dict[str, str] = {
                    k: str(v) for k, v in (block.properties or {}).items()
                }
                facing_grid[x][y][z] = _extract_facing(props)
                axis_grid[x][y][z]   = _extract_axis(props)
                half_grid[x][y][z]   = _extract_half(props)
                shape_grid[x][y][z]  = _extract_shape(props)

    level.close()

    if not is_interesting(block_id_grid):
        return None

    def _to_tensor(grid: list[list[list[int]]]) -> torch.Tensor:
        return torch.tensor(grid, dtype=torch.int16)

    return {
        "block_ids": _to_tensor(block_id_grid),
        "facing":    _to_tensor(facing_grid),
        "axis":      _to_tensor(axis_grid),
        "half":      _to_tensor(half_grid),
        "shape":     _to_tensor(shape_grid),
    }


# ---------------------------------------------------------------------------
# World scraper
# ---------------------------------------------------------------------------

def scrape_world(
    world_path: str,
    output_dir: str,
    max_chunks: int = 10_000,
    y_offset: int = 64,
) -> int:
    """
    Walk all .mca region files in a world, extract interesting chunks, and
    save each as a numbered .pt tensor file.

    Args:
        world_path: Path to the Minecraft world directory.
        output_dir: Directory to write .pt chunk files into.
        max_chunks: Maximum number of chunks to save.
        y_offset:   Y-level of the bottom of the 16-block slice to extract.

    Returns:
        Number of chunks saved.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    region_dir = Path(world_path) / "region"
    if not region_dir.exists():
        # Try nested DIM structure
        region_dir = Path(world_path) / "dimensions" / "minecraft" / "overworld" / "region"

    mca_files = list(region_dir.glob("*.mca"))
    if not mca_files:
        raise FileNotFoundError(f"No .mca files found under {region_dir}")

    saved     = 0
    pbar      = tqdm.tqdm(total=max_chunks, desc="Scraping chunks", unit="chunk")

    for mca_file in mca_files:
        if saved >= max_chunks:
            break

        # Parse region coords from filename "r.X.Z.mca"
        match = re.match(r"r\.(-?\d+)\.(-?\d+)\.mca", mca_file.name)
        if not match:
            continue
        rx, rz = int(match.group(1)), int(match.group(2))

        # Each region file contains up to 32×32 chunks
        for local_x in range(32):
            for local_z in range(32):
                if saved >= max_chunks:
                    break

                cx = rx * 32 + local_x
                cz = rz * 32 + local_z

                tensors = load_chunk(world_path, cx, cz, y_offset=y_offset)
                if tensors is None:
                    continue

                out_path = out / f"chunk_{saved:06d}.pt"
                torch.save(tensors, out_path)
                saved += 1
                pbar.update(1)

    pbar.close()
    save_vocab(output_dir)
    print(f"Saved {saved} chunks to {output_dir}. Vocabulary size: {len(_vocab)} blocks.")
    return saved
