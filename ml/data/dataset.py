"""
ChunkDataset — torch.utils.data.Dataset for 16×16×16 Minecraft chunk tensors.

Each item is loaded from a .pt file produced by chunk_scraper.py, then
augmented with a random rotation and optional mirror flip.

During Phase 1 (unconditional pre-training) the returned condition dict is
empty. Once condition vectors are available (Phase 5), a conditioning
subclass or wrapper can supply them.

Dataset returns:
    tensors_dict  — {block_ids, facing, axis, half, shape} each [16,16,16] int16
    condition_dict — {"style": int} if the data dir has an index.json with
                     style labels (village biome), else {}
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .augmentation import augment_chunk, mirror_chunk


class ChunkDataset(Dataset[tuple[dict[str, torch.Tensor], dict[str, Any]]]):
    """
    Dataset of pre-processed Minecraft chunk tensors.

    Args:
        data_dir: Directory containing .pt chunk files (from template_extractor
                  or chunk_scraper).
        augment:  If True, apply random 90°/180°/270° rotation and a 50%
                  chance of mirror flip per item.
        files:    Optional explicit subset of .pt files (e.g. a train/val split).
                  Defaults to every .pt file in data_dir.
        repeat:   Virtually repeat the file list this many times per epoch.
                  Useful for small datasets so one "epoch" sees each item under
                  several random augmentations.
    """

    _ROTATIONS: list[int] = [0, 90, 180, 270]

    def __init__(
        self,
        data_dir: str | Path,
        augment: bool = True,
        files: list[Path] | None = None,
        repeat: int = 1,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.augment  = augment
        self.repeat   = max(1, repeat)

        self.files: list[Path] = sorted(files) if files is not None else sorted(self.data_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(
                f"No .pt chunk files found in {self.data_dir}. "
                "Run ml.data.template_extractor (or chunk_scraper.scrape_world) first."
            )

        # The village datasets are tiny (~10 MB), so load every file once up
        # front instead of hitting the disk each step (which would starve a GPU).
        self._cache: list[dict[str, torch.Tensor]] = [
            torch.load(f, weights_only=True) for f in self.files
        ]

        # Style labels (village biome) from the template extractor's index.json
        index_path = self.data_dir / "index.json"
        index = json.loads(index_path.read_text()) if index_path.exists() else {}
        self.styles: list[int] | None = None
        if index and all("style" in index.get(f.name, {}) for f in self.files):
            self.styles = [int(index[f.name]["style"]) for f in self.files]

    def __len__(self) -> int:
        return len(self.files) * self.repeat

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        tensors: dict[str, torch.Tensor] = self._cache[idx % len(self.files)]

        if self.augment:
            # Random rotation around Y axis
            rotation = random.choice(self._ROTATIONS)
            tensors  = augment_chunk(tensors, rotation)

            # 50% chance of mirror flip on X axis
            if random.random() < 0.5:
                tensors = mirror_chunk(tensors)

        # Style label when available (used by --condition style); else empty
        condition_dict: dict[str, Any] = {}
        if self.styles is not None:
            condition_dict["style"] = self.styles[idx % len(self.files)]

        return tensors, condition_dict


class SyntheticChunkDataset(Dataset[tuple[dict[str, torch.Tensor], dict[str, Any]]]):
    """
    Generates random synthetic chunks for smoke-testing the training loop
    without real Minecraft data.

    All block states are sampled uniformly at random from their valid ranges.

    Args:
        size:      Number of synthetic chunks to generate.
        augment:   If True, apply the same augmentation as ChunkDataset.
        chunk_size: Spatial dimension (default 16).
    """

    _ROTATIONS: list[int] = [0, 90, 180, 270]

    def __init__(
        self,
        size: int = 1000,
        augment: bool = True,
        chunk_size: int = 16,
    ) -> None:
        from ..model.block_state import (
            NUM_AXIS,
            NUM_BLOCK_TYPES,
            NUM_FACING,
            NUM_HALF,
            NUM_SHAPE,
        )

        self.size       = size
        self.augment    = augment
        self.chunk_size = chunk_size

        self._ranges = {
            "block_ids": NUM_BLOCK_TYPES,
            "facing":    NUM_FACING,
            "axis":      NUM_AXIS,
            "half":      NUM_HALF,
            "shape":     NUM_SHAPE,
        }

    def __len__(self) -> int:
        return self.size

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        C = self.chunk_size
        tensors: dict[str, torch.Tensor] = {
            key: torch.randint(0, high, (C, C, C), dtype=torch.int16)
            for key, high in self._ranges.items()
        }

        if self.augment:
            rotation = random.choice(self._ROTATIONS)
            tensors  = augment_chunk(tensors, rotation)
            if random.random() < 0.5:
                tensors = mirror_chunk(tensors)

        return tensors, {}
