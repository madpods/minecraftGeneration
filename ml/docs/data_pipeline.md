# Data Pipeline

## Overview

Training data is 16×16×16 chunk slices extracted from real Minecraft world saves. Each chunk is encoded as five parallel integer grids and saved as a `.pt` file. A vocabulary file maps Minecraft namespaced block IDs to the integer indices used by the model.

For early development and smoke testing, a `SyntheticChunkDataset` generates random valid tensors so the full training pipeline can be exercised without any Minecraft data.

## Scraping real Minecraft worlds

File: `ml/data/chunk_scraper.py`

Requires the `amulet-core` library (`pip install amulet-core`). Point it at any Java Edition Minecraft world folder (the directory that contains `level.dat`).

```python
from ml.data.chunk_scraper import scrape_world

scrape_world(
    world_path="/path/to/MinecraftWorld",
    output_dir="ml/data/chunks",
    max_chunks=10_000,
    y_offset=64,       # Y-level of the bottom of each 16-block slice
)
```

This walks all `.mca` region files, applies the interest heuristic (see below), and saves each accepted chunk as `chunk_000000.pt`, `chunk_000001.pt`, etc. A `vocab.json` file is written alongside the chunks mapping block namespace IDs to integer indices.

Typical yield: 20–40% of overworld chunks pass the filter in a built-up survival world.

### Interest heuristic

Chunks are discarded if they are too empty or too monotonous:

```python
def is_interesting(chunk):
    # Reject if more than 60% of voxels are air
    if air_ratio > 0.6:
        return False
    # Reject if fewer than 8 distinct block types
    if unique_block_types < 8:
        return False
    return True
```

The goal is to keep chunks from built structures and settlements while discarding open terrain, caves, and ocean chunks.

### Block vocabulary

Block IDs are assigned in order of first encounter up to `NUM_BLOCK_TYPES = 256`. `minecraft:air` is always index 0. Unknown blocks above the vocabulary limit map to index 255 (a sentinel).

```python
# Load an existing vocabulary (call before scraping to extend an existing vocab)
from ml.data.chunk_scraper import load_vocab
load_vocab("ml/data/chunks/vocab.json")
```

The `vocab.json` format:
```json
{
  "minecraft:air":          0,
  "minecraft:stone":        1,
  "minecraft:stone_bricks": 2,
  ...
}
```

At inference time, the reverse mapping (integer → namespace ID) is used to produce human-readable block names in the schematic output.

## Augmentation

File: `ml/data/augmentation.py`

Each chunk can be augmented with 4 rotations × 2 mirror states = **8× dataset expansion** at no additional data collection cost.

### Rotation

```python
from ml.data.augmentation import augment_chunk

rotated = augment_chunk(tensors, rotation=90)   # 0, 90, 180, or 270
```

Rotation is around the vertical (Y) axis, rotating the XZ plane. Internally, `torch.rot90` is applied to the spatial tensors. Critically, facing state values are remapped using `ROTATION_MAP` so block orientations remain physically correct:

```
90°  rotation:  north→east,  east→south,  south→west,  west→north
180° rotation:  north→south, east→west,   south→north, west→east
270° rotation:  north→west,  west→south,  south→east,  east→north
```

Up, down, and none facings are unaffected by horizontal rotation.

### Mirror

```python
from ml.data.augmentation import mirror_chunk

mirrored = mirror_chunk(tensors)
```

Flips the X axis (`torch.flip(t, dims=(0,))`) and swaps east↔west facing values. Combined with the four rotations this gives the full set of 8 distinct orientations (the dihedral group D4).

### Why facing remapping matters

Without remapping, a block that faces north in the original chunk would still be tagged as "facing north" after a 90° rotation — but the block is now pointing in a different direction relative to the world. The remapping ensures the saved state always agrees with the spatial arrangement.

## Dataset classes

File: `ml/data/dataset.py`

### ChunkDataset

Loads `.pt` files from a directory produced by `scrape_world()`.

```python
from ml.data.dataset import ChunkDataset

ds = ChunkDataset("ml/data/chunks", augment=True)
tensors, condition = ds[0]
# tensors: dict with keys block_ids, facing, axis, half, shape  — each [16,16,16] int16
# condition: {} empty dict (Phase 1 unconditional training)
```

Each `__getitem__` call loads the file from disk, applies a randomly chosen rotation (0/90/180/270) and a 50% chance of mirror flip.

### SyntheticChunkDataset

Generates random valid integer tensors in memory. No files needed.

```python
from ml.data.dataset import SyntheticChunkDataset

ds = SyntheticChunkDataset(size=1000, augment=True)
```

All values are sampled uniformly from `[0, vocab_size)` for each component. This produces nonsensical blocks but exercises the full training pipeline, including augmentation and the DataLoader.

The training script automatically falls back to `SyntheticChunkDataset` if the specified `--data-dir` contains no `.pt` files.

## Saved file format

Each `.pt` file is a dict of five `torch.int16` tensors of shape `[16, 16, 16]`:

```python
{
    "block_ids": torch.int16,   # vocabulary integer for each voxel
    "facing":    torch.int16,   # FACING_INDEX value
    "axis":      torch.int16,   # AXIS_INDEX value
    "half":      torch.int16,   # HALF_INDEX value
    "shape":     torch.int16,   # SHAPE_INDEX value
}
```

The index mappings are defined in `ml/model/block_state.py`:
- `FACING_INDEX`: north=0, south=1, east=2, west=3, up=4, down=5, none=6
- `AXIS_INDEX`: x=0, y=1, z=2, none=3
- `HALF_INDEX`: top=0, bottom=1, none=2
- `SHAPE_INDEX`: straight=0, inner_left=1, inner_right=2, outer_left=3, outer_right=4, none=5
