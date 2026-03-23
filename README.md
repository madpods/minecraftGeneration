# BetterNPC Building Generation

A machine learning pipeline that enables Minecraft NPCs to autonomously construct contextually appropriate structures. NPC personality and faction identity drive generation, producing architecturally distinct buildings that evolve as the world develops.

The system uses a hybrid approach: Wave Function Collapse (WFC) determines the layout of tiles across a building footprint, then a Conditional Variational Autoencoder (cVAE) generates the actual block-level content of each 16×16×16 tile. Faction identity vectors in latent space encode architectural style — two factions that trade will gradually share stylistic features; a conquered faction will adopt its conqueror's aesthetic.

## Architecture overview

```
NPC Brain (Claude Haiku)
         |
  Building Intent JSON
         |
   WFC Layout Pass          <-- Java, runs in-mod, synchronous
         |
   Tile Slot Grid           <-- typed 3D grid of tile types
         |
  Tile Generation Server   <-- Node.js, calls Python model  [future]
         |
  Schematic Per Slot        <-- 16x16x16 block + state tensors
         |
  Block Placement Queue     <-- NPC builds over in-game time [future]
         |
  Faction Vector Update     <-- record_build() nudges faction home
```

**What is implemented today:** the Python ML pipeline (`ml/`).
The Node.js inference server (`server/`) and Fabric mod (`mod/`) are planned for future phases.

## Repository layout

```
minecraftGeneration/
├── ml/                     # Python ML pipeline  <-- implemented
│   ├── model/              # VAE, encoder, decoder, block state, factions
│   ├── data/               # data scraping, augmentation, dataset classes
│   ├── tests/              # smoke test suite (13/13 passing)
│   ├── docs/               # in-depth documentation
│   ├── train.py            # training script
│   ├── inference.py        # Flask inference server
│   └── requirements.txt
├── server/                 # Node.js inference server  [future]
└── mod/                    # Fabric mod (Java)          [future]
```

## Quick start

**Requirements:** Python 3.11+, pip.

```bash
# 1. Install dependencies
pip install -r ml/requirements.txt

# 2. Run the smoke tests to verify everything is working
cd /path/to/minecraftGeneration
python -m ml.tests.smoke_test
# Expected: All 13/13 tests passed.

# 3. Train on synthetic data (no Minecraft world needed)
python -m ml.train \
    --data-dir   /tmp/no-data-here \
    --output-dir ml/checkpoints \
    --epochs     50 \
    --batch-size 16
# The trainer auto-detects the missing data dir and falls back to
# SyntheticChunkDataset for smoke-testing the full loop.

# 4. Start the inference server
python -m ml.inference
# Loads the latest checkpoint from ml/checkpoints/
# Listening on http://localhost:5000
```

To train on real Minecraft data, see [ml/docs/data_pipeline.md](ml/docs/data_pipeline.md).

## Documentation

| Document | Contents |
|---|---|
| [ml/docs/architecture.md](ml/docs/architecture.md) | VAE design, layer dimensions, conditioning, KL annealing |
| [ml/docs/faction_system.md](ml/docs/faction_system.md) | Faction vectors, world events, FactionStore |
| [ml/docs/data_pipeline.md](ml/docs/data_pipeline.md) | Scraping .mca files, augmentation, dataset classes |
| [ml/docs/training.md](ml/docs/training.md) | Training phases, CLI args, checkpoints, log output |
| [ml/docs/inference_api.md](ml/docs/inference_api.md) | Flask API reference, request/response schemas |

## Implementation phases

| Phase | Status | Description |
|---|---|---|
| 1 — ML pipeline | **Done** | cVAE model, synthetic training, shape tests |
| 2 — Faction system | **Done** | Vector arithmetic, persistence, world events |
| 3 — Node.js server | Planned | Tile caching, CouchDB persistence, mod proxy |
| 4 — Fabric mod | Planned | WFC layout, block placement, NPC behaviour |
| 5 — Real data + conditioning | Planned | Amulet scraping, condition vectors, faction tuning |
