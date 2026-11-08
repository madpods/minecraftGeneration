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
│   ├── tests/              # smoke test suite (21/21 passing)
│   ├── docs/               # in-depth documentation
│   ├── train.py            # training script
│   ├── evaluate.py         # reconstructions / samples -> structure .nbt files
│   ├── inference.py        # Flask inference server
│   └── requirements.txt
├── server/                 # Node.js inference server  [future]
└── mod/                    # Fabric mod (Java)          [future]
```

## Quick start

**Requirements:** Python 3.11+, pip, and a Minecraft Java client jar (1.20.1 tested) for training data. Training runs fine on CPU.

```bash
# 1. Install dependencies (CPU-only torch is fine)
python -m venv .venv && .venv/Scripts/activate        # Windows; use bin/activate elsewhere
pip install -r ml/requirements.txt

# 2. Run the smoke tests to verify everything is working
python -m ml.tests.smoke_test
# Expected: All 21/21 tests passed.

# 3. Extract the vanilla village houses from your game jar
python -m ml.data.template_extractor \
    --jar "<minecraft dir>/versions/1.20.1/1.20.1.jar" \
    --output-dir ml/data/chunks
# Extracted 149 templates to ml/data/chunks (157 block types).

# 4. Train (held-out 10% validation split; --air-weight < 1 counters the ~95% air)
python -m ml.train --data-dir ml/data/chunks --output-dir ml/checkpoints \
    --epochs 80 --repeat 8 --air-weight 0.3

# 5. Inspect results: metrics, ASCII slices, and .nbt files you can load in game
python -m ml.evaluate --checkpoint ml/checkpoints/best.pt --out ml/samples

# 6. Start the inference server
python -m ml.inference
# Loads ml/checkpoints/best.pt (or the latest epoch checkpoint)
# Listening on http://localhost:5000
```

Without a data dir, `ml.train` falls back to random synthetic chunks, which only exercises the plumbing. See [ml/docs/data_pipeline.md](ml/docs/data_pipeline.md) for the data sources.

## Documentation

| Document | Contents |
|---|---|
| [ml/docs/architecture.md](ml/docs/architecture.md) | VAE design, layer dimensions, conditioning, KL annealing |
| [ml/docs/faction_system.md](ml/docs/faction_system.md) | Faction vectors, world events, FactionStore |
| [ml/docs/data_pipeline.md](ml/docs/data_pipeline.md) | Scraping .mca files, augmentation, dataset classes |
| [ml/docs/training.md](ml/docs/training.md) | Training phases, CLI args, checkpoints, log output |
| [ml/docs/inference_api.md](ml/docs/inference_api.md) | Flask API reference, request/response schemas |
| [ml/docs/hpc.md](ml/docs/hpc.md) | Running training sweeps on the OSU CoE HPC (Slurm) |

## Implementation phases

| Phase | Status | Description |
|---|---|---|
| 1 — ML pipeline | **Done** | cVAE model, synthetic training, shape tests |
| 2 — Faction system | **Done** | Vector arithmetic, persistence, world events |
| 3 — Node.js server | Planned | Tile caching, CouchDB persistence, mod proxy |
| 4 — Fabric mod | Planned | WFC layout, block placement, NPC behaviour |
| 5 — Real data + conditioning | Planned | Amulet scraping, condition vectors, faction tuning |
