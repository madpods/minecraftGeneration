# Training Guide

## Training phases

The model is trained in phases. Earlier phases require less infrastructure and can be run immediately; later phases require real data and the full conditioning pipeline.

| Phase | Data | Condition | Goal |
|---|---|---|---|
| 1 | Synthetic (random) | None (zeros) | Prove the architecture trains; catch shape bugs |
| 2 | Real `.mca` chunks | None (zeros) | Learn real block distributions and spatial patterns |
| 5 | Real chunks | Full condition vectors | Learn to honour tile type, face requirements, faction |

Phases 3 and 4 refer to the Node.js server and Fabric mod respectively — they don't involve training.

Phase 5 is not yet implemented. The conditioning infrastructure (ConditionBuilder, face embeddings) is in place but the training loop currently passes zero condition vectors.

## Quick start

**Phase 1 — synthetic smoke test:**

```bash
python -m ml.train \
    --data-dir   /tmp/no-data \
    --output-dir ml/checkpoints \
    --epochs     50 \
    --batch-size 16
```

The trainer detects the missing directory and falls back to `SyntheticChunkDataset`. This is useful for verifying the pipeline end-to-end on any machine without Minecraft data.

**Phase 2 — real data:**

First scrape a world (see [data_pipeline.md](data_pipeline.md)):

```bash
python -c "
from ml.data.chunk_scraper import scrape_world
scrape_world('/path/to/world', 'ml/data/chunks', max_chunks=10000)
"
```

Then train:

```bash
python -m ml.train \
    --data-dir   ml/data/chunks \
    --output-dir ml/checkpoints \
    --epochs     100 \
    --batch-size 16
```

## CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `ml/data/chunks` | Directory of `.pt` chunk files; falls back to synthetic if empty |
| `--output-dir` | `ml/checkpoints` | Directory to save checkpoints |
| `--epochs` | `100` | Total training epochs |
| `--batch-size` | `16` | Batch size; reduce if you hit OOM |
| `--lr` | `1e-3` | Adam learning rate |
| `--repeat` | `1` | Repeat the training file list N× per epoch (each pass gets fresh random augmentation) — use ~8 for the 149-house village set |
| `--val-fraction` | `0.1` | Fraction of files held out for validation (split by file, deterministic from `--seed`); 0 disables |
| `--air-weight` | `1.0` | Cross-entropy weight for the air class in the block_id loss; < 1 down-weights air |
| `--kl-anneal-epochs` | `20` | Epochs for kl_weight to ramp 0 → 1 |
| `--num-workers` | `0` | DataLoader workers (0 is fine for small datasets and avoids Windows spawn overhead) |
| `--seed` | `0` | Seed for the train/val split and torch RNG |
| `--resume` | `None` | Path to a checkpoint file to resume from |
| `--chunk-size` | `16` | Must be 16; only accepted value (other values raise an error) |

## Hardware requirements

The model has approximately **4.5M parameters**. Rough VRAM requirements:

| Batch size | VRAM |
|---|---|
| 4 | ~2 GB |
| 8 | ~3 GB |
| 16 | ~5 GB |
| 32 | ~9 GB |

FP16 mixed precision is used automatically when CUDA is available, halving memory relative to float32. CPU training works but is very slow — use batch size 4 and expect ~30–60s per epoch on synthetic data.

## KL annealing

The KL divergence term is scaled by a weight that ramps from 0 to 1 over the first 20 epochs:

```
Epoch 0:  kl_weight = 0.000  (pure reconstruction loss)
Epoch 10: kl_weight = 0.500
Epoch 20: kl_weight = 1.000  (full ELBO)
Epochs 21+: kl_weight = 1.000
```

Do not disable this schedule. Running at full KL from epoch 0 causes posterior collapse — the encoder ignores its input and the decoder ignores the latent code, producing unconditional average outputs.

## Learning rate schedule

Adam is used with `lr=1e-3` for the first 50 epochs. After epoch 50, `ReduceLROnPlateau` with `patience=5` and `factor=0.5` monitors total loss and halves the learning rate when it stops improving. This is handled automatically by the training loop.

## Checkpoints

A checkpoint is saved at the end of every epoch. Only the 5 most recent are kept, plus `best.pt` (lowest validation loss once kl_weight has reached 1; the inference server prefers it). `vocab.json` and `index.json` are copied from the data dir so a checkpoint directory is self-contained:

```
ml/checkpoints/checkpoint_epoch_0000.pt
ml/checkpoints/checkpoint_epoch_0001.pt
...
```

Checkpoint format:

```python
{
    "epoch":        int,
    "model_state":  dict,          # model.state_dict()
    "optim_state":  dict,          # optimizer.state_dict()
    "scaler_state": dict,          # GradScaler state (for AMP)
    "config":       dict,          # CLI args at training time
    "metrics":      dict,          # {"train": {...}, "val": {...}} for this epoch
}
```

Resume from a checkpoint:

```bash
python -m ml.train \
    --data-dir   ml/data/chunks \
    --output-dir ml/checkpoints \
    --epochs     200 \
    --resume     ml/checkpoints/checkpoint_epoch_0099.pt
```

The epoch counter continues from where training left off. The optimizer and scaler states are fully restored so momentum and scale history are preserved.

## Reading the log output

Each epoch prints a training line and (with a validation split) a val line:

```
Epoch   42/80 | loss=812.3  recon=760.1  kl=52.2  kl_w=1.000 | acc=0.972  solid_acc=0.512  solid_recall=0.701  precision=0.801  fill=0.88 | VRAM=N/A (CPU)
  val      | loss=901.7  recon=850.2  kl=51.5              | acc=0.968  solid_acc=0.430  solid_recall=0.655  precision=0.770  fill=0.85
```

Validation runs in eval mode (z = mu, no sampling noise).

| Field | Meaning |
|---|---|
| `loss` | Total ELBO loss (recon + kl_weight × kl) |
| `recon` | Weighted reconstruction loss across all 5 block state components (nats per chunk, so values are in the thousands) |
| `kl` | KL divergence, normalized by batch size |
| `acc` | Fraction of *all* voxels with the correct block id. **Misleading on its own**: village houses are ~95% air, so predicting air everywhere scores ~0.95 |
| `solid_acc` | Of voxels that should contain a block, fraction with exactly the right block |
| `solid_recall` | Of voxels that should contain a block, fraction predicted as *any* non-air block. Near 0 = the "air everywhere" failure |
| `precision` | Of voxels predicted non-air, fraction that really are non-air. Low = spraying blocks into empty space |
| `fill` | Predicted non-air voxels ÷ true non-air voxels. 1.0 = the right amount of stuff; well below 1 = too much air |
| `kl_w` | Current KL weight (0 → 1 over first 20 epochs) |
| `VRAM` | GPU memory usage / total (shows "N/A (CPU)" if no GPU) |

**Expected progression:**

- Epoch 0: the model typically over-fills (fill ≫ 1), then quickly learns that most of the volume is air.
- Early epochs: watch `fill` and `solid_recall`. With the plain loss the model tends to under-fill (fill ≈ 0.2–0.3 by epoch 5 on the village set) because air dominates the loss; `--air-weight 0.3` roughly doubles fill at the same point.
- Mid training (to `--kl-anneal-epochs`): `kl` rises as the weight anneals to 1. `recon` may temporarily increase before settling.
- Compare train vs val `solid_acc`: a large gap on the small village set means the model is memorising houses rather than learning to build them.

## Smoke test

After training, run the smoke test to verify the model checkpoint loads and generates valid outputs:

```bash
python -m ml.tests.smoke_test
```

All 18 tests should pass. Test 13 exercises a 2-epoch training loop end-to-end and verifies the checkpoint format; tests 14–18 cover augmentation geometry, property encoding, the template extractor/exporter, the air-aware metrics, and the inference server.

## Evaluating a checkpoint

```bash
python -m ml.evaluate --checkpoint ml/checkpoints/best.pt --out ml/samples
```

Reconstructs the held-out houses (same split as training), samples new buildings from the prior, interpolates between two houses, prints metrics plus ASCII slices, and writes everything as structure `.nbt` files you can load in game (see [data_pipeline.md](data_pipeline.md#exporting-back-to-minecraft)).
