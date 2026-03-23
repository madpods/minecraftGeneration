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

A checkpoint is saved at the end of every epoch. Only the 5 most recent are kept:

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

Each epoch prints one line:

```
Epoch   42/100 | loss=2.3451  recon=1.8234  kl=0.5217  block_acc=0.4821  kl_w=1.000  VRAM=4.12GB / 8.00GB
```

| Field | Meaning |
|---|---|
| `loss` | Total ELBO loss (recon + kl_weight × kl) |
| `recon` | Weighted reconstruction loss across all 5 block state components |
| `kl` | KL divergence, normalized by batch size |
| `block_acc` | Fraction of voxels where the predicted block ID argmax matches ground truth |
| `kl_w` | Current KL weight (0 → 1 over first 20 epochs) |
| `VRAM` | GPU memory usage / total (shows "N/A (CPU)" if no GPU) |

**Expected progression:**

- Early epochs (0–20): `kl` is near zero, `recon` dominates. `block_acc` rises quickly from random (~0.004 for 256 classes) toward a useful level.
- Mid training (20–60): `kl` rises as the weight anneals to 1. `recon` may temporarily increase before settling.
- Later epochs (60+): Both losses should decrease smoothly. `block_acc` above 0.6 on real data indicates the model has learned meaningful block type distributions.

## Smoke test

After training, run the smoke test to verify the model checkpoint loads and generates valid outputs:

```bash
python -m ml.tests.smoke_test
```

All 13 tests should pass. Test 13 specifically exercises a 2-epoch training loop end-to-end and verifies the checkpoint format.
