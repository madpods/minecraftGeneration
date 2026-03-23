# ML Architecture

## Overview

The model is a **Conditional Variational Autoencoder (cVAE)** that generates 16×16×16 Minecraft chunk schematics. Each chunk is represented as five parallel integer grids (one per block state component). The VAE learns a continuous latent space over these chunks; at generation time, a faction identity vector selects a region of that space, and a condition vector steers the output toward the requested tile type, purpose, and face interface requirements.

## Why a cVAE?

**Controllability.** A plain VAE generates plausible chunks but gives no mechanism to request a specific interface (e.g. "north face must be a doorway"). A conditional model accepts explicit requirements as input.

**Diversity within style.** GANs produce sharp outputs but collapse around modes — all fortress rooms end up looking identical. The VAE's continuous latent space allows sampling nearby points to produce varied-but-coherent outputs. The faction variance scalar directly controls this spread.

**Latent space arithmetic.** Faction identities are just vectors. Cultural exchange, conquest, and schism all reduce to simple vector operations that remain meaningful because the VAE encoder organises related styles into contiguous regions.

## Block state representation

Minecraft blocks have a base type plus several orthogonal state properties. A naive approach would enumerate every (block_id, facing, axis, half, shape) combination as a unique token, producing a vocabulary of hundreds of thousands of entries — too large for practical softmax heads.

Instead, each voxel is encoded as **five separate integer components**:

| Component | Vocabulary size | Embedding dim | Notes |
|---|---|---|---|
| `block_id` | 256 | 16 | Maps to Minecraft namespaced IDs via `vocab.json` |
| `facing` | 7 | 3 (raw vector) | north/south/east/west/up/down/none; no learned embedding |
| `axis` | 4 | 4 | x/y/z/none |
| `half` | 3 | 2 | top/bottom/none |
| `shape` | 7 | 8 | straight/inner_left/inner_right/outer_left/outer_right/none |

Facing uses raw 3D unit vectors instead of a learned embedding because the geometry is already fully specified — north is always `[0,0,-1]`. The five component representations are concatenated to form a **33-dimensional per-voxel feature vector**.

The `BlockStateEncoder` (`ml/model/block_state.py`) applies the embedding tables across the full 16×16×16 grid and permutes to channels-first layout `[B, 33, 16, 16, 16]`.

The decoder produces **five separate logit heads** — one per component — and each is trained with an independent cross-entropy loss. This keeps the output space small and lets each component be weighted differently in the loss.

## Encoder

File: `ml/model/encoder.py`

```
Input:  [B, 33, 16, 16, 16]   (from BlockStateEncoder)
        + condition c [B, 209]

Conv3d(33 → 32,  k=4, stride=2, pad=1)  BN  ReLU   →  [B,  32, 8, 8, 8]
Conv3d(32 → 64,  k=4, stride=2, pad=1)  BN  ReLU   →  [B,  64, 4, 4, 4]
Conv3d(64 → 128, k=4, stride=2, pad=1)  BN  ReLU   →  [B, 128, 2, 2, 2]

Flatten  →  [B, 1024]
cat(c)   →  [B, 1024 + 209 = 1233]

Linear(1233, 128)  →  mu      [B, 128]
Linear(1233, 128)  →  logvar  [B, 128]
```

## Decoder

File: `ml/model/decoder.py`

```
Input:  z_c = cat(z, c)  →  [B, 128 + 209 = 337]

Linear(337, 1024)  ReLU
Reshape  →  [B, 128, 2, 2, 2]

ConvTranspose3d(128 → 64, k=4, stride=2, pad=1)  BN  ReLU   →  [B, 64, 4, 4, 4]
ConvTranspose3d( 64 → 32, k=4, stride=2, pad=1)  BN  ReLU   →  [B, 32, 8, 8, 8]
ConvTranspose3d( 32 → 32, k=4, stride=2, pad=1)  BN  ReLU   →  [B, 32, 16, 16, 16]

Conv3d(32 → 256, k=1)  →  block_id logits  [B, 256, 16, 16, 16]
Conv3d(32 →   7, k=1)  →  facing logits    [B,   7, 16, 16, 16]
Conv3d(32 →   4, k=1)  →  axis logits      [B,   4, 16, 16, 16]
Conv3d(32 →   3, k=1)  →  half logits      [B,   3, 16, 16, 16]
Conv3d(32 →   7, k=1)  →  shape logits     [B,   7, 16, 16, 16]
```

No activation is applied after the output heads — the loss function handles the softmax internally via `F.cross_entropy`.

## Condition vector

File: `ml/model/vae.py` — `ConditionBuilder`

The condition vector encodes the tile's interface requirements and the requesting faction's identity:

| Component | Dim | Notes |
|---|---|---|
| `tile_type` embedding | 32 | 20 tile types (ROOM_SMALL, HALLWAY_H, GATE, ...) |
| `purpose` embedding | 32 | 30 purpose tokens (throne_room, storage, forge, ...) |
| 6 face direction embeddings | 8 × 6 = 48 | north/south/east/west/up/down face type |
| `size` projection | 16 | scalar 0–1 → Linear(1, 16) |
| `variance` projection | 8 | scalar 0–1 → Linear(1, 8) |
| `faction_vector` | 128 | passed through unchanged |
| **Raw total** | **264** | |
| Final projection | → **209** | Linear(264, 209) |

`COND_DIM = 209` was chosen so that `LATENT_DIM + COND_DIM = 337`, which factors as `128 × 2 × 2 × 2 + 209` with no padding needed.

## Loss function

File: `ml/model/vae.py` — `vae_loss`

```python
recon = (
    cross_entropy(block_id) * 1.0   # wrong block type is the worst error
  + cross_entropy(facing)   * 0.5
  + cross_entropy(axis)     * 0.4
  + cross_entropy(half)     * 0.3
  + cross_entropy(shape)    * 0.3
)

kl = -0.5 * sum(1 + logvar - mu² - exp(logvar)) / batch_size

total = recon + kl_weight * kl
```

`block_id` is weighted highest because predicting the wrong material is more damaging to visual quality than predicting a slightly wrong orientation.

## KL annealing

Without annealing, the KL term dominates early in training and forces the encoder to collapse all inputs onto the prior — the decoder then ignores the latent code entirely (posterior collapse), and generation becomes unconditional noise.

KL annealing ramps the `kl_weight` from 0 to 1 over the first 20 epochs:

```python
kl_weight = min(1.0, epoch / 20.0)
```

This lets the reconstruction loss teach the encoder to encode meaningful structure before the KL regularisation pressure kicks in.

## FP16 training

Training uses `torch.amp.autocast("cuda")` and `torch.amp.GradScaler("cuda")` for mixed-precision. The scaler handles gradient underflow automatically. On CPU (no CUDA), both are disabled and training runs in full float32.

## Reparameterization

During training, `z = mu + sigma * epsilon` where `epsilon ~ N(0, I)`. During evaluation and generation, `z = mu` (deterministic). This is enforced by checking `self.training` inside `reparameterize()`.

For generation, the faction latent vector is used directly as `z` — bypassing the encoder entirely — so the decoder is conditioned purely on faction style rather than an encoded input chunk.
