"""
Training script for ConditionalChunkVAE.

Usage:
    python -m ml.train \\
        --data-dir   ml/data/chunks \\
        --output-dir ml/checkpoints \\
        --epochs     100 \\
        --batch-size 16

When --data-dir contains no .pt files, training automatically falls back to a
SyntheticChunkDataset so the pipeline can be smoke-tested without real data.

Key training features:
  - FP16 mixed precision on CUDA (plain fp32 on CPU)
  - KL annealing: kl_weight ramps 0 → 1 over the first 20 epochs
  - Adam lr=1e-3; ReduceLROnPlateau (patience=5) kicks in after epoch 50
  - Held-out validation split by file (--val-fraction), evaluated with z = mu
  - Checkpoint every epoch; the 5 most recent are kept, plus best.pt
  - Per-epoch log: loss terms, plus air-aware block metrics (see BlockMetrics)
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml.data.dataset import ChunkDataset, SyntheticChunkDataset
from ml.model.block_state import NUM_BLOCK_TYPES
from ml.model.vae import ConditionalChunkVAE, vae_loss

AIR_ID = 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class BlockMetrics:
    """
    Accumulates block-id prediction counts over an epoch.

    Village chunks are ~95% air, so plain voxel accuracy is ~0.95 for a model
    that predicts nothing but air. The solid-block metrics expose that:

        acc          — fraction of all voxels with the correct block id
        solid_acc    — of voxels that should be a block, fraction with the
                       exactly-correct block id
        solid_recall — of voxels that should be a block, fraction predicted
                       as *some* non-air block (0 => "air everywhere")
        precision    — of voxels predicted non-air, fraction that really are
                       non-air (low => spraying blocks into empty space)
        fill         — predicted non-air fraction / true non-air fraction
                       (1.0 = right amount of stuff; <<1 = too much air)
    """
    voxels:        int = 0
    correct:       int = 0
    solid:         int = 0
    solid_correct: int = 0
    pred_solid:    int = 0
    both_solid:    int = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = logits.argmax(dim=1)
        t     = targets.long()
        solid      = t != AIR_ID
        pred_solid = preds != AIR_ID
        hit        = preds == t
        self.voxels        += t.numel()
        self.correct       += hit.sum().item()
        self.solid         += solid.sum().item()
        self.solid_correct += (hit & solid).sum().item()
        self.pred_solid    += pred_solid.sum().item()
        self.both_solid    += (pred_solid & solid).sum().item()

    def summary(self) -> dict[str, float]:
        d = lambda a, b: a / b if b else 0.0
        return {
            "acc":          d(self.correct, self.voxels),
            "solid_acc":    d(self.solid_correct, self.solid),
            "solid_recall": d(self.both_solid, self.solid),
            "precision":    d(self.both_solid, self.pred_solid),
            "fill":         d(self.pred_solid, self.solid),
        }

    def fmt(self) -> str:
        s = self.summary()
        return (
            f"acc={s['acc']:.3f}  solid_acc={s['solid_acc']:.3f}  "
            f"solid_recall={s['solid_recall']:.3f}  precision={s['precision']:.3f}  "
            f"fill={s['fill']:.2f}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vram_str() -> str:
    """Return a human-readable VRAM usage string (GPU only)."""
    if not torch.cuda.is_available():
        return "N/A (CPU)"
    allocated = torch.cuda.memory_allocated() / 1e9
    total     = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{allocated:.2f}GB / {total:.2f}GB"


def _build_neutral_condition(batch_size: int, cond_dim: int, device: torch.device) -> torch.Tensor:
    """Zero condition vector for unconditional (Phase 1/2) training."""
    return torch.zeros(batch_size, cond_dim, device=device)


def _save_checkpoint(
    path: Path,
    epoch: int,
    model: ConditionalChunkVAE,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict,
    metrics: dict | None = None,
) -> None:
    torch.save(
        {
            "epoch":        epoch,
            "model_state":  model.state_dict(),
            "optim_state":  optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "config":       config,
            "metrics":      metrics or {},
        },
        path,
    )


def _prune_checkpoints(output_dir: Path, keep: int = 5) -> None:
    """Delete oldest checkpoints, keeping only the `keep` most recent."""
    ckpts = sorted(output_dir.glob("checkpoint_epoch_*.pt"))
    for old in ckpts[:-keep]:
        old.unlink()


def split_files(data_dir: str | Path, val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    """Deterministic train/val split of the .pt files in data_dir (by file)."""
    files = sorted(Path(data_dir).glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files in {data_dir}")
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    n_val = int(round(len(files) * val_fraction)) if val_fraction > 0 else 0
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


def _run_epoch(
    model: ConditionalChunkVAE,
    loader: DataLoader,
    device: torch.device,
    kl_weight: float,
    class_weights: torch.Tensor | None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[dict[str, float], BlockMetrics]:
    """One pass over `loader`. Trains if an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    use_cuda = device.type == "cuda"

    sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    metrics = BlockMetrics()
    n = 0

    with torch.set_grad_enabled(training):
        for tensors_dict, _condition_dict in loader:
            block_ids = tensors_dict["block_ids"].long().to(device)   # [B,16,16,16]
            facing    = tensors_dict["facing"].long().to(device)
            axis      = tensors_dict["axis"].long().to(device)
            half      = tensors_dict["half"].long().to(device)
            shape     = tensors_dict["shape"].long().to(device)
            condition = _build_neutral_condition(block_ids.shape[0], model.cond_dim, device)

            with torch.amp.autocast("cuda", enabled=use_cuda):
                outputs, mu, logvar = model(block_ids, facing, axis, half, shape, condition)
                targets = {
                    "block_id": block_ids, "facing": facing, "axis": axis,
                    "half": half, "shape": shape,
                }
                loss, recon, kl = vae_loss(
                    outputs, targets, mu, logvar, kl_weight, block_class_weights=class_weights
                )

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            sums["loss"]  += loss.item()
            sums["recon"] += recon.item()
            sums["kl"]    += kl.item()
            metrics.update(outputs["block_id"].detach(), block_ids)
            n += 1

    return {k: v / max(n, 1) for k, v in sums.items()}, metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    if args.chunk_size != 16:
        raise ValueError(
            "--chunk-size must be 16; the encoder/decoder are hardcoded for 16³ input "
            "(three stride-2 convolutions: 16→8→4→2). Use the default --chunk-size 16."
        )

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset / DataLoader
    # ------------------------------------------------------------------
    val_dataset = None
    try:
        train_files, val_files = split_files(args.data_dir, args.val_fraction, args.seed)
        dataset = ChunkDataset(args.data_dir, augment=True, files=train_files, repeat=args.repeat)
        if val_files:
            val_dataset = ChunkDataset(args.data_dir, augment=False, files=val_files)
        print(
            f"Loaded ChunkDataset from {args.data_dir}: {len(train_files)} train files "
            f"(x{args.repeat} repeat = {len(dataset)} samples/epoch), {len(val_files)} val files"
        )
        # Ship the vocabulary/index with the checkpoints so inference can decode them
        for meta in ("vocab.json", "index.json"):
            if (Path(args.data_dir) / meta).exists():
                shutil.copy(Path(args.data_dir) / meta, output_dir / meta)
    except FileNotFoundError:
        print(
            f"No .pt files found in {args.data_dir}. "
            "Falling back to SyntheticChunkDataset (smoke-test mode)."
        )
        dataset = SyntheticChunkDataset(size=1000, augment=True, chunk_size=args.chunk_size)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    loader = DataLoader(dataset, shuffle=True, drop_last=len(dataset) > args.batch_size, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs) if val_dataset else None

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = ConditionalChunkVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler    = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # LR scheduler (activated after epoch 50)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    class_weights = None
    if args.air_weight != 1.0:
        class_weights = torch.ones(NUM_BLOCK_TYPES, device=device)
        class_weights[AIR_ID] = args.air_weight
        print(f"Block loss: air class weighted x{args.air_weight}")

    start_epoch = 0
    config = vars(args)
    best_loss = float("inf")

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        config      = ckpt.get("config", config)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # ------------------------------------------------------------------
    # Training epochs
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        # KL annealing: 0 → 1 over first kl_anneal_epochs epochs
        kl_weight = min(1.0, epoch / args.kl_anneal_epochs) if args.kl_anneal_epochs > 0 else 1.0

        tr, tr_metrics = _run_epoch(model, loader, device, kl_weight, class_weights, optimizer, scaler)
        print(
            f"Epoch {epoch:4d}/{args.epochs} | "
            f"loss={tr['loss']:.1f}  recon={tr['recon']:.1f}  kl={tr['kl']:.1f}  "
            f"kl_w={kl_weight:.3f} | {tr_metrics.fmt()} | VRAM={_vram_str()}",
            flush=True,
        )
        epoch_metrics = {"train": {**tr, **tr_metrics.summary()}}
        monitor = tr["loss"]

        if val_loader is not None:
            va, va_metrics = _run_epoch(model, val_loader, device, kl_weight, class_weights)
            print(
                f"  val      | loss={va['loss']:.1f}  recon={va['recon']:.1f}  kl={va['kl']:.1f}  "
                f"            | {va_metrics.fmt()}",
                flush=True,
            )
            epoch_metrics["val"] = {**va, **va_metrics.summary()}
            monitor = va["loss"]

        # Step LR scheduler after epoch 50
        if epoch >= 50:
            scheduler.step(monitor)

        # ------------------------------------------------------------------
        # Checkpoint
        # ------------------------------------------------------------------
        ckpt_path = output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        _save_checkpoint(ckpt_path, epoch, model, optimizer, scaler, config, epoch_metrics)
        _prune_checkpoints(output_dir, keep=5)
        # Only consider "best" once KL is fully weighted, so losses are comparable
        if kl_weight >= 1.0 and monitor < best_loss:
            best_loss = monitor
            shutil.copy(ckpt_path, output_dir / "best.pt")

    print("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConditionalChunkVAE")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="ml/data/chunks",
        help="Directory of .pt chunk files (falls back to synthetic if empty)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ml/checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Repeat the training file list N times per epoch (for small datasets)",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.1,
        help="Fraction of .pt files held out for validation (0 disables)",
    )
    parser.add_argument(
        "--air-weight", type=float, default=1.0,
        help="Cross-entropy weight for the air class in block_id loss (<1 down-weights air)",
    )
    parser.add_argument(
        "--kl-anneal-epochs", type=int, default=20,
        help="Epochs over which kl_weight ramps from 0 to 1",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Spatial chunk dimension (must be 16)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(_parse_args())
