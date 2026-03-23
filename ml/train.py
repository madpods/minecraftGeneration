"""
Training script for ConditionalChunkVAE.

Usage:
    python -m ml.train \\
        --data-dir   /path/to/chunks \\
        --output-dir /path/to/checkpoints \\
        --epochs     100 \\
        --batch-size 16

When --data-dir contains no .pt files, training automatically falls back to a
SyntheticChunkDataset so the pipeline can be smoke-tested without real data.

Key training features:
  - FP16 mixed precision via torch.cuda.amp
  - KL annealing: kl_weight ramps 0 → 1 over the first 20 epochs
  - Adam lr=1e-3; ReduceLROnPlateau (patience=5) kicks in after epoch 50
  - Checkpoint every epoch; the 5 most recent checkpoints are kept
  - Per-epoch log: total_loss, recon_loss, kl_loss, block_accuracy, VRAM
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.data.dataset import ChunkDataset, SyntheticChunkDataset
from ml.model.vae import ConditionalChunkVAE, vae_loss


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
    """Zero condition vector for unconditional (Phase 1) training."""
    return torch.zeros(batch_size, cond_dim, device=device)


def _block_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of voxels where the predicted block ID matches ground truth."""
    preds = logits.argmax(dim=1)                           # [B, 16, 16, 16]
    return (preds == targets.long()).float().mean().item()


def _save_checkpoint(
    path: Path,
    epoch: int,
    model: ConditionalChunkVAE,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    config: dict,
) -> None:
    torch.save(
        {
            "epoch":        epoch,
            "model_state":  model.state_dict(),
            "optim_state":  optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "config":       config,
        },
        path,
    )


def _prune_checkpoints(output_dir: Path, keep: int = 5) -> None:
    """Delete oldest checkpoints, keeping only the `keep` most recent."""
    ckpts = sorted(output_dir.glob("checkpoint_epoch_*.pt"))
    for old in ckpts[:-keep]:
        old.unlink()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset / DataLoader
    # ------------------------------------------------------------------
    try:
        dataset = ChunkDataset(args.data_dir, augment=True)
        print(f"Loaded ChunkDataset: {len(dataset)} chunks from {args.data_dir}")
    except FileNotFoundError:
        print(
            f"No .pt files found in {args.data_dir}. "
            "Falling back to SyntheticChunkDataset (smoke-test mode)."
        )
        dataset = SyntheticChunkDataset(size=1000, augment=True, chunk_size=args.chunk_size)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = ConditionalChunkVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler    = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # LR scheduler (activated after epoch 50)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, verbose=True
    )

    start_epoch = 0
    config = vars(args)

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
        model.train()

        # KL annealing: 0 → 1 over first 20 epochs
        kl_weight = min(1.0, epoch / 20.0)

        total_loss_sum  = 0.0
        recon_loss_sum  = 0.0
        kl_loss_sum     = 0.0
        block_acc_sum   = 0.0
        num_batches     = 0

        for tensors_dict, _condition_dict in loader:
            # Move tensors to device
            block_ids = tensors_dict["block_ids"].long().to(device)   # [B,16,16,16]
            facing    = tensors_dict["facing"].long().to(device)
            axis      = tensors_dict["axis"].long().to(device)
            half      = tensors_dict["half"].long().to(device)
            shape     = tensors_dict["shape"].long().to(device)

            B         = block_ids.shape[0]
            condition = _build_neutral_condition(B, model.cond_dim, device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs, mu, logvar = model(block_ids, facing, axis, half, shape, condition)

                targets = {
                    "block_id": block_ids,
                    "facing":   facing,
                    "axis":     axis,
                    "half":     half,
                    "shape":    shape,
                }

                loss, recon, kl = vae_loss(outputs, targets, mu, logvar, kl_weight)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss_sum += loss.item()
            recon_loss_sum += recon.item()
            kl_loss_sum    += kl.item()
            block_acc_sum  += _block_accuracy(outputs["block_id"], block_ids)
            num_batches    += 1

        # ------------------------------------------------------------------
        # Epoch summary
        # ------------------------------------------------------------------
        avg_total = total_loss_sum / num_batches
        avg_recon = recon_loss_sum / num_batches
        avg_kl    = kl_loss_sum    / num_batches
        avg_acc   = block_acc_sum  / num_batches

        print(
            f"Epoch {epoch:4d}/{args.epochs} | "
            f"loss={avg_total:.4f}  recon={avg_recon:.4f}  "
            f"kl={avg_kl:.4f}  block_acc={avg_acc:.4f}  "
            f"kl_w={kl_weight:.3f}  VRAM={_vram_str()}"
        )

        # Step LR scheduler after epoch 50
        if epoch >= 50:
            scheduler.step(avg_total)

        # ------------------------------------------------------------------
        # Checkpoint
        # ------------------------------------------------------------------
        ckpt_path = output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        _save_checkpoint(ckpt_path, epoch, model, optimizer, scaler, config)
        _prune_checkpoints(output_dir, keep=5)

    print("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
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
    parser.add_argument("--epochs",     type=int,  default=100)
    parser.add_argument("--batch-size", type=int,  default=16)
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
        help="Spatial chunk dimension (8 for fast prototyping, 16 for full)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(_parse_args())
