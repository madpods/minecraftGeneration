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
  - Held-out validation split by building (--val-fraction), evaluated with z = mu;
    the occupancy threshold (density dial) is tuned on it each epoch
  - --loss joint|split (see vae_loss), --condition none|style (village biome)
  - Checkpoint every epoch; the 5 most recent are kept, plus best.pt
  - Per-epoch log: loss terms, plus air-aware block metrics (see BlockMetrics)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml.data.dataset import ChunkDataset, SyntheticChunkDataset
from ml.model.block_state import NUM_BLOCK_TYPES
from ml.model.vae import ConditionalChunkVAE, decode_block_ids, occupancy_logit, vae_loss

AIR_ID = 0
THRESHOLDS: list[float] = [round(0.05 * i, 2) for i in range(1, 20)]   # 0.05 .. 0.95


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

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """preds, targets: integer block ids of the same shape."""
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


class ThresholdSweep:
    """
    Accumulates block metrics at every candidate occupancy threshold at once,
    so validation can report the threshold with the best solid-voxel F1
    (the "density dial" setting to use at generation time).
    """

    def __init__(self, thresholds: list[float] = THRESHOLDS) -> None:
        self.thresholds = thresholds
        self.metrics = [BlockMetrics() for _ in thresholds]

    def update(self, block_logits: torch.Tensor, targets: torch.Tensor) -> None:
        p_solid = torch.sigmoid(occupancy_logit(block_logits))
        types = block_logits[:, AIR_ID + 1:].argmax(dim=1) + AIR_ID + 1
        for thr, m in zip(self.thresholds, self.metrics):
            preds = torch.where(p_solid > thr, types, torch.full_like(types, AIR_ID))
            m.update(preds, targets)

    def best(self) -> tuple[float, BlockMetrics]:
        def f1(m: BlockMetrics) -> float:
            s = m.summary()
            return 2 * s["precision"] * s["solid_recall"] / max(s["precision"] + s["solid_recall"], 1e-9)
        i = max(range(len(self.metrics)), key=lambda j: f1(self.metrics[j]))
        return self.thresholds[i], self.metrics[i]


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
    """
    Deterministic train/val split of the .pt files in data_dir.

    Files are grouped by building — with an index.json from the template
    extractor, a zombie variant is grouped with its normal house (same
    biome/category/template name), so a held-out house can't leak into
    training through its decayed copy. Each group is assigned to val by a
    hash of (seed, group), so a building's split doesn't change when other
    data (zombie variants, town centres, ...) is added.

    The val set holds only normal (non-zombie) houses, so validation numbers
    stay comparable across dataset variants; other files from val groups are
    left out of training entirely.
    """
    files = sorted(Path(data_dir).glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files in {data_dir}")
    index_path = Path(data_dir) / "index.json"
    index: dict = json.loads(index_path.read_text()) if index_path.exists() else {}

    def group(f: Path) -> str:
        m = index.get(f.name)
        return f"{m['biome']}/{m['category']}/{m['template']}" if m else f.stem

    def in_val(f: Path) -> bool:
        h = hashlib.sha1(f"{seed}:{group(f)}".encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64 < val_fraction

    def val_eligible(f: Path) -> bool:
        m = index.get(f.name)
        return m is None or (m["category"] == "houses" and not m["zombie"])

    train = [f for f in files if not in_val(f)]
    val   = [f for f in files if in_val(f) and val_eligible(f)]
    return train, val


def build_condition(
    model: ConditionalChunkVAE,
    condition_dict: dict,
    batch_size: int,
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    """Condition vectors for a batch: zeros (unconditional) or learned style."""
    if mode == "none":
        return _build_neutral_condition(batch_size, model.cond_dim, device)
    if mode == "style":
        if "style" not in condition_dict:
            raise ValueError("--condition style needs index.json with style labels (re-run the extractor)")
        style = condition_dict["style"].long().to(device)
        return model.style_condition(model.style_embed(style))
    raise ValueError(f"unknown condition mode {mode!r}")


def _split_seed(args: argparse.Namespace) -> int:
    """Seed for the train/val split: --split-seed if given, else --seed."""
    return args.split_seed if getattr(args, "split_seed", None) is not None else args.seed


def _run_epoch(
    model: ConditionalChunkVAE,
    loader: DataLoader,
    device: torch.device,
    kl_weight: float,
    args: argparse.Namespace,
    class_weights: torch.Tensor | None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[dict[str, float], BlockMetrics, ThresholdSweep | None]:
    """
    One pass over `loader`. Trains if an optimizer is given, else evaluates.

    Training metrics use the default decode (argmax for --loss joint, 0.5
    occupancy threshold for --loss split). Evaluation also sweeps the
    occupancy threshold (returned ThresholdSweep).
    """
    training = optimizer is not None
    model.train(training)
    use_cuda = device.type == "cuda"
    default_thr = 0.5 if args.loss == "split" else None

    sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    metrics = BlockMetrics()
    sweep = None if training else ThresholdSweep()
    n = 0

    with torch.set_grad_enabled(training):
        for tensors_dict, condition_dict in loader:
            block_ids = tensors_dict["block_ids"].long().to(device)   # [B,16,16,16]
            facing    = tensors_dict["facing"].long().to(device)
            axis      = tensors_dict["axis"].long().to(device)
            half      = tensors_dict["half"].long().to(device)
            shape     = tensors_dict["shape"].long().to(device)
            condition = build_condition(model, condition_dict, block_ids.shape[0], args.condition, device)

            with torch.amp.autocast("cuda", enabled=use_cuda):
                outputs, mu, logvar = model(block_ids, facing, axis, half, shape, condition)
                targets = {
                    "block_id": block_ids, "facing": facing, "axis": axis,
                    "half": half, "shape": shape,
                }
                loss, recon, kl = vae_loss(
                    outputs, targets, mu, logvar, kl_weight,
                    block_class_weights=class_weights,
                    mode=args.loss,
                    occ_pos_weight=args.occ_pos_weight,
                    focal_gamma=args.focal_gamma,
                )

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            sums["loss"]  += loss.item()
            sums["recon"] += recon.item()
            sums["kl"]    += kl.item()
            block_logits = outputs["block_id"].detach()
            metrics.update(decode_block_ids(block_logits, default_thr), block_ids)
            if sweep is not None:
                sweep.update(block_logits, block_ids)
            n += 1

    return {k: v / max(n, 1) for k, v in sums.items()}, metrics, sweep


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
        train_files, val_files = split_files(args.data_dir, args.val_fraction, _split_seed(args))
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

    print(f"Loss: {args.loss}   condition: {args.condition}")
    class_weights = None
    if args.air_weight != 1.0 and args.loss == "joint":
        class_weights = torch.ones(NUM_BLOCK_TYPES, device=device)
        class_weights[AIR_ID] = args.air_weight
        print(f"Block loss: air class weighted x{args.air_weight}")

    start_epoch = 0
    config = vars(args)
    best_loss = float("inf")

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    if args.resume == "auto":
        # For preemptible cluster jobs: continue from the newest checkpoint in
        # --output-dir if there is one, otherwise start fresh.
        latest = sorted(output_dir.glob("checkpoint_epoch_*.pt"))
        args.resume = str(latest[-1]) if latest else None
        if args.resume is None:
            print("--resume auto: no checkpoint found, starting fresh")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ConditionalChunkVAE.from_checkpoint(ckpt).state_dict())
        optimizer.load_state_dict(ckpt["optim_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        config      = ckpt.get("config", config)
        best_path   = output_dir / "best.pt"
        if best_path.exists():
            bm = torch.load(best_path, map_location="cpu").get("metrics", {})
            best_loss = bm.get("val", bm.get("train", {})).get("loss", best_loss)
        print(f"Resumed from {args.resume} at epoch {start_epoch} (best loss so far {best_loss:.1f})")

    # ------------------------------------------------------------------
    # Training epochs
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        # KL annealing: 0 → 1 over first kl_anneal_epochs epochs
        kl_weight = min(1.0, epoch / args.kl_anneal_epochs) if args.kl_anneal_epochs > 0 else 1.0

        tr, tr_metrics, _ = _run_epoch(model, loader, device, kl_weight, args, class_weights, optimizer, scaler)
        print(
            f"Epoch {epoch:4d}/{args.epochs} | "
            f"loss={tr['loss']:.1f}  recon={tr['recon']:.1f}  kl={tr['kl']:.1f}  "
            f"kl_w={kl_weight:.3f} | {tr_metrics.fmt()} | VRAM={_vram_str()}",
            flush=True,
        )
        epoch_metrics = {"train": {**tr, **tr_metrics.summary()}}
        monitor = tr["loss"]

        if val_loader is not None:
            va, va_metrics, sweep = _run_epoch(model, val_loader, device, kl_weight, args, class_weights)
            thr, thr_metrics = sweep.best()
            print(
                f"  val      | loss={va['loss']:.1f}  recon={va['recon']:.1f}  kl={va['kl']:.1f}  "
                f"            | {va_metrics.fmt()}",
                flush=True,
            )
            print(
                f"  val@thr  | best occupancy threshold {thr:.2f}{'':27s}| {thr_metrics.fmt()}",
                flush=True,
            )
            epoch_metrics["val"] = {
                **va, **va_metrics.summary(),
                "occ_threshold": thr,
                "at_threshold": thr_metrics.summary(),
            }
            monitor = va["loss"]

        # Step LR scheduler after epoch 50
        if epoch >= 50:
            scheduler.step(monitor)

        # ------------------------------------------------------------------
        # Checkpoint
        # ------------------------------------------------------------------
        ckpt_path = output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        _save_checkpoint(ckpt_path, epoch, model, optimizer, scaler, config, epoch_metrics)
        _prune_checkpoints(output_dir, keep=args.keep_checkpoints)
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
        "--loss", choices=["joint", "split"], default="joint",
        help="joint: one softmax over all blocks incl. air. split: block-vs-air "
             "occupancy loss + block-type loss on solid voxels only",
    )
    parser.add_argument(
        "--occ-pos-weight", type=float, default=1.0,
        help="(--loss split) weight on solid voxels in the occupancy loss",
    )
    parser.add_argument(
        "--focal-gamma", type=float, default=0.0,
        help="(--loss split) focal-loss gamma for the occupancy loss; 0 = plain BCE",
    )
    parser.add_argument(
        "--condition", choices=["none", "style"], default="none",
        help="none: zero condition. style: learned per-biome style vector (needs index.json)",
    )
    parser.add_argument(
        "--kl-anneal-epochs", type=int, default=20,
        help="Epochs over which kl_weight ramps from 0 to 1",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--seed",        type=int, default=0, help="Torch RNG seed (and split seed unless --split-seed)")
    parser.add_argument(
        "--split-seed", type=int, default=None,
        help="Seed for the train/val split only, so repeated runs with different --seed "
             "share one held-out set (14 = all five biomes held out on the village data)",
    )
    parser.add_argument("--keep-checkpoints", type=int, default=5, help="Most recent epoch checkpoints to keep")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint to resume from, or 'auto' for the newest one in --output-dir",
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
