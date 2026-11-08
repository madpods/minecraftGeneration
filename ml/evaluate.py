"""
Evaluate a trained ConditionalChunkVAE checkpoint on village data.

Produces three kinds of output, each saved as vanilla structure .nbt files
(see ml/data/structure_export.py for how to load them in game):

  1. Reconstructions of held-out (validation) houses — the model never trained
     on these, so this shows whether it learned "houses" or memorised.
  2. Samples from the prior, z ~ N(0, I) — brand-new buildings.
  3. Interpolations between two held-out houses in latent space.

Also prints per-house metrics (the same air-aware metrics as training) and
ASCII slices so you can eyeball results in the terminal.

Usage:
    python -m ml.evaluate --checkpoint ml/checkpoints/best.pt --out ml/samples
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ml.data.structure_export import ascii_slices, save_structure
from ml.model.vae import ConditionalChunkVAE
from ml.train import AIR_ID, BlockMetrics, split_files

_KEYS = ("block_ids", "facing", "axis", "half", "shape")
_OUT_KEYS = {"block_id": "block_ids", "facing": "facing", "axis": "axis", "half": "half", "shape": "shape"}


def _load_vocab(*dirs: Path) -> dict[int, str]:
    for d in dirs:
        p = d / "vocab.json"
        if p.exists():
            return {v: k for k, v in json.loads(p.read_text()).items()}
    raise FileNotFoundError(f"vocab.json not found in any of {[str(d) for d in dirs]}")


def _decode(model: ConditionalChunkVAE, z: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    """Decode latents [N, LATENT_DIM] with a neutral condition -> list of tensor dicts."""
    cond = torch.zeros(z.shape[0], model.cond_dim)
    out = model.generate(cond, z)
    return [{_OUT_KEYS[k]: v[i] for k, v in out.items()} for i in range(z.shape[0])]


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    data_dir = Path(args.data_dir or cfg.get("data_dir", "ml/data/chunks"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ConditionalChunkVAE()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    rev_vocab = _load_vocab(ckpt_path.parent, data_dir)
    torch.manual_seed(args.seed)

    print(f"Checkpoint {ckpt_path} (epoch {ckpt.get('epoch')}), data {data_dir}")

    # The same deterministic split training used, so "val" really is unseen
    train_files, val_files = split_files(data_dir, cfg.get("val_fraction", 0.1), cfg.get("seed", 0))
    eval_files = val_files or train_files[: args.num_recon]
    label = "held-out" if val_files else "training (no val split in checkpoint)"

    # ------------------------------------------------------------------
    # 1. Reconstructions
    # ------------------------------------------------------------------
    print(f"\n=== Reconstructions of {len(eval_files)} {label} houses (z = mu) ===")
    total = BlockMetrics()
    mus: list[torch.Tensor] = []
    originals: list[dict[str, torch.Tensor]] = []
    for f in eval_files:
        t = torch.load(f, weights_only=True)
        batch = [t[k].long().unsqueeze(0) for k in _KEYS]
        cond = torch.zeros(1, model.cond_dim)
        outputs, mu, _ = model(*batch, cond)
        recon = {_OUT_KEYS[k]: v.argmax(1)[0] for k, v in outputs.items()}

        m = BlockMetrics()
        m.update(outputs["block_id"], batch[0])
        total.update(outputs["block_id"], batch[0])
        print(f"  {f.stem:48s} {m.fmt()}")

        save_structure(out_dir / f"recon_{f.stem}_original.nbt", t, rev_vocab)
        save_structure(out_dir / f"recon_{f.stem}_model.nbt", recon, rev_vocab)
        mus.append(mu[0])
        originals.append({"stem": f.stem, "orig": t["block_ids"], "recon": recon["block_ids"]})
    print(f"  {'ALL':48s} {total.fmt()}")

    for o in originals[: args.ascii]:
        ys = _interesting_ys(o["orig"])
        print(f"\n--- {o['stem']}: original vs reconstruction ---")
        print(ascii_slices([o["orig"], o["recon"]], rev_vocab, ys, ["original", "model"]))

    # ------------------------------------------------------------------
    # 2. Prior samples
    # ------------------------------------------------------------------
    print(f"\n=== {args.num_samples} samples from the prior z ~ N(0, I) ===")
    true_fill = torch.stack([o["orig"] != AIR_ID for o in originals]).float().mean().item() if originals else 0.0
    samples = _decode(model, torch.randn(args.num_samples, model.latent_dim))
    for i, s in enumerate(samples):
        b = s["block_ids"]
        solid = (b != AIR_ID).float().mean().item()
        kinds = len(torch.unique(b[b != AIR_ID]))
        print(f"  sample_{i:02d}  solid={solid:.3f} (houses avg {true_fill:.3f})  distinct blocks={kinds}")
        save_structure(out_dir / f"sample_{i:02d}.nbt", s, rev_vocab)
    for i, s in enumerate(samples[: args.ascii]):
        print(f"\n--- sample_{i:02d} ---")
        print(ascii_slices([s["block_ids"]], rev_vocab, _interesting_ys(s["block_ids"]), [f"sample_{i:02d}"]))

    # ------------------------------------------------------------------
    # 3. Interpolation between the first two held-out houses
    # ------------------------------------------------------------------
    if len(mus) >= 2:
        a, b = mus[0], mus[1]
        steps = torch.linspace(0, 1, args.interp_steps)
        z = torch.stack([(1 - s) * a + s * b for s in steps])
        for i, s in enumerate(_decode(model, z)):
            save_structure(out_dir / f"interp_{i:02d}.nbt", s, rev_vocab)
        print(f"\nInterpolation {originals[0]['stem']} -> {originals[1]['stem']}: "
              f"{args.interp_steps} steps saved as interp_XX.nbt")

    print(f"\nStructure files written to {out_dir}")


def _interesting_ys(block_ids: torch.Tensor, n: int = 3) -> list[int]:
    """Pick up to n Y levels that contain blocks, spread over the structure height."""
    ys = [y for y in range(block_ids.shape[1]) if (block_ids[:, y, :] != AIR_ID).any()]
    if not ys:
        return [0]
    if len(ys) <= n:
        return ys
    return [ys[round(i * (len(ys) - 1) / (n - 1))] for i in range(n)]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a ConditionalChunkVAE checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default=None, help="Defaults to the data dir used in training")
    p.add_argument("--out", default="ml/samples")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--num-recon", type=int, default=8, help="Houses to reconstruct if there is no val split")
    p.add_argument("--interp-steps", type=int, default=6)
    p.add_argument("--ascii", type=int, default=2, help="How many houses/samples to print as ASCII")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(_parse_args())
