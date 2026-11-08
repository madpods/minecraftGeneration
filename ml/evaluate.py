"""
Evaluate a trained ConditionalChunkVAE checkpoint on village data.

Produces, each saved as vanilla structure .nbt files (see
ml/data/structure_export.py for how to load them in game):

  1. Reconstructions of held-out (validation) houses — the model never trained
     on these, so this shows whether it learned "houses" or memorised.
  2. Samples from the prior, z ~ N(0, I) — brand-new buildings. For a
     style-conditioned checkpoint, samples are drawn for every biome style,
     plus hybrids that blend two styles.
  3. Interpolations between two held-out houses in latent space.

Checks printed alongside:
  - air-aware block metrics (same as training), at plain argmax and at the
    tuned occupancy threshold saved in the checkpoint
  - novelty: each sample's closest training house (best occupancy IoU over all
    8 rotations/mirrors). Compare with how close the real held-out houses are
    to their nearest training house — similar numbers mean the samples are as
    "new" as a genuinely different house; values near 1.0 mean copies.
  - style: a simple block-palette classifier (trained on the training houses'
    per-biome block frequencies) labels each sample with the biome it looks
    like. Its accuracy on the real held-out houses is reported first, as a
    sanity check of the classifier itself.

Usage:
    python -m ml.evaluate --checkpoint ml/checkpoints/best.pt --out ml/samples
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from ml.data.structure_export import ascii_slices, save_structure
from ml.model.vae import ConditionalChunkVAE, decode_block_ids
from ml.train import AIR_ID, BlockMetrics, split_files

_KEYS = ("block_ids", "facing", "axis", "half", "shape")
_OUT_KEYS = {"block_id": "block_ids", "facing": "facing", "axis": "axis", "half": "half", "shape": "shape"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _find_meta(name: str, *dirs: Path) -> Path | None:
    for d in dirs:
        if (d / name).exists():
            return d / name
    return None


def _load_vocab(*dirs: Path) -> dict[int, str]:
    p = _find_meta("vocab.json", *dirs)
    if p is None:
        raise FileNotFoundError(f"vocab.json not found in any of {[str(d) for d in dirs]}")
    return {v: k for k, v in json.loads(p.read_text()).items()}


# ---------------------------------------------------------------------------
# Novelty: nearest training house under all 8 orientations
# ---------------------------------------------------------------------------

def _orientations(grid: torch.Tensor) -> list[torch.Tensor]:
    """All 8 rotations/mirrors of an (X, Y, Z) grid around the Y axis."""
    out = []
    for g in (grid, torch.flip(grid, dims=(0,))):
        out.extend(torch.rot90(g, k, dims=(0, 2)) for k in range(4))
    return out


class NearestNeighbour:
    """Occupancy-IoU nearest neighbour against a set of reference houses."""

    def __init__(self, files: list[Path]) -> None:
        self.names = [f.stem for f in files]
        occ = []
        for f in files:
            b = torch.load(f, weights_only=True)["block_ids"].long()
            occ.append(torch.stack([o != AIR_ID for o in _orientations(b)]))   # [8, X, Y, Z]
        self.occ = torch.stack(occ).flatten(2).float()                            # [N, 8, V]
        self.size = self.occ.sum(-1)                                              # [N, 8]

    def query(self, block_ids: torch.Tensor) -> tuple[str, float]:
        q = (block_ids.long() != AIR_ID).flatten().float()                        # [V]
        inter = self.occ @ q                                                      # [N, 8]
        iou = inter / (self.size + q.sum() - inter).clamp(min=1)
        best = iou.max(dim=1).values
        i = int(best.argmax())
        return self.names[i], float(best[i])


# ---------------------------------------------------------------------------
# Style: block-palette biome classifier
# ---------------------------------------------------------------------------

class PaletteClassifier:
    """Naive-Bayes biome guess from which blocks a structure uses."""

    def __init__(self, files: list[Path], index: dict, num_blocks: int = 256) -> None:
        self.biomes = sorted({index[f.name]["biome"] for f in files if f.name in index})
        counts = torch.ones(len(self.biomes), num_blocks)                        # add-one smoothing
        for f in files:
            if f.name not in index:
                continue
            b = torch.load(f, weights_only=True)["block_ids"].long()
            solid = b[b != AIR_ID]
            counts[self.biomes.index(index[f.name]["biome"])] += torch.bincount(solid, minlength=num_blocks)
        self.log_p = (counts / counts.sum(1, keepdim=True)).log()                 # [biomes, blocks]

    def predict(self, block_ids: torch.Tensor) -> str:
        b = block_ids.long()
        hist = torch.bincount(b[b != AIR_ID], minlength=self.log_p.shape[1]).float()
        return self.biomes[int((self.log_p @ hist).argmax())]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    data_dir = Path(args.data_dir or cfg.get("data_dir", "ml/data/chunks"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ConditionalChunkVAE.from_checkpoint(ckpt)
    model.eval()
    rev_vocab = _load_vocab(ckpt_path.parent, data_dir)
    index_path = _find_meta("index.json", data_dir, ckpt_path.parent)
    index: dict = json.loads(index_path.read_text()) if index_path else {}
    torch.manual_seed(args.seed)

    conditioned = cfg.get("condition", "none") == "style"
    saved_thr = ckpt.get("metrics", {}).get("val", {}).get("occ_threshold")
    thr = args.threshold if args.threshold is not None else saved_thr
    style_of = dict(sorted({m["biome"]: m["style"] for m in index.values() if "style" in m}.items()))

    print(f"Checkpoint {ckpt_path} (epoch {ckpt.get('epoch')}), data {data_dir}")
    print(f"loss={cfg.get('loss', 'joint')}  condition={cfg.get('condition', 'none')}  "
          f"occupancy threshold={thr if thr is not None else 'argmax'}")

    def condition_for(files: list[Path]) -> torch.Tensor:
        if not conditioned:
            return torch.zeros(len(files), model.cond_dim)
        styles = torch.tensor([index[f.name]["style"] for f in files])
        return model.style_condition(model.style_embed(styles))

    # The same deterministic split training used, so "val" really is unseen
    split_seed = cfg.get("split_seed") if cfg.get("split_seed") is not None else cfg.get("seed", 0)
    train_files, val_files = split_files(data_dir, cfg.get("val_fraction", 0.1), split_seed)
    eval_files = val_files or train_files[: args.num_recon]
    label = "held-out" if val_files else "training (no val split in checkpoint)"

    # ------------------------------------------------------------------
    # 1. Reconstructions
    # ------------------------------------------------------------------
    print(f"\n=== Reconstructions of {len(eval_files)} {label} houses (z = mu) ===")
    total_argmax, total_thr = BlockMetrics(), BlockMetrics()
    mus: list[torch.Tensor] = []
    originals: list[dict] = []
    for f in eval_files:
        t = torch.load(f, weights_only=True)
        batch = [t[k].long().unsqueeze(0) for k in _KEYS]
        outputs, mu, _ = model(*batch, condition_for([f]))
        recon = {_OUT_KEYS[k]: v.argmax(1)[0] for k, v in outputs.items()}
        recon["block_ids"] = decode_block_ids(outputs["block_id"], thr)[0]

        total_argmax.update(outputs["block_id"].argmax(1), batch[0])
        m = BlockMetrics()
        m.update(recon["block_ids"].unsqueeze(0), batch[0])
        total_thr.update(recon["block_ids"].unsqueeze(0), batch[0])
        print(f"  {f.stem:48s} {m.fmt()}")

        save_structure(out_dir / f"recon_{f.stem}_original.nbt", t, rev_vocab)
        save_structure(out_dir / f"recon_{f.stem}_model.nbt", recon, rev_vocab)
        mus.append(mu[0])
        originals.append({"file": f, "orig": t["block_ids"], "recon": recon["block_ids"]})
    print(f"  {'ALL (plain argmax)':48s} {total_argmax.fmt()}")
    if thr is not None:
        print(f"  {f'ALL (threshold {thr:.2f})':48s} {total_thr.fmt()}")

    for o in originals[: args.ascii]:
        ys = _interesting_ys(o["orig"])
        print(f"\n--- {o['file'].stem}: original vs reconstruction ---")
        print(ascii_slices([o["orig"], o["recon"]], rev_vocab, ys, ["original", "model"]))

    # ------------------------------------------------------------------
    # Reference numbers for novelty and style checks
    # ------------------------------------------------------------------
    nn = NearestNeighbour(train_files)
    clf = PaletteClassifier(train_files, index) if index else None
    true_fill = torch.stack([o["orig"] != AIR_ID for o in originals]).float().mean().item()

    print("\n=== Reference: held-out houses vs training set ===")
    val_nn = [nn.query(o["orig"])[1] for o in originals]
    print(f"  nearest-training-house IoU of REAL unseen houses: "
          f"median {sorted(val_nn)[len(val_nn) // 2]:.2f}, max {max(val_nn):.2f}  "
          f"(samples near this are as novel as a real new house; ~1.0 = copy)")
    if clf and index:
        hits = sum(clf.predict(o["orig"]) == index[o["file"].name]["biome"] for o in originals if o["file"].name in index)
        print(f"  palette classifier on real held-out houses: {hits}/{len(originals)} biomes correct")

    # ------------------------------------------------------------------
    # 2. Samples
    # ------------------------------------------------------------------
    def report_samples(title: str, names: list[str], cond: torch.Tensor, requested: list[str] | None) -> None:
        z = torch.randn(cond.shape[0], model.latent_dim)
        samples = _decode(model, z, cond, thr)
        print(f"\n=== {title} ===")
        hits = 0
        for i, (name, s) in enumerate(zip(names, samples)):
            b = s["block_ids"]
            solid = (b != AIR_ID).float().mean().item()
            near, iou = nn.query(b)
            guess = clf.predict(b) if clf else "-"
            if requested and guess == requested[i]:
                hits += 1
            print(f"  {name:22s} solid={solid:.3f} (houses {true_fill:.3f})  "
                  f"looks like: {guess:8s} nearest: {near} (IoU {iou:.2f})")
            save_structure(out_dir / f"{name}.nbt", s, rev_vocab)
        if requested:
            print(f"  -> {hits}/{len(names)} samples look like the biome they were asked for")
        for name, s in list(zip(names, samples))[: args.ascii]:
            print(f"\n--- {name} ---")
            print(ascii_slices([s["block_ids"]], rev_vocab, _interesting_ys(s["block_ids"]), [name]))

    n = args.num_samples
    if conditioned and style_of:
        names, styles, requested = [], [], []
        for biome, sid in style_of.items():
            for i in range(n):
                names.append(f"sample_{biome}_{i:02d}")
                styles.append(sid)
                requested.append(biome)
        cond = model.style_condition(model.style_embed(torch.tensor(styles)))
        report_samples(f"{n} samples per biome style from the prior", names, cond, requested)

        # Hybrids: blend two style vectors, the core operation of the faction system
        a, b = args.hybrid.split("+")
        va, vb = (model.style_embed(torch.tensor([style_of[x]])) for x in (a, b))
        mixes = [0.0, 0.25, 0.5, 0.75, 1.0]
        hnames = [f"hybrid_{a}{int(100 * (1 - w))}_{b}{int(100 * w)}" for w in mixes]
        z = torch.randn(1, model.latent_dim).repeat(len(mixes), 1)   # same z: only style changes
        cond = model.style_condition(torch.cat([(1 - w) * va + w * vb for w in mixes]))
        print(f"\n=== Style blend {a} -> {b} (same z, style vector mixed) ===")
        for name, s in zip(hnames, _decode(model, z, cond, thr)):
            guess = clf.predict(s["block_ids"]) if clf else "-"
            print(f"  {name:28s} looks like: {guess}")
            save_structure(out_dir / f"{name}.nbt", s, rev_vocab)
    else:
        names = [f"sample_{i:02d}" for i in range(n)]
        report_samples(f"{n} samples from the prior z ~ N(0, I)", names,
                       torch.zeros(n, model.cond_dim), None)
        if clf:
            print("  biome mix of unconditioned samples is expected — nothing asks for a style")

    # ------------------------------------------------------------------
    # 3. Interpolation between the first two held-out houses
    # ------------------------------------------------------------------
    if len(mus) >= 2:
        steps = torch.linspace(0, 1, args.interp_steps)
        z = torch.stack([(1 - s) * mus[0] + s * mus[1] for s in steps])
        c0, c1 = condition_for([originals[0]["file"]]), condition_for([originals[1]["file"]])
        cond = torch.cat([(1 - s) * c0 + s * c1 for s in steps])
        for i, s in enumerate(_decode(model, z, cond, thr)):
            save_structure(out_dir / f"interp_{i:02d}.nbt", s, rev_vocab)
        print(f"\nInterpolation {originals[0]['file'].stem} -> {originals[1]['file'].stem}: "
              f"{args.interp_steps} steps saved as interp_XX.nbt")

    print(f"\nStructure files written to {out_dir}")


def _decode(
    model: ConditionalChunkVAE,
    z: torch.Tensor,
    cond: torch.Tensor,
    thr: float | None,
) -> list[dict[str, torch.Tensor]]:
    """Decode latents + conditions -> list of tensor dicts."""
    out = model.generate(cond, z, occ_threshold=thr)
    return [{_OUT_KEYS[k]: v[i] for k, v in out.items()} for i in range(z.shape[0])]


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
    p.add_argument("--num-samples", type=int, default=8, help="Samples (per style, if conditioned)")
    p.add_argument("--num-recon", type=int, default=8, help="Houses to reconstruct if there is no val split")
    p.add_argument("--interp-steps", type=int, default=6)
    p.add_argument("--ascii", type=int, default=2, help="How many houses/samples to print as ASCII")
    p.add_argument(
        "--threshold", type=float, default=None,
        help="Occupancy threshold (density dial). Default: the value tuned on validation during "
             "training, or plain argmax for older checkpoints",
    )
    p.add_argument("--hybrid", default="desert+taiga", help="Two biome styles to blend, e.g. desert+snowy")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(_parse_args())
