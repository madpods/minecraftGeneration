"""
Style blending: turn one style into another gradually.

Needs a style-conditioned checkpoint (ml.train --condition style). For a few
random building shapes (latent z), it produces a series of structures that go
from style A to style B, with three methods:

  condition  Average the two style vectors and decode once. The model has only
             seen the pure styles, so the halfway vector is unfamiliar ground
             and the whole building tends to flip at once ("snap").

  logit      Decode the shape in both styles and average the per-voxel
             log-probabilities. Each voxel flips at its own point, so the mix
             is gradual but can look speckled.

  swap       Material translation learned from the model. Decoding the same z
             in both styles gives the same shape (z = shape, style =
             materials), so voxels line up and the model tells us what each
             style-A material becomes in style B (sandstone wall -> spruce
             plank wall, sandstone stairs roof -> spruce stairs roof, ...).
             The blend keeps style A's building and swaps whole materials over
             in order of height — roof and trim first, walls next, floor last —
             so the halfway point is a coherent hybrid (e.g. a sandstone house
             with a spruce roof). This is the faction "cultural exchange"
             operation.

Writes .nbt structure files and prints, per method and step, how much of the
building uses style-A-typical vs style-B-typical blocks.

    python -m ml.blend --checkpoint ml/checkpoints/local/style_joint/best.pt \\
        --a desert --b taiga --out ml/samples/blends
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from ml.data.structure_export import ascii_slices, save_structure
from ml.model.vae import ConditionalChunkVAE, decode_block_ids
from ml.train import AIR_ID

_KEYS = {"block_id": "block_ids", "facing": "facing", "axis": "axis", "half": "half", "shape": "shape"}


def _decode_logits(model: ConditionalChunkVAE, z: torch.Tensor, style_vec: torch.Tensor) -> dict[str, torch.Tensor]:
    cond = model.style_condition(style_vec.expand(z.shape[0], -1))
    return model.decoder(torch.cat([z, cond], dim=-1))


def _to_tensors(logits: dict[str, torch.Tensor], thr: float | None) -> list[dict[str, torch.Tensor]]:
    out = {_KEYS[k]: v.argmax(1) for k, v in logits.items()}
    out["block_ids"] = decode_block_ids(logits["block_id"], thr)
    return [{k: v[i] for k, v in out.items()} for i in range(out["block_ids"].shape[0])]


# Blocks with a job (storage, beds, light, workstations, water, decoration):
# a culture change restyles the building, it doesn't turn a bed into a wall.
_FUNCTIONAL_WORDS = (
    "chest", "barrel", "bed", "torch", "lantern", "campfire", "water", "lava", "crafting_table",
    "furnace", "smoker", "bell", "composter", "lectern", "loom", "cartography_table",
    "fletching_table", "smithing_table", "stonecutter", "grindstone", "brewing_stand", "cauldron",
    "anvil", "hay_block", "flower_pot", "potted_", "bookshelf", "jukebox", "note_block", "ladder",
    "pressure_plate", "button", "lever", "rail", "sign", "banner", "carpet", "wool", "melon",
    "pumpkin", "wheat", "carrots", "potatoes", "beetroots", "cactus", "flower", "poppy",
    "dandelion", "tulip", "bush", "fern", "grass", "sugar_cane", "vine", "leaves", "sapling",
)
_GROUND = {"sand", "red_sand", "dirt", "coarse_dirt", "grass_block", "dirt_path", "podzol",
           "farmland", "gravel", "snow_block", "snow", "mud", "rooted_dirt"}


def block_kind(name: str) -> str:
    """Coarse shape/role class of a block, so materials only swap like for like."""
    n = name.split(":")[-1]
    if n in _GROUND:
        return "ground"
    if any(w in n for w in _FUNCTIONAL_WORDS):
        return "functional"
    for suffix in ("_stairs", "_slab", "_fence_gate", "_fence", "_trapdoor", "_door", "_pane", "_wall"):
        if n.endswith(suffix):
            return suffix[1:]
    if n.endswith(("_log", "_wood")) or n.startswith("stripped_"):
        return "pillar"
    if "glass" in n:
        return "glass"
    return "full"


def learn_translation(
    a_blocks: torch.Tensor,
    b_blocks: torch.Tensor,
    rev_vocab: dict[int, str],
    b_training_freq: Counter | None = None,
) -> dict[int, int]:
    """
    For every style-A material, its style-B counterpart, by first match of:
      1. the same-kind B block that most often occupies the same voxel when
         the same shapes are decoded in style B (the model's own answer)
      2. the most common same-kind block in the decoded B buildings
      3. the most common same-kind block in real style-B training houses
    Functional blocks, and materials with no same-kind counterpart, are kept.
    a_blocks, b_blocks: [N, X, Y, Z] block ids from the same z's.
    """
    kind = lambda bid: block_kind(rev_vocab.get(bid, "minecraft:stone"))
    both = (a_blocks != AIR_ID) & (b_blocks != AIR_ID)
    pairs = Counter(zip(a_blocks[both].tolist(), b_blocks[both].tolist()))
    sources = [Counter(b_blocks[b_blocks != AIR_ID].tolist()), b_training_freq or Counter()]

    translation: dict[int, int] = {}
    for a in set(a_blocks[a_blocks != AIR_ID].tolist()):
        k = kind(a)
        if k == "functional":
            translation[a] = a
            continue
        candidates = {b: n for (aa, b), n in pairs.items() if aa == a and kind(b) == k}
        for freq in sources:
            if candidates:
                break
            candidates = {b: n for b, n in freq.items() if kind(b) == k}
        translation[a] = max(candidates, key=candidates.get) if candidates else a
    return translation


def swap_series(
    house: dict[str, torch.Tensor],
    translation: dict[int, int],
    steps: list[float],
) -> list[dict[str, torch.Tensor]]:
    """
    Gradually swap a style-A house's materials to their style-B partners,
    highest-placed materials (roof, trim) first, in proportion to voxel count.
    """
    b = house["block_ids"]
    mats = [m for m in torch.unique(b).tolist() if m != AIR_ID and m in translation and translation[m] != m]
    ys = torch.arange(b.shape[1]).view(1, -1, 1).expand_as(b)
    mean_y = {m: ys[b == m].float().mean().item() for m in mats}
    order = sorted(mats, key=lambda m: -mean_y[m])
    counts = {m: int((b == m).sum()) for m in order}
    total = max(sum(counts.values()), 1)

    series = []
    for w in steps:
        out = {k: v.clone() for k, v in house.items()}
        done = 0
        for m in order:
            # swap a material once the blend weight reaches the middle of its share
            if (done + counts[m] / 2) / total <= w + 1e-9:
                out["block_ids"][b == m] = translation[m]
            done += counts[m]
        series.append(out)
    return series


class PairwisePalette:
    """Labels each block as typical of style A, of style B, or shared."""

    def __init__(self, data_dir: Path, a: str, b: str) -> None:
        index = json.loads((data_dir / "index.json").read_text())
        freq = {a: Counter(), b: Counter()}
        for fname, meta in index.items():
            if meta["biome"] in freq:
                t = torch.load(data_dir / fname, weights_only=True)["block_ids"].long()
                freq[meta["biome"]].update(t[t != AIR_ID].tolist())
        self.freq = freq          # block usage per style in the real training houses
        tot = {k: max(sum(v.values()), 1) for k, v in freq.items()}
        self.label: dict[int, str] = {}
        for bid in set(freq[a]) | set(freq[b]):
            fa, fb = freq[a][bid] / tot[a], freq[b][bid] / tot[b]
            share = fa / (fa + fb)
            self.label[bid] = "A" if share >= 0.75 else "B" if share <= 0.25 else "shared"

    def composition(self, blocks: torch.Tensor) -> tuple[float, float]:
        solid = blocks[blocks != AIR_ID].tolist()
        c = Counter(self.label.get(x, "shared") for x in solid)
        n = max(len(solid), 1)
        return c["A"] / n, c["B"] / n


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if ckpt.get("config", {}).get("condition") != "style":
        raise SystemExit("This checkpoint is not style-conditioned (train with --condition style)")
    model = ConditionalChunkVAE.from_checkpoint(ckpt).eval()
    thr = ckpt.get("metrics", {}).get("val", {}).get("occ_threshold")
    data_dir = Path(args.data_dir or ckpt["config"]["data_dir"])
    vocab = json.loads((ckpt_path.parent / "vocab.json").read_text())
    rev = {v: k for k, v in vocab.items()}
    index = json.loads((data_dir / "index.json").read_text())
    style_of = {m["biome"]: m["style"] for m in index.values()}
    va = model.style_embed(torch.tensor([style_of[args.a]]))
    vb = model.style_embed(torch.tensor([style_of[args.b]]))
    palette = PairwisePalette(data_dir, args.a, args.b)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    steps = [i / (args.steps - 1) for i in range(args.steps)]

    # Material translation A -> B, learned from many shapes decoded in both styles
    zt = torch.randn(args.translation_samples, model.latent_dim)
    a_blocks = decode_block_ids(_decode_logits(model, zt, va)["block_id"], thr)
    b_blocks = decode_block_ids(_decode_logits(model, zt, vb)["block_id"], thr)
    translation = learn_translation(a_blocks, b_blocks, rev, palette.freq[args.b])
    occ_a, occ_b = a_blocks != AIR_ID, b_blocks != AIR_ID
    shape_iou = ((occ_a & occ_b).sum() / (occ_a | occ_b).sum().clamp(min=1)).item()
    print(f"Same shape decoded as {args.a} vs {args.b}: occupancy IoU {shape_iou:.2f} "
          f"(1.0 = identical shape, so materials line up voxel for voxel)")
    common = Counter(a_blocks[a_blocks != AIR_ID].tolist()).most_common(16)
    print(f"\nLearned material translation {args.a} -> {args.b} (most used {args.a} blocks):")
    for a, _ in common:
        name_a = rev.get(a, a).split(":")[-1]
        b = translation.get(a, a)
        name_b = "(kept)" if b == a else rev.get(b, "?").split(":")[-1]
        print(f"  {name_a:28s} -> {name_b:24s} [{block_kind(rev.get(a, ''))}]")

    z = torch.randn(args.num_shapes, model.latent_dim)
    la, lb = _decode_logits(model, z, va), _decode_logits(model, z, vb)
    houses_a = _to_tensors(la, thr)

    results: dict[str, list[list[tuple[float, float]]]] = {"condition": [], "logit": [], "swap": []}
    for s in range(args.num_shapes):
        zs = z[s:s + 1]
        series = {
            "condition": [_to_tensors(_decode_logits(model, zs, (1 - w) * va + w * vb), thr)[0] for w in steps],
            "logit": [
                _to_tensors({k: (1 - w) * la[k][s:s + 1].log_softmax(1) + w * lb[k][s:s + 1].log_softmax(1)
                             for k in la}, thr)[0]
                for w in steps
            ],
            "swap": swap_series(houses_a[s], translation, steps),
        }
        for method, items in series.items():
            comp = []
            for w, t in zip(steps, items):
                comp.append(palette.composition(t["block_ids"]))
                save_structure(out / f"blend_{method}_shape{s}_{args.a}{round(100 * (1 - w))}_{args.b}{round(100 * w)}.nbt",
                               t, rev)
            results[method].append(comp)

    print(f"\nShare of blocks typical of {args.a} / {args.b}, averaged over {args.num_shapes} shapes")
    print(f"{'method':10s} " + " ".join(f"{f'{round(100*(1-w))}/{round(100*w)}':>14s}" for w in steps))
    for method, per_shape in results.items():
        cells = []
        for i in range(len(steps)):
            a_share = sum(c[i][0] for c in per_shape) / len(per_shape)
            b_share = sum(c[i][1] for c in per_shape) / len(per_shape)
            cells.append(f"{a_share:5.0%} / {b_share:<5.0%}")
        print(f"{method:10s} " + " ".join(f"{c:>14s}" for c in cells))

    # Midpoint of shape 0, all three methods side by side
    mid = len(steps) // 2
    grids = []
    for method in ("condition", "logit", "swap"):
        f = out / f"blend_{method}_shape0_{args.a}{round(100 * (1 - steps[mid]))}_{args.b}{round(100 * steps[mid])}.nbt"
        from ml.data.template_extractor import parse_template
        t = parse_template(f.read_bytes(), "x", "x", False, method)
        g = torch.zeros(16, 16, 16, dtype=torch.long)
        for x, y, zz, name, _ in t.blocks:
            g[x, y, zz] = vocab.get(name, 1)
        grids.append(g)
    ys = [y for y in range(16) if (grids[2][:, y, :] != AIR_ID).any()]
    pick = [ys[0], ys[len(ys) // 2], ys[-1]] if len(ys) >= 3 else ys
    print(f"\n--- shape 0 at the 50/50 point: condition | logit | swap ---")
    print(ascii_slices(grids, rev, pick, ["condition", "logit", "swap"]))
    print(f"\nStructure files written to {out}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Blend two building styles gradually")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--a", default="desert", help="Start style (biome name)")
    p.add_argument("--b", default="taiga", help="End style (biome name)")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--num-shapes", type=int, default=3)
    p.add_argument("--translation-samples", type=int, default=64,
                   help="Random shapes used to learn the A->B material translation")
    p.add_argument("--out", default="ml/samples/blends")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
