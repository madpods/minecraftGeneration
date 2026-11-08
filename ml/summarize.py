"""
Summarise a set of training runs into one comparison table.

Reads <runs-dir>/<run>/best.pt (validation metrics at the tuned occupancy
threshold) and <run>/eval.txt (sample checks from ml.evaluate), groups runs
that differ only by a trailing seed suffix (_s0, _s1, ...), and prints
mean ± spread per configuration.

    python -m ml.summarize ml/checkpoints
"""

from __future__ import annotations

import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path

import torch

_SEED = re.compile(r"_s\d+$")
_STYLE_HITS = re.compile(r"-> (\d+)/(\d+) samples look like the biome")
_NN = re.compile(r"nearest: \S+ \(IoU ([0-9.]+)\)")


def _run_row(run_dir: Path) -> dict | None:
    best = run_dir / "best.pt"
    if not best.exists():
        return None
    ckpt = torch.load(best, map_location="cpu", weights_only=False)
    val = ckpt.get("metrics", {}).get("val", {})
    at = val.get("at_threshold", {})
    row = {
        "epoch":        ckpt.get("epoch"),
        "solid_acc":    at.get("solid_acc"),
        "solid_recall": at.get("solid_recall"),
        "precision":    at.get("precision"),
        "fill":         at.get("fill"),
        "thr":          val.get("occ_threshold"),
    }
    report = run_dir / "eval.txt"
    if report.exists():
        text = report.read_text(encoding="utf-8", errors="replace")
        hits = _STYLE_HITS.search(text)
        if hits:
            row["style_hit"] = int(hits.group(1)) / int(hits.group(2))
        ious = [float(x) for x in _NN.findall(text)]
        if ious:
            row["sample_nn_iou"] = statistics.median(ious)
    return row


def summarize(runs_dir: Path) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        row = _run_row(d)
        if row is not None:
            groups[_SEED.sub("", d.name)].append(row)
    if not groups:
        print(f"No runs with best.pt under {runs_dir}")
        return

    cols = ["solid_acc", "solid_recall", "precision", "fill", "thr", "style_hit", "sample_nn_iou", "epoch"]
    print("Validation metrics at the tuned occupancy threshold, mean +/- (max-min)/2 over seeds")
    print("style_hit = conditioned samples classified as the requested biome; "
          "sample_nn_iou = median IoU of samples to their nearest training house\n")
    header = f"{'config':34s} {'n':>2s} " + " ".join(f"{c:>15s}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, rows in sorted(groups.items()):
        cells = []
        for c in cols:
            vals = [r[c] for r in rows if r.get(c) is not None]
            if not vals:
                cells.append(f"{'-':>15s}")
            elif len(vals) == 1:
                cells.append(f"{vals[0]:>15.3f}" if c != "epoch" else f"{vals[0]:>15d}")
            else:
                m = statistics.mean(vals)
                h = (max(vals) - min(vals)) / 2
                cells.append(f"{m:>7.3f} +/-{h:<4.3f}" if c != "epoch" else f"{m:>15.0f}")
        print(f"{name:34s} {len(rows):>2d} " + " ".join(cells))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Summarise training runs")
    p.add_argument("runs_dir", nargs="?", default="ml/checkpoints")
    summarize(Path(p.parse_args().runs_dir))
