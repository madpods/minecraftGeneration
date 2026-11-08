"""
Extract vanilla village structure templates from a Minecraft client jar into
16×16×16 training tensors.

The client jar ships the hand-built pieces villages are assembled from as
gzipped NBT structure files under:

    data/minecraft/structures/village/<biome>/<category>/<name>.nbt
    data/minecraft/structures/village/<biome>/zombie/<category>/<name>.nbt

Biomes: plains, desert, savanna, snowy, taiga.
Categories: houses, streets, town_centers, terminators, villagers, ...

Unlike scraping a generated world, these templates contain exactly one
building each, with no surrounding terrain, so they make a clean first
training set. Each template is placed in an empty 16³ grid, centred in X/Z
and resting on y=0. Templates larger than 16 in any dimension are skipped.

Special blocks:
    structure_void        -> air  (means "leave the world as-is")
    jigsaw                -> its `final_state` (usually air or dirt_path)

Usage:
    python -m ml.data.template_extractor \\
        --jar "C:/Users/<you>/curseforge/minecraft/Install/versions/1.20.1/1.20.1.jar" \\
        --output-dir ml/data/chunks

Writes:
    <output-dir>/<biome>_[zombie_]<category>_<template>.pt   one per template
    <output-dir>/vocab.json    block name -> id (air = 0, rest sorted by name)
    <output-dir>/index.json    file name  -> {biome, style, category, zombie, template, size, solid_blocks}
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch

from ..model.block_state import AXIS_INDEX, CHUNK_SIZE, FACING_INDEX, HALF_INDEX, NUM_BLOCK_TYPES, SHAPE_INDEX
from .block_encoding import AIR_LIKE, encode_properties, parse_block_string

VILLAGE_PREFIX = "data/minecraft/structures/village/"
BIOMES = ("desert", "plains", "savanna", "snowy", "taiga")
# Style id per biome, used as the condition label for --condition style
STYLE_INDEX: dict[str, int] = {b: i for i, b in enumerate(BIOMES)}


@dataclass
class Template:
    """A parsed structure template: resolved (name, props) per non-air block."""
    biome:    str
    category: str
    zombie:   bool
    name:     str
    size:     tuple[int, int, int]
    blocks:   list[tuple[int, int, int, str, dict[str, str]]]   # x, y, z, name, props

    @property
    def file_stem(self) -> str:
        z = "zombie_" if self.zombie else ""
        return f"{self.biome}_{z}{self.category}_{self.name}"

    def fits(self, size: int = CHUNK_SIZE) -> bool:
        return all(d <= size for d in self.size)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_template(data: bytes, biome: str, category: str, zombie: bool, name: str) -> Template:
    """Parse one gzipped structure .nbt file into a Template."""
    import nbtlib  # imported lazily so the rest of ml/ doesn't require it

    root = nbtlib.File.parse(io.BytesIO(gzip.decompress(data)))
    if "size" not in root:          # some nbtlib versions wrap in an unnamed root
        root = root[""]

    size = tuple(int(v) for v in root["size"])
    palette = root["palette"] if "palette" in root else root["palettes"][0]
    resolved_palette = [
        (str(p["Name"]), {str(k): str(v) for k, v in (p.get("Properties") or {}).items()})
        for p in palette
    ]

    blocks: list[tuple[int, int, int, str, dict[str, str]]] = []
    for b in root["blocks"]:
        block_name, props = resolved_palette[int(b["state"])]
        if block_name == "minecraft:jigsaw":
            final_state = str((b.get("nbt") or {}).get("final_state", "minecraft:air"))
            block_name, props = parse_block_string(final_state)
        if block_name in AIR_LIKE:
            continue
        x, y, z = (int(v) for v in b["pos"])
        blocks.append((x, y, z, block_name, props))

    return Template(biome, category, zombie, name, size, blocks)  # type: ignore[arg-type]


def iter_jar_templates(
    jar_path: str | Path,
    categories: set[str],
    include_zombie: bool,
):
    """Yield Template objects for matching village templates in a client jar."""
    with zipfile.ZipFile(jar_path) as jar:
        for entry in sorted(jar.namelist()):
            if not (entry.startswith(VILLAGE_PREFIX) and entry.endswith(".nbt")):
                continue
            parts = entry[len(VILLAGE_PREFIX):-len(".nbt")].split("/")
            # parts: [biome, category, name] or [biome, "zombie", category, name]
            if parts[0] not in BIOMES:
                continue
            zombie = len(parts) == 4 and parts[1] == "zombie"
            if len(parts) != (4 if zombie else 3):
                continue
            category = parts[2] if zombie else parts[1]
            if category not in categories or (zombie and not include_zombie):
                continue
            yield parse_template(jar.read(entry), parts[0], category, zombie, parts[-1])


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def build_vocab(templates: list[Template]) -> dict[str, int]:
    """Deterministic vocabulary: air = 0, then all block names sorted."""
    names = sorted({b[3] for t in templates for b in t.blocks} - {"minecraft:air"})
    if len(names) + 1 > NUM_BLOCK_TYPES:
        raise ValueError(
            f"{len(names) + 1} distinct blocks exceeds NUM_BLOCK_TYPES={NUM_BLOCK_TYPES}"
        )
    vocab = {"minecraft:air": 0}
    vocab.update({n: i + 1 for i, n in enumerate(names)})
    return vocab


def template_to_tensors(t: Template, vocab: dict[str, int]) -> dict[str, torch.Tensor]:
    """Place a template centred (X/Z) on the floor (Y=0) of an empty 16³ grid."""
    if not t.fits():
        raise ValueError(f"{t.file_stem} is {t.size}, larger than {CHUNK_SIZE}³")

    C = CHUNK_SIZE
    out = {
        "block_ids": torch.zeros((C, C, C), dtype=torch.int16),
        "facing":    torch.full((C, C, C), FACING_INDEX["none"], dtype=torch.int16),
        "axis":      torch.full((C, C, C), AXIS_INDEX["none"],   dtype=torch.int16),
        "half":      torch.full((C, C, C), HALF_INDEX["none"],   dtype=torch.int16),
        "shape":     torch.full((C, C, C), SHAPE_INDEX["none"],  dtype=torch.int16),
    }
    ox = (C - t.size[0]) // 2
    oz = (C - t.size[2]) // 2

    for x, y, z, name, props in t.blocks:
        X, Y, Z = x + ox, y, z + oz
        facing, axis, half, shape = encode_properties(props)
        out["block_ids"][X, Y, Z] = vocab[name]
        out["facing"][X, Y, Z]    = facing
        out["axis"][X, Y, Z]      = axis
        out["half"][X, Y, Z]      = half
        out["shape"][X, Y, Z]     = shape

    return out


def extract_villages(
    jar_path: str | Path,
    output_dir: str | Path,
    categories: tuple[str, ...] = ("houses",),
    include_zombie: bool = False,
) -> dict[str, dict]:
    """
    Extract matching village templates to .pt files plus vocab.json/index.json.

    Returns the index dict (file name -> metadata).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    templates = list(iter_jar_templates(jar_path, set(categories), include_zombie))
    kept    = [t for t in templates if t.fits()]
    skipped = [t for t in templates if not t.fits()]

    vocab = build_vocab(kept)
    index: dict[str, dict] = {}
    for t in kept:
        fname = f"{t.file_stem}.pt"
        torch.save(template_to_tensors(t, vocab), out / fname)
        index[fname] = {
            "biome":        t.biome,
            "style":        STYLE_INDEX[t.biome],
            "category":     t.category,
            "zombie":       t.zombie,
            "template":     t.name,
            "size":         list(t.size),
            "solid_blocks": len(t.blocks),
        }

    (out / "vocab.json").write_text(json.dumps(vocab, indent=2))
    (out / "index.json").write_text(json.dumps(index, indent=2))

    print(f"Extracted {len(kept)} templates to {out} ({len(vocab)} block types).")
    for biome in BIOMES:
        n = sum(1 for m in index.values() if m["biome"] == biome)
        if n:
            print(f"  {biome:8s} {n}")
    if skipped:
        print("Skipped (larger than 16³): " + ", ".join(f"{t.file_stem} {t.size}" for t in skipped))
    return index


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract village templates from a Minecraft jar")
    p.add_argument("--jar", required=True, help="Path to a Minecraft client .jar (e.g. 1.20.1.jar)")
    p.add_argument("--output-dir", default="ml/data/chunks")
    p.add_argument(
        "--categories", default="houses",
        help="Comma-separated template categories (houses,town_centers,streets,...)",
    )
    p.add_argument(
        "--include-zombie", action="store_true",
        help="Also extract zombie-village variants (same layouts, decayed materials)",
    )
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    extract_villages(a.jar, a.output_dir, tuple(a.categories.split(",")), a.include_zombie)
