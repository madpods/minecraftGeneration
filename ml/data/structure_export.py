"""
Convert model tensors back into Minecraft data you can look at.

    tensors_to_blocks()  -> list of (x, y, z, name, props) for non-air voxels
    save_structure()     -> vanilla structure .nbt file (load with a Structure Block)
    ascii_slices()       -> quick text view of horizontal slices, for terminals

Viewing a saved structure in game (Java 1.20.1):
    1. Copy the .nbt into <world>/generated/minecraft/structures/
    2. /give @s structure_block, place it, switch it to LOAD mode
    3. Structure name: minecraft:<file name without .nbt>, then LOAD
"""

from __future__ import annotations

from pathlib import Path

import torch

from .block_encoding import decode_properties

DATA_VERSION_1_20_1 = 3465
_KEYS = ("block_ids", "facing", "axis", "half", "shape")


def tensors_to_blocks(
    tensors: dict[str, torch.Tensor],
    rev_vocab: dict[int, str],
    air_id: int = 0,
) -> list[tuple[int, int, int, str, dict[str, str]]]:
    """List every non-air voxel as (x, y, z, block_name, properties)."""
    grids = {k: tensors[k].long().cpu() for k in _KEYS}
    blocks = []
    for x, y, z in (grids["block_ids"] != air_id).nonzero().tolist():
        name = rev_vocab.get(grids["block_ids"][x, y, z].item(), "minecraft:stone")
        props = decode_properties(
            name,
            grids["facing"][x, y, z].item(),
            grids["axis"][x, y, z].item(),
            grids["half"][x, y, z].item(),
            grids["shape"][x, y, z].item(),
        )
        blocks.append((x, y, z, name, props))
    return blocks


def save_structure(
    path: str | Path,
    tensors: dict[str, torch.Tensor],
    rev_vocab: dict[int, str],
    include_air: bool = True,
) -> Path:
    """
    Write tensors as a gzipped vanilla structure file.

    include_air=True writes explicit air so loading the structure clears the
    16³ volume first (otherwise it merges with whatever is already there).
    """
    from nbtlib import Compound, File, Int, List, String

    blocks = tensors_to_blocks(tensors, rev_vocab)
    size = tuple(tensors["block_ids"].shape)

    palette_index: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}
    palette: list[Compound] = []

    def state_id(name: str, props: dict[str, str]) -> int:
        key = (name, tuple(sorted(props.items())))
        if key not in palette_index:
            palette_index[key] = len(palette)
            entry = Compound({"Name": String(name)})
            if props:
                entry["Properties"] = Compound({k: String(v) for k, v in props.items()})
            palette.append(entry)
        return palette_index[key]

    block_entries = []
    if include_air:
        air = state_id("minecraft:air", {})
        solid = {(x, y, z) for x, y, z, _, _ in blocks}
        for x in range(size[0]):
            for y in range(size[1]):
                for z in range(size[2]):
                    if (x, y, z) not in solid:
                        block_entries.append(Compound({"pos": List[Int]([x, y, z]), "state": Int(air)}))
    for x, y, z, name, props in blocks:
        block_entries.append(Compound({"pos": List[Int]([x, y, z]), "state": Int(state_id(name, props))}))

    root = File({
        "DataVersion": Int(DATA_VERSION_1_20_1),
        "size":        List[Int](list(size)),
        "palette":     List[Compound](palette),
        "blocks":      List[Compound](block_entries),
        "entities":    List[Compound]([]),
    })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root.save(path, gzipped=True)
    return path


_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789#$%&*+=?@"


def ascii_slices(
    grids: list[torch.Tensor],
    rev_vocab: dict[int, str],
    ys: list[int],
    titles: list[str] | None = None,
    air_id: int = 0,
) -> str:
    """
    Render horizontal slices of one or more block_id grids side by side.

    Each slice is a top-down 16×16 view (rows = Z from north to south,
    columns = X from west to east). Air is '.', other blocks get a letter
    from a shared legend.
    """
    legend: dict[int, str] = {}
    for g in grids:
        for bid in torch.unique(g).tolist():
            if bid != air_id and bid not in legend:
                legend[bid] = _CHARS[len(legend) % len(_CHARS)]

    titles = titles or [f"grid {i}" for i in range(len(grids))]
    sx, _, sz = grids[0].shape
    lines: list[str] = []
    for y in ys:
        lines.append(f"y={y}".ljust(4) + "   ".join(t.ljust(sx) for t in titles))
        for z in range(sz):
            row = []
            for g in grids:
                row.append("".join(
                    "." if g[x, y, z].item() == air_id else legend[g[x, y, z].item()]
                    for x in range(sx)
                ))
            lines.append("    " + "   ".join(row))
        lines.append("")

    lines.append("legend: " + "  ".join(
        f"{c}={str(rev_vocab.get(b, b)).split(':')[-1]}" for b, c in legend.items()
    ))
    return "\n".join(lines)
