"""
Mapping between Minecraft block-state properties and the model's four
integer state channels (facing, axis, half, shape).

Used by every component that converts between real Minecraft data and model
tensors (template extractor, world scraper, structure exporter, inference
server), so the mapping is defined exactly once.

Channel semantics (see ml/model/block_state.py for the index tables):

    facing  <- `facing`                                (north/south/east/west/up/down)
    axis    <- `axis`                                  (x/y/z)
    half    <- `half`  top/bottom   (stairs, trapdoors)
               `half`  upper/lower  (doors, tall plants)
               `type`  top/bottom/double (slabs)
               `part`  head/foot    (beds)
    shape   <- `shape` straight/inner_*/outer_* (stairs)
               `hinge` left/right   (doors)
               `type`  left/right   (chests)

Properties that are not captured (dropped on encode): fence/pane/wall
connections (north/east/...=true — recomputed by the game from neighbours),
waterlogged, open, powered, lit, snow `layers`, farmland `moisture`, crop
`age`, lantern `hanging`, button/grindstone `face`, bell `attachment`.
"""

from __future__ import annotations

from ..model.block_state import AXIS_INDEX, FACING_INDEX, HALF_INDEX, SHAPE_INDEX

_FACING_NONE = FACING_INDEX["none"]
_AXIS_NONE   = AXIS_INDEX["none"]
_HALF_NONE   = HALF_INDEX["none"]
_SHAPE_NONE  = SHAPE_INDEX["none"]

_STAIR_SHAPES = {"straight", "inner_left", "inner_right", "outer_left", "outer_right"}

# Blocks that read as air for training purposes.
AIR_LIKE: frozenset[str] = frozenset({
    "minecraft:air",
    "minecraft:cave_air",
    "minecraft:void_air",
    "minecraft:structure_void",
})

_REV_FACING = {v: k for k, v in FACING_INDEX.items()}
_REV_AXIS   = {v: k for k, v in AXIS_INDEX.items()}
_REV_HALF   = {v: k for k, v in HALF_INDEX.items()}
_REV_SHAPE  = {v: k for k, v in SHAPE_INDEX.items()}


def encode_properties(props: dict[str, str] | None) -> tuple[int, int, int, int]:
    """
    Convert a block's property dict to (facing, axis, half, shape) indices.

    Unknown properties or values map to the channel's "none" index.
    """
    props = props or {}

    facing = FACING_INDEX.get(props.get("facing", "none"), _FACING_NONE)
    axis   = AXIS_INDEX.get(props.get("axis", "none"), _AXIS_NONE)

    half = _HALF_NONE
    if "half" in props:
        half = HALF_INDEX.get(props["half"], _HALF_NONE)          # top/bottom/upper/lower
    elif props.get("type") in ("top", "bottom", "double"):
        half = HALF_INDEX[props["type"]]                          # slab
    elif props.get("part") in ("head", "foot"):
        half = HALF_INDEX[props["part"]]                          # bed

    shape = _SHAPE_NONE
    if props.get("shape") in _STAIR_SHAPES:
        shape = SHAPE_INDEX[props["shape"]]
    elif props.get("hinge") in ("left", "right"):
        shape = SHAPE_INDEX[props["hinge"]]                       # door
    elif props.get("type") in ("left", "right"):
        shape = SHAPE_INDEX[props["type"]]                        # chest

    return facing, axis, half, shape


def decode_properties(
    block_name: str,
    facing: int,
    axis: int,
    half: int,
    shape: int,
) -> dict[str, str]:
    """
    Convert channel indices back to Minecraft property names for `block_name`.

    Only non-"none" channels produce a property. The game ignores properties a
    block doesn't have, so a stray value (e.g. facing on a plank) is harmless.
    """
    props: dict[str, str] = {}
    base = block_name.split(":", 1)[-1]

    f = _REV_FACING.get(facing, "none")
    if f != "none":
        props["facing"] = f

    a = _REV_AXIS.get(axis, "none")
    if a != "none":
        props["axis"] = a

    h = _REV_HALF.get(half, "none")
    if h in ("top", "bottom"):
        props["type" if base.endswith("_slab") else "half"] = h
    elif h == "double":
        props["type"] = "double"
    elif h in ("upper", "lower"):
        props["half"] = h
    elif h in ("head", "foot"):
        props["part"] = h

    s = _REV_SHAPE.get(shape, "none")
    if s in _STAIR_SHAPES:
        props["shape"] = s
    elif s in ("left", "right"):
        props["type" if base in ("chest", "trapped_chest") else "hinge"] = s

    return props


def parse_block_string(state: str) -> tuple[str, dict[str, str]]:
    """
    Parse a block-state string such as "minecraft:oak_log[axis=y]" into
    (name, properties). Used for jigsaw `final_state` values.
    """
    state = state.strip()
    if "[" not in state:
        return state, {}
    name, rest = state.split("[", 1)
    props: dict[str, str] = {}
    for pair in rest.rstrip("]").split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            props[k.strip()] = v.strip()
    return name, props
