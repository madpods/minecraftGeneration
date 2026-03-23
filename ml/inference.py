"""
Flask inference server for ConditionalChunkVAE.

The Node.js tile generation server calls this via POST /generate to request
a 16×16×16 schematic conditioned on faction identity + tile metadata.

Startup:
  - Loads the latest checkpoint from ML_CHECKPOINT_DIR (env var or default)
  - Loads FactionStore from ML_FACTION_STORE (env var or default)
  - Watches the faction store JSON for on-disk changes (mtime check per request)

POST /generate
  Request JSON:
    {
      "faction_id":  "uuid-string",
      "tile_type":   0-19,           // integer or name string
      "purpose":     0-29,           // integer or name string
      "size":        0.0-1.0,
      "variance":    0.0-1.0,        // optional, overrides faction variance
      "face_north":  "OPEN"|"DOOR"|"WALL"|"NONE",
      "face_south":  ...,
      "face_east":   ...,
      "face_west":   ...,
      "face_up":     ...,
      "face_down":   ...
    }

  Response JSON:
    {
      "schematic":           [[[{block state dict}, ...], ...], ...],
      "generation_time_ms":  12
    }

POST /faction/event
  Forwards world events to FactionIdentity vector arithmetic.
  Request / response: same contract as FactionStore.apply_event()

GET /health
  Returns {"status": "ok", "model_loaded": true, "faction_count": N}
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
from flask import Flask, Response, jsonify, request

from ml.model.block_state import (
    AXIS_INDEX,
    COND_DIM,
    FACING_INDEX,
    HALF_INDEX,
    LATENT_DIM,
    SHAPE_INDEX,
)
from ml.model.faction import FactionStore
from ml.model.vae import ConditionalChunkVAE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINT_DIR:  str = os.environ.get("ML_CHECKPOINT_DIR", "ml/checkpoints")
FACTION_STORE_PATH: str = os.environ.get("ML_FACTION_STORE", "ml/factions.json")

# ---------------------------------------------------------------------------
# Reverse vocab (integer → namespaced block ID)
# Used to convert model outputs back to Minecraft block name strings.
# ---------------------------------------------------------------------------

_reverse_vocab: dict[int, str] = {}


def _load_vocab(checkpoint_dir: str) -> None:
    """Load the block vocabulary saved alongside training data."""
    global _reverse_vocab
    vocab_candidates = [
        Path(checkpoint_dir) / "vocab.json",
        Path("ml/data/chunks/vocab.json"),
    ]
    import json
    for vp in vocab_candidates:
        if vp.exists():
            vocab: dict[str, int] = json.loads(vp.read_text())
            _reverse_vocab = {v: k for k, v in vocab.items()}
            print(f"Loaded vocab: {len(_reverse_vocab)} blocks from {vp}")
            return
    print("Warning: no vocab.json found; block IDs will be returned as integers.")


# ---------------------------------------------------------------------------
# Face type mapping
# ---------------------------------------------------------------------------

FACE_TYPE_INDEX: dict[str, int] = {
    "OPEN":  0,
    "DOOR":  1,
    "WALL":  2,
    "NONE":  3,
    "FLOOR": 2,   # treated as WALL for conditioning purposes
}

# ---------------------------------------------------------------------------
# Model + faction store (module-level singletons)
# ---------------------------------------------------------------------------

_model: ConditionalChunkVAE | None = None
_faction_store: FactionStore = FactionStore()
_faction_store_mtime: float = 0.0
_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_latest_checkpoint(checkpoint_dir: str) -> Path | None:
    """Return the path to the most recently saved checkpoint, or None."""
    ckpts = sorted(Path(checkpoint_dir).glob("checkpoint_epoch_*.pt"))
    return ckpts[-1] if ckpts else None


def _load_model() -> None:
    global _model
    ckpt_path = _find_latest_checkpoint(CHECKPOINT_DIR)
    if ckpt_path is None:
        print(
            f"No checkpoint found in {CHECKPOINT_DIR}. "
            "Starting with a randomly initialised model."
        )
        _model = ConditionalChunkVAE().to(_device)
    else:
        ckpt = torch.load(ckpt_path, map_location=_device)
        _model = ConditionalChunkVAE().to(_device)
        _model.load_state_dict(ckpt["model_state"])
        print(f"Loaded model from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
    _model.eval()


def _reload_faction_store_if_changed() -> None:
    """Reload the faction store JSON if it has been modified on disk."""
    global _faction_store, _faction_store_mtime
    p = Path(FACTION_STORE_PATH)
    if not p.exists():
        return
    mtime = p.stat().st_mtime
    if mtime != _faction_store_mtime:
        _faction_store = FactionStore.load(p)
        _faction_store_mtime = mtime
        print(f"Reloaded faction store ({len(_faction_store)} factions)")


# ---------------------------------------------------------------------------
# Block state decoding
# ---------------------------------------------------------------------------

def _decode_block_state(
    block_id:  int,
    facing_id: int,
    axis_id:   int,
    half_id:   int,
    shape_id:  int,
) -> dict[str, str | None]:
    """Convert integer indices back to human-readable block state dict."""
    _rev_facing = {v: k for k, v in FACING_INDEX.items()}
    _rev_axis   = {v: k for k, v in AXIS_INDEX.items()}
    _rev_half   = {v: k for k, v in HALF_INDEX.items()}
    _rev_shape  = {v: k for k, v in SHAPE_INDEX.items()}

    ns_id = _reverse_vocab.get(block_id, f"unknown:{block_id}")

    return {
        "id":     ns_id,
        "facing": _rev_facing.get(facing_id, "none"),
        "axis":   _rev_axis.get(axis_id,   "none"),
        "half":   _rev_half.get(half_id,   "none"),
        "shape":  _rev_shape.get(shape_id, "none"),
    }


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health() -> Response:
    return jsonify({
        "status":       "ok",
        "model_loaded": _model is not None,
        "faction_count": len(_faction_store),
    })


@app.route("/generate", methods=["POST"])
def generate() -> Response:
    assert _model is not None, "Model not initialised"
    _reload_faction_store_if_changed()

    data: dict[str, Any] = request.get_json(force=True)

    # ------------------------------------------------------------------
    # Parse request
    # ------------------------------------------------------------------
    faction_id: str = data["faction_id"]
    tile_type:  int = int(data.get("tile_type", 0))
    purpose:    int = int(data.get("purpose",   0))
    size:     float = float(data.get("size",     0.5))

    face_keys = ["face_north", "face_south", "face_east", "face_west", "face_up", "face_down"]
    face_ids: list[int] = [
        FACE_TYPE_INDEX.get(str(data.get(k, "NONE")).upper(), 3)
        for k in face_keys
    ]

    # ------------------------------------------------------------------
    # Faction vector
    # ------------------------------------------------------------------
    faction = _faction_store.get(faction_id)
    variance_override = data.get("variance")
    if variance_override is not None:
        faction.variance = float(variance_override)

    faction_z = faction.sample_build_vector().unsqueeze(0).to(_device)   # [1, LATENT_DIM]

    # ------------------------------------------------------------------
    # Condition vector
    # ------------------------------------------------------------------
    def _scalar(val: float) -> torch.Tensor:
        return torch.tensor([val], dtype=torch.float32, device=_device)

    def _int(val: int) -> torch.Tensor:
        return torch.tensor([val], dtype=torch.long, device=_device)

    condition = _model.condition_builder.build_condition(
        tile_type      = _int(tile_type),
        purpose        = _int(purpose),
        face_n         = _int(face_ids[0]),
        face_s         = _int(face_ids[1]),
        face_e         = _int(face_ids[2]),
        face_w         = _int(face_ids[3]),
        face_u         = _int(face_ids[4]),
        face_d         = _int(face_ids[5]),
        size           = _scalar(size),
        variance       = _scalar(faction.variance),
        faction_vector = faction_z,
    )  # [1, COND_DIM]

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    decoded = _model.generate(condition, faction_z)   # dict of [1,16,16,16] int tensors
    gen_ms = int((time.perf_counter() - t0) * 1000)

    block_id_grid = decoded["block_id"][0].cpu().tolist()   # [16][16][16]
    facing_grid   = decoded["facing"][0].cpu().tolist()
    axis_grid     = decoded["axis"][0].cpu().tolist()
    half_grid     = decoded["half"][0].cpu().tolist()
    shape_grid    = decoded["shape"][0].cpu().tolist()

    # Build schematic as 3D list of block state dicts [x][y][z]
    schematic: list[list[list[dict[str, str | None]]]] = [
        [
            [
                _decode_block_state(
                    block_id_grid[x][y][z],
                    facing_grid[x][y][z],
                    axis_grid[x][y][z],
                    half_grid[x][y][z],
                    shape_grid[x][y][z],
                )
                for z in range(16)
            ]
            for y in range(16)
        ]
        for x in range(16)
    ]

    return jsonify({"schematic": schematic, "generation_time_ms": gen_ms})


@app.route("/faction/event", methods=["POST"])
def faction_event() -> Response:
    _reload_faction_store_if_changed()
    data: dict[str, Any] = request.get_json(force=True)
    event_type: str = data.pop("type")

    result = _faction_store.apply_event(event_type, **data)

    # Persist updated factions
    _faction_store.save(FACTION_STORE_PATH)

    return jsonify({"success": True, "result": result})


@app.route("/faction/<faction_id>", methods=["GET"])
def get_faction(faction_id: str) -> Response:
    _reload_faction_store_if_changed()
    faction = _faction_store.get(faction_id)
    return jsonify(faction.to_dict())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _load_model()
    _load_vocab(CHECKPOINT_DIR)

    # Load faction store
    global _faction_store, _faction_store_mtime
    p = Path(FACTION_STORE_PATH)
    if p.exists():
        _faction_store = FactionStore.load(p)
        _faction_store_mtime = p.stat().st_mtime
        print(f"Loaded {len(_faction_store)} factions from {p}")
    else:
        print(f"No faction store at {p}; starting fresh.")

    port = int(os.environ.get("ML_PORT", 5000))
    print(f"Starting inference server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
