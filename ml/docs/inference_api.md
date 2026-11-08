# Inference API

The Flask inference server exposes a REST API consumed by the Node.js tile generation server (Phase 3). It loads the latest model checkpoint on startup and watches the faction store JSON for on-disk changes.

## Starting the server

```bash
python -m ml.inference
```

Or with custom paths:

```bash
ML_CHECKPOINT_DIR=ml/checkpoints \
ML_FACTION_STORE=ml/factions.json \
ML_PORT=5000 \
python -m ml.inference
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `ML_CHECKPOINT_DIR` | `ml/checkpoints` | Loads `best.pt` if present, otherwise the latest `checkpoint_epoch_*.pt`. `vocab.json` is read from here too |
| `ML_USE_CONDITION_BUILDER` | `0` | Set to `1` only once conditional training exists. Until then the server uses a zero condition vector, matching training — the `ConditionBuilder` is untrained and its output would be noise to the decoder, so `tile_type`, `purpose`, `size` and face fields currently have no effect |
| `ML_FACTION_STORE` | `ml/factions.json` | Path to the faction store JSON; reloaded automatically when modified |
| `ML_PORT` | `5000` | Port to listen on |

If no checkpoint exists, the server starts with a randomly initialised model and logs a warning. This is acceptable for development but will produce meaningless schematics.

### Faction store auto-reload

The server checks the modification time of `ML_FACTION_STORE` on every request to `/generate` and `/faction/event`. If the file has changed since the last load, it is reloaded before processing the request. This allows external processes (e.g. a world event system) to update the faction store without restarting the server.

---

## Routes

### GET /health

Returns server status.

**Response:**

```json
{
  "status":        "ok",
  "model_loaded":  true,
  "faction_count": 12
}
```

`model_loaded` is `false` only if initialisation failed (should not occur in normal operation).

---

### POST /generate

Generate a 16×16×16 schematic tile conditioned on a faction identity and tile metadata.

**Request:**

```json
{
  "faction_id":  "3f8a2b1c-4d5e-6f7a-8b9c-0d1e2f3a4b5c",
  "tile_type":   3,
  "purpose":     7,
  "size":        0.6,
  "variance":    0.4,
  "face_north":  "OPEN",
  "face_south":  "WALL",
  "face_east":   "DOOR",
  "face_west":   "WALL",
  "face_up":     "NONE",
  "face_down":   "FLOOR",
  "style":       "desert"
}
```

`style` (optional) applies only to style-conditioned checkpoints (`--condition style`): a biome name from the training `index.json` (`desert`, `plains`, `savanna`, `snowy`, `taiga`) or its integer id. Without it, the server uses the average of all known style vectors. An unknown name returns HTTP 400. The faction's sampled build vector is still used as the latent z (variation within the style).

The occupancy threshold tuned during training (density dial, see the training guide) is applied automatically when the checkpoint has one.

| Field | Type | Required | Description |
|---|---|---|---|
| `faction_id` | string (UUID) | Yes | Identifies the building faction; created on first encounter |
| `tile_type` | int 0–19 | Yes | Tile type index (see tile type table below) |
| `purpose` | int 0–29 | Yes | Purpose index (throne_room, storage, forge, etc.) |
| `size` | float 0–1 | Yes | Relative size influence |
| `variance` | float 0–1 | No | Overrides the faction's stored variance for this request |
| `face_*` | string | Yes (×6) | Interface requirement for each face: `OPEN`, `DOOR`, `WALL`, `FLOOR`, `NONE` |

Face values:
- `OPEN` — the face must be passable (no blocks at the boundary)
- `DOOR` — the face must have a door or archway
- `WALL` — the face must be solid
- `FLOOR` — the face is a floor/ceiling (treated as WALL for conditioning)
- `NONE` — no constraint

**Response:**

```json
{
  "schematic": [
    [
      [
        { "id": "minecraft:oak_slab", "facing": "none", "axis": "none", "half": "top", "shape": "none",
          "properties": { "type": "top" } },
        { "id": "minecraft:air",      "facing": "none", "axis": "none", "half": "none", "shape": "none",
          "properties": {} },
        ...
      ],
      ...
    ],
    ...
  ],
  "generation_time_ms": 14
}
```

The schematic is a 3D array indexed `[x][y][z]`, each element a block state dict. All 16×16×16 = 4096 voxels are present. `minecraft:air` represents empty space.

Block state fields:
- `id` — Minecraft namespaced block ID (e.g. `minecraft:oak_stairs`)
- `facing` — one of: `north`, `south`, `east`, `west`, `up`, `down`, `none`
- `axis` — one of: `x`, `y`, `z`, `none`
- `half` — one of: `top`, `bottom`, `double`, `upper`, `lower`, `head`, `foot`, `none` (raw model channel)
- `shape` — one of: `straight`, `inner_left`, `inner_right`, `outer_left`, `outer_right`, `left`, `right`, `none` (raw model channel)
- `properties` — the same information as real Minecraft property names for this block (e.g. slab → `type`, door → `half` + `hinge`, bed → `part`, chest → `type`). This is what the mod should apply to the `BlockState`.

If the model was trained without a vocabulary file, block IDs are returned as `unknown:N` strings.

---

### GET /faction/:id

Return the serialised faction identity for the given UUID. Creates the faction (with a random home vector) if it does not exist.

**Response:**

```json
{
  "faction_id":  "3f8a2b1c-...",
  "home_vector": [0.142, -0.891, 0.324, ...],
  "variance":    0.3
}
```

`home_vector` has 128 elements. This endpoint is intended for inspection and debugging — the vector is not consumed directly by the Fabric mod.

---

### POST /faction/event

Apply a world event to one or more factions. The server performs the vector arithmetic and persists the updated store to disk.

**Request — cultural_exchange:**

```json
{
  "type":       "cultural_exchange",
  "faction_a":  "uuid-a",
  "faction_b":  "uuid-b",
  "influence":  0.05
}
```

**Request — conquest:**

```json
{
  "type":         "conquest",
  "conqueror":    "uuid-winner",
  "conquered":    "uuid-loser",
  "assimilation": 0.3
}
```

**Request — schism:**

```json
{
  "type":     "schism",
  "parent":   "uuid-parent",
  "child_id": "uuid-child",
  "drift":    0.5
}
```

**Request — alliance:**

```json
{
  "type":       "alliance",
  "faction_a":  "uuid-a",
  "faction_b":  "uuid-b",
  "merged_id":  "uuid-merged"
}
```

**Response:**

```json
{
  "success": true,
  "result": {
    "updated": [
      { "faction_id": "uuid-a", "home_vector": [...] },
      { "faction_id": "uuid-b", "home_vector": [...] }
    ]
  }
}
```

For `schism` and `alliance`, the `result` key contains `"created"` rather than `"updated"`.

---

## Tile type index reference

| Index | Name | Index | Name |
|---|---|---|---|
| 0 | WALL | 10 | FLOOR |
| 1 | CORNER | 11 | ROOM_SMALL |
| 2 | GATE | 12 | ROOM_LARGE |
| 3 | TOWER_BASE | 13 | HALLWAY_H |
| 4 | TOWER_TOP | 14 | HALLWAY_V |
| 5 | PARAPET | 15 | STAIRWELL |
| 6 | DOORWAY | 16 | THRONE_ROOM |
| 7 | WINDOW | 17 | FORGE |
| 8 | ARCH | 18 | LIBRARY |
| 9 | DUNGEON_CELL | 19 | STORAGE |

These indices must match what the Node.js server sends. The model embeds tile type as a learned 32-dimensional vector, so the mapping is learned — reordering indices would require retraining.
