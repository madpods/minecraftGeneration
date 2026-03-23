# Faction System

## Concept

A faction's architectural style is encoded as a single point in the VAE's 128-dimensional latent space, called the **home vector**. When an NPC builds a structure, a vector is sampled near that point; the decoder turns it into block data. Over time, world events push home vectors toward or away from each other, causing factions to develop shared or divergent aesthetics organically.

No retraining is needed. All style evolution happens through vector arithmetic on a fixed, pre-trained model.

## FactionIdentity

File: `ml/model/faction.py`

```python
class FactionIdentity:
    faction_id:  str            # UUID
    home_vector: torch.Tensor   # [128]  float32
    variance:    float          # how far individual builds can stray
```

### Initialisation

```python
faction = FactionIdentity(faction_id="my-faction", variance=0.3)
```

`home_vector` is initialised by sampling `N(0, I)`. All factions start at random positions in latent space.

### Sampling a build vector

```python
z = faction.sample_build_vector()   # [128]
```

Returns `home_vector + N(0, variance * I)`. Pass `z` to `ConditionalChunkVAE.generate()` along with a condition vector to get a schematic.

A lower variance (e.g. 0.1) produces buildings that all look nearly identical — useful for militaristic factions with strict construction standards. A higher variance (e.g. 0.6) produces eclectic, varied architecture.

### Learning from completed buildings

```python
faction.record_build(vae, chunk_tensors, lr=0.01)
```

After an NPC finishes placing a building, call this to nudge the home vector toward what was actually built. Internally it encodes the finished chunk with a neutral condition and applies an exponential moving average update:

```
home_vector += lr * (encoded_mu - home_vector)
```

This means repeated building in a consistent style will reinforce that style in the faction's identity, while varied builds will average out.

`chunk_tensors` is a tuple of `(block_ids, facing, axis, half, shape)` tensors, each `[1, 16, 16, 16]`. The `vae` argument is the trained `ConditionalChunkVAE` instance.

## World events

### cultural_exchange

Two factions nudge toward each other. Appropriate after trade or diplomatic contact.

```python
faction_a.cultural_exchange(faction_b, influence=0.05)
```

Both home vectors move toward each other by `influence * distance`. The default `influence=0.05` means each interaction closes 5% of the gap — subtle enough that a single trade deal doesn't dramatically change architecture, but repeated contact over many in-game years will produce visible convergence.

### conquest

The conquered faction partially adopts the conqueror's style.

```python
conquered_faction.conquest(conqueror_faction, assimilation=0.3)
```

`self` is the conquered faction. Its home vector moves `assimilation` fraction of the way toward the conqueror. The conqueror is unchanged. An `assimilation=0.3` (30%) represents significant cultural absorption while preserving some native character.

### spawn_splinter

A breakaway group inherits the parent faction's general style with added drift, and slightly higher variance to reflect cultural uncertainty.

```python
child = parent_faction.spawn_splinter(drift=0.5)
```

`drift` is the standard deviation of the Gaussian noise added to the home vector. The child starts as a separate `FactionIdentity` with a new UUID.

### alliance / merge

Two factions combine into a new entity with an averaged home vector and averaged variance. Both originals remain unchanged.

```python
merged = FactionIdentity.merge(faction_a, faction_b)
```

## FactionStore

File: `ml/model/faction.py`

`FactionStore` manages a collection of `FactionIdentity` objects and persists them to a JSON file. The inference server loads and auto-reloads this file.

```python
store = FactionStore.load("ml/factions.json")

# Get or create a faction by ID
faction = store.get("some-uuid")

# Save all factions
store.save("ml/factions.json")
```

### JSON format

```json
{
  "3f8a2b1c-...": {
    "faction_id":  "3f8a2b1c-...",
    "home_vector": [0.142, -0.891, 0.324, ...],
    "variance":    0.3
  }
}
```

### Applying events via the store

```python
store.apply_event("cultural_exchange",
    faction_a="uuid-a", faction_b="uuid-b", influence=0.05)

store.apply_event("conquest",
    conqueror="uuid-a", conquered="uuid-b", assimilation=0.3)

store.apply_event("schism",
    parent="uuid-a", child_id="uuid-child", drift=0.5)

store.apply_event("alliance",
    faction_a="uuid-a", faction_b="uuid-b", merged_id="uuid-merged")
```

Each call returns a dict describing what was created or updated. After calling `apply_event`, call `store.save()` to persist changes.

## Example: schism followed by partial reconciliation

```python
from ml.model.faction import FactionIdentity, FactionStore

store = FactionStore()

# Founding faction — established style
founders = store.get("founders")

# A religious schism occurs after 10 in-game years
reformers = founders.spawn_splinter(drift=0.4)
store._factions["reformers"] = reformers

# 50 in-game years later: trade resumed, some cultural blending
founders.cultural_exchange(reformers, influence=0.1)
founders.cultural_exchange(reformers, influence=0.1)
founders.cultural_exchange(reformers, influence=0.1)

# At this point the factions are closer in style than right after the schism,
# but still distinct — the drift from spawn_splinter persists.

store.save("ml/factions.json")
```

## Notes for future phases

- The Node.js server (Phase 3) will persist faction vectors to CouchDB and forward world events to the Python inference server.
- The Fabric mod (Phase 4) will trigger `record_build` calls after NPC construction completes and dispatch world events from the NPC relationship graph.
- The faction vector is never sent to the Fabric mod — only the resulting schematic.
