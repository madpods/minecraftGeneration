"""
FactionIdentity and FactionStore — faction vector arithmetic and persistence.

A faction is represented as a home vector in VAE latent space plus a variance
scalar that controls architectural consistency across builds. No training is
required; all updates are pure vector arithmetic.

World event methods:
    cultural_exchange — bidirectional style drift between two factions
    conquest          — conquered faction partially adopts conqueror's style
    spawn_splinter    — create a child faction with a drifted home vector
    merge             — combine two factions into a new averaged faction

Serialization: FactionStore loads/saves all factions to a JSON file keyed by
faction_id. Individual FactionIdentity objects are serializable to/from dicts.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import torch

from .block_state import COND_DIM, LATENT_DIM


class FactionIdentity:
    """
    Represents a faction's stylistic identity as a vector in latent space.

    Attributes:
        faction_id:  Unique string identifier (UUID by default).
        home_vector: Float tensor [LATENT_DIM] representing the faction's
                     central architectural style.
        variance:    Scalar float controlling how much individual builds can
                     deviate from the home vector.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        variance: float = 0.3,
        faction_id: str | None = None,
    ) -> None:
        self.faction_id:  str           = faction_id or str(uuid.uuid4())
        self.home_vector: torch.Tensor  = torch.randn(latent_dim)
        self.variance:    float         = variance

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_build_vector(self) -> torch.Tensor:
        """Sample a build-specific latent vector near the faction home."""
        noise = torch.randn_like(self.home_vector) * self.variance
        return self.home_vector + noise

    # ------------------------------------------------------------------
    # Learning from built structures
    # ------------------------------------------------------------------

    def record_build(
        self,
        vae: Any,              # ConditionalChunkVAE — typed as Any to avoid circular import
        chunk_tensors: tuple[torch.Tensor, ...],
        lr: float = 0.01,
    ) -> None:
        """
        After a build completes, nudge the home vector toward what was built.

        Encodes the finished chunk with a neutral (zero) condition vector and
        applies an exponential moving average update to home_vector.

        Args:
            vae:           The trained ConditionalChunkVAE.
            chunk_tensors: Tuple of (block_ids, facing, axis, half, shape) tensors,
                           each [1, 16, 16, 16].
            lr:            EMA learning rate (step size).
        """
        device = self.home_vector.device

        with torch.no_grad():
            neutral_condition = torch.zeros(1, COND_DIM, device=device)
            x   = vae.state_encoder(*chunk_tensors)
            mu, _ = vae.encoder(x, neutral_condition)

        encoded = mu.squeeze(0).to(device)
        self.home_vector = self.home_vector + lr * (encoded - self.home_vector)

    # ------------------------------------------------------------------
    # World event methods
    # ------------------------------------------------------------------

    def cultural_exchange(
        self,
        other: FactionIdentity,
        influence: float = 0.05,
    ) -> None:
        """
        Bidirectional style drift: each faction nudges toward the other.

        The net effect on `self` and `other` is symmetric — both move closer
        to each other by `influence` fraction of the distance between them.
        """
        delta = other.home_vector - self.home_vector
        self.home_vector  = self.home_vector  + influence * delta
        other.home_vector = other.home_vector - influence * delta

    def conquest(
        self,
        conqueror: FactionIdentity,
        assimilation: float = 0.3,
    ) -> None:
        """
        `self` is the conquered faction — partially adopt the conqueror's style.

        Args:
            conqueror:     The victorious faction.
            assimilation:  Fraction (0–1) of conqueror style to absorb.
        """
        delta = conqueror.home_vector - self.home_vector
        self.home_vector = self.home_vector + assimilation * delta

    def spawn_splinter(self, drift: float = 0.5) -> FactionIdentity:
        """
        Create a child faction with a randomly drifted home vector.

        The child inherits a slightly higher variance to reflect its unsettled
        cultural identity.

        Args:
            drift: Standard deviation of the random drift applied.

        Returns:
            A new FactionIdentity instance.
        """
        child = FactionIdentity(
            latent_dim=self.home_vector.shape[0],
            variance=self.variance * 1.2,
        )
        child.home_vector = self.home_vector + torch.randn_like(self.home_vector) * drift
        return child

    @classmethod
    def merge(
        cls,
        a: FactionIdentity,
        b: FactionIdentity,
    ) -> FactionIdentity:
        """
        Combine two factions into a new merged faction.

        The merged faction's home vector is the average of `a` and `b`, and its
        variance is the average of their variances.

        Args:
            a: First faction.
            b: Second faction.

        Returns:
            A new FactionIdentity representing the merged culture.
        """
        latent_dim = a.home_vector.shape[0]
        merged = cls(latent_dim=latent_dim, variance=(a.variance + b.variance) / 2)
        merged.home_vector = (a.home_vector + b.home_vector) / 2
        return merged

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "faction_id":  self.faction_id,
            "home_vector": self.home_vector.tolist(),
            "variance":    self.variance,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FactionIdentity:
        """Deserialize from a dict produced by to_dict()."""
        f = cls()
        f.faction_id  = d["faction_id"]
        f.home_vector = torch.tensor(d["home_vector"], dtype=torch.float32)
        f.variance    = d["variance"]
        return f

    def __repr__(self) -> str:
        return (
            f"FactionIdentity(id={self.faction_id!r}, "
            f"variance={self.variance:.3f}, "
            f"norm={self.home_vector.norm().item():.3f})"
        )


# ---------------------------------------------------------------------------
# FactionStore
# ---------------------------------------------------------------------------

class FactionStore:
    """
    Manages a collection of FactionIdentity objects, persisted as JSON.

    The backing file is a JSON object mapping faction_id → serialized dict.
    The store creates new factions on demand when an unknown ID is requested.

    Usage::

        store = FactionStore.load("factions.json")
        faction = store.get("some-uuid")
        faction.cultural_exchange(store.get("other-uuid"))
        store.save("factions.json")
    """

    def __init__(self) -> None:
        self._factions: dict[str, FactionIdentity] = {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write all faction identities to a JSON file."""
        data = {fid: f.to_dict() for fid, f in self._factions.items()}
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> FactionStore:
        """
        Load faction identities from a JSON file.

        If the file does not exist, returns an empty store.
        """
        store = cls()
        p = Path(path)
        if p.exists():
            data: dict[str, Any] = json.loads(p.read_text())
            for fid, d in data.items():
                store._factions[fid] = FactionIdentity.from_dict(d)
        return store

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get(self, faction_id: str) -> FactionIdentity:
        """
        Return the FactionIdentity for the given ID, creating it if needed.
        """
        if faction_id not in self._factions:
            f = FactionIdentity(faction_id=faction_id)
            self._factions[faction_id] = f
        return self._factions[faction_id]

    def all_ids(self) -> list[str]:
        """Return all known faction IDs."""
        return list(self._factions.keys())

    def __len__(self) -> int:
        return len(self._factions)

    # ------------------------------------------------------------------
    # World event dispatcher
    # ------------------------------------------------------------------

    def apply_event(self, event_type: str, **kwargs: Any) -> dict[str, Any]:
        """
        Dispatch a world event and return any new faction IDs created.

        Supported event_type values:
            "cultural_exchange" — kwargs: faction_a, faction_b, influence
            "conquest"          — kwargs: conqueror, conquered, assimilation
            "schism"            — kwargs: parent, child_id, drift
            "alliance"          — kwargs: faction_a, faction_b, merged_id

        Returns:
            A dict containing the IDs of affected/created factions and their
            updated home vectors (as lists) for downstream persistence.
        """
        if event_type == "cultural_exchange":
            a = self.get(kwargs["faction_a"])
            b = self.get(kwargs["faction_b"])
            a.cultural_exchange(b, influence=kwargs.get("influence", 0.05))
            return {
                "updated": [
                    {"faction_id": a.faction_id, "home_vector": a.home_vector.tolist()},
                    {"faction_id": b.faction_id, "home_vector": b.home_vector.tolist()},
                ]
            }

        elif event_type == "conquest":
            conqueror = self.get(kwargs["conqueror"])
            conquered = self.get(kwargs["conquered"])
            conquered.conquest(conqueror, assimilation=kwargs.get("assimilation", 0.3))
            return {
                "updated": [
                    {"faction_id": conquered.faction_id, "home_vector": conquered.home_vector.tolist()},
                ]
            }

        elif event_type == "schism":
            parent = self.get(kwargs["parent"])
            child  = parent.spawn_splinter(drift=kwargs.get("drift", 0.5))
            if "child_id" in kwargs:
                child.faction_id = kwargs["child_id"]
            self._factions[child.faction_id] = child
            return {
                "created": {"faction_id": child.faction_id, "home_vector": child.home_vector.tolist()},
            }

        elif event_type == "alliance":
            a = self.get(kwargs["faction_a"])
            b = self.get(kwargs["faction_b"])
            merged = FactionIdentity.merge(a, b)
            if "merged_id" in kwargs:
                merged.faction_id = kwargs["merged_id"]
            self._factions[merged.faction_id] = merged
            return {
                "created": {"faction_id": merged.faction_id, "home_vector": merged.home_vector.tolist()},
            }

        else:
            raise ValueError(f"Unknown event_type: {event_type!r}")
