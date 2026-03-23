"""
Smoke tests for the BetterNPC ML pipeline.

Runs without a test framework — each section prints PASS or raises on failure.
Execute from the repo root:

    cd /home/user/minecraftGeneration
    python -m ml.tests.smoke_test
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

B  = 2          # batch size used throughout
C  = 16         # chunk spatial dimension
LD = 128        # LATENT_DIM
CD = 209        # COND_DIM


def section(name: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")


def ok(label: str) -> None:
    print(f"  {PASS}  {label}")


def _rand_chunk_tensors(batch: int = B) -> dict[str, torch.Tensor]:
    """Random [B,16,16,16] int16 tensors for all block state components."""
    from ml.model.block_state import NUM_AXIS, NUM_BLOCK_TYPES, NUM_FACING, NUM_HALF, NUM_SHAPE
    return {
        "block_ids": torch.randint(0, NUM_BLOCK_TYPES, (batch, C, C, C), dtype=torch.int16),
        "facing":    torch.randint(0, NUM_FACING,      (batch, C, C, C), dtype=torch.int16),
        "axis":      torch.randint(0, NUM_AXIS,        (batch, C, C, C), dtype=torch.int16),
        "half":      torch.randint(0, NUM_HALF,        (batch, C, C, C), dtype=torch.int16),
        "shape":     torch.randint(0, NUM_SHAPE,       (batch, C, C, C), dtype=torch.int16),
    }


# ---------------------------------------------------------------------------
# 1. BlockStateEncoder shapes
# ---------------------------------------------------------------------------

def test_block_state_encoder() -> None:
    section("1. BlockStateEncoder — output shape [B, 33, 16, 16, 16]")
    from ml.model.block_state import BlockStateEncoder, VOXEL_DIM

    enc = BlockStateEncoder()
    t   = _rand_chunk_tensors()

    out = enc(
        t["block_ids"].long(),
        t["facing"].long(),
        t["axis"].long(),
        t["half"].long(),
        t["shape"].long(),
    )
    assert out.shape == (B, VOXEL_DIM, C, C, C), f"Expected {(B, VOXEL_DIM, C, C, C)}, got {out.shape}"
    assert out.dtype == torch.float32
    ok(f"output shape {tuple(out.shape)}")


# ---------------------------------------------------------------------------
# 2. Encoder shapes
# ---------------------------------------------------------------------------

def test_encoder() -> None:
    section("2. Encoder — mu/logvar shape [B, 128]")
    from ml.model.encoder import Encoder

    enc  = Encoder()
    x    = torch.randn(B, 33, C, C, C)
    c    = torch.zeros(B, CD)
    mu, logvar = enc(x, c)

    assert mu.shape     == (B, LD), f"mu: {mu.shape}"
    assert logvar.shape == (B, LD), f"logvar: {logvar.shape}"
    ok(f"mu {tuple(mu.shape)}, logvar {tuple(logvar.shape)}")


# ---------------------------------------------------------------------------
# 3. Decoder shapes
# ---------------------------------------------------------------------------

def test_decoder() -> None:
    section("3. Decoder — all 5 output heads correct shapes")
    from ml.model.decoder import Decoder
    from ml.model.block_state import NUM_BLOCK_TYPES, NUM_FACING, NUM_AXIS, NUM_HALF, NUM_SHAPE

    dec = Decoder()
    z_c = torch.randn(B, LD + CD)
    out = dec(z_c)

    expected = {
        "block_id": (B, NUM_BLOCK_TYPES, C, C, C),
        "facing":   (B, NUM_FACING,      C, C, C),
        "axis":     (B, NUM_AXIS,        C, C, C),
        "half":     (B, NUM_HALF,        C, C, C),
        "shape":    (B, NUM_SHAPE,       C, C, C),
    }
    for key, shape in expected.items():
        assert key in out,               f"Missing key: {key}"
        assert out[key].shape == shape,  f"{key}: expected {shape}, got {tuple(out[key].shape)}"
        ok(f"{key:10s} {tuple(out[key].shape)}")


# ---------------------------------------------------------------------------
# 4. VAE full forward pass + reparameterize behaviour
# ---------------------------------------------------------------------------

def test_vae_forward() -> None:
    section("4. ConditionalChunkVAE — forward pass, reparameterize guard")
    from ml.model.vae import ConditionalChunkVAE

    model = ConditionalChunkVAE()
    t     = _rand_chunk_tensors()
    cond  = torch.zeros(B, CD)

    # --- training mode: z ≠ mu (stochastic)
    model.train()
    outputs, mu, logvar = model(
        t["block_ids"].long(), t["facing"].long(),
        t["axis"].long(),      t["half"].long(),
        t["shape"].long(),     cond,
    )
    z_train = model.reparameterize(mu, logvar)
    assert not torch.allclose(z_train, mu), "reparameterize should add noise in training mode"
    ok("reparameterize adds noise during train()")

    # --- eval mode: z == mu (deterministic)
    model.eval()
    with torch.no_grad():
        outputs_eval, mu_eval, logvar_eval = model(
            t["block_ids"].long(), t["facing"].long(),
            t["axis"].long(),      t["half"].long(),
            t["shape"].long(),     cond,
        )
    z_eval = model.reparameterize(mu_eval, logvar_eval)
    assert torch.allclose(z_eval, mu_eval), "reparameterize should return mu in eval mode"
    ok("reparameterize returns mu during eval()")

    # shapes
    assert mu_eval.shape == (B, LD)
    from ml.model.block_state import NUM_BLOCK_TYPES
    assert outputs_eval["block_id"].shape == (B, NUM_BLOCK_TYPES, C, C, C)
    ok(f"forward outputs correct shapes")


# ---------------------------------------------------------------------------
# 5. vae_loss — backward pass, finite loss
# ---------------------------------------------------------------------------

def test_vae_loss() -> None:
    section("5. vae_loss — gradients flow, loss is finite")
    from ml.model.vae import ConditionalChunkVAE, vae_loss

    model = ConditionalChunkVAE()
    model.train()
    t    = _rand_chunk_tensors()
    cond = torch.zeros(B, CD)

    outputs, mu, logvar = model(
        t["block_ids"].long(), t["facing"].long(),
        t["axis"].long(),      t["half"].long(),
        t["shape"].long(),     cond,
    )
    targets = {
        "block_id": t["block_ids"].long(),
        "facing":   t["facing"].long(),
        "axis":     t["axis"].long(),
        "half":     t["half"].long(),
        "shape":    t["shape"].long(),
    }
    total, recon, kl = vae_loss(outputs, targets, mu, logvar, kl_weight=0.5)

    assert torch.isfinite(total), f"total loss is not finite: {total}"
    assert torch.isfinite(recon), f"recon loss is not finite: {recon}"
    assert torch.isfinite(kl),    f"kl loss is not finite: {kl}"

    total.backward()
    # At least one gradient should be non-None
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No gradients computed"

    ok(f"total={total.item():.4f}  recon={recon.item():.4f}  kl={kl.item():.4f}")
    ok("backward() succeeded, gradients present")


# ---------------------------------------------------------------------------
# 6. ConditionBuilder
# ---------------------------------------------------------------------------

def test_condition_builder() -> None:
    section("6. ConditionBuilder — output shape [B, 209]")
    from ml.model.vae import ConditionBuilder

    cb   = ConditionBuilder()
    cond = cb.build_condition(
        tile_type      = torch.zeros(B, dtype=torch.long),
        purpose        = torch.zeros(B, dtype=torch.long),
        face_n         = torch.zeros(B, dtype=torch.long),
        face_s         = torch.zeros(B, dtype=torch.long),
        face_e         = torch.zeros(B, dtype=torch.long),
        face_w         = torch.zeros(B, dtype=torch.long),
        face_u         = torch.zeros(B, dtype=torch.long),
        face_d         = torch.zeros(B, dtype=torch.long),
        size           = torch.full((B,), 0.5),
        variance       = torch.full((B,), 0.3),
        faction_vector = torch.randn(B, LD),
    )
    assert cond.shape == (B, CD), f"Expected ({B},{CD}), got {cond.shape}"
    ok(f"condition shape {tuple(cond.shape)}")


# ---------------------------------------------------------------------------
# 7. VAE generate()
# ---------------------------------------------------------------------------

def test_vae_generate() -> None:
    section("7. ConditionalChunkVAE.generate() — argmax decoded tensors")
    from ml.model.vae import ConditionalChunkVAE, ConditionBuilder

    model = ConditionalChunkVAE()
    cb    = ConditionBuilder()

    cond = cb.build_condition(
        tile_type      = torch.zeros(1, dtype=torch.long),
        purpose        = torch.zeros(1, dtype=torch.long),
        face_n=torch.zeros(1,dtype=torch.long), face_s=torch.zeros(1,dtype=torch.long),
        face_e=torch.zeros(1,dtype=torch.long), face_w=torch.zeros(1,dtype=torch.long),
        face_u=torch.zeros(1,dtype=torch.long), face_d=torch.zeros(1,dtype=torch.long),
        size           = torch.tensor([0.5]),
        variance       = torch.tensor([0.3]),
        faction_vector = torch.randn(1, LD),
    )
    faction_z = torch.randn(1, LD)
    decoded   = model.generate(cond, faction_z)

    for key in ("block_id", "facing", "axis", "half", "shape"):
        assert key in decoded,                        f"Missing key: {key}"
        assert decoded[key].shape == (1, C, C, C),   f"{key}: {decoded[key].shape}"
        assert decoded[key].dtype in (torch.int64, torch.long), f"{key} not integer"
        ok(f"{key:10s} {tuple(decoded[key].shape)}  dtype={decoded[key].dtype}")


# ---------------------------------------------------------------------------
# 8. augment_chunk — rotation correctness and full-cycle identity
# ---------------------------------------------------------------------------

def test_augmentation_rotation() -> None:
    section("8. augment_chunk — facing remap and 4×90° identity")
    from ml.data.augmentation import augment_chunk
    from ml.model.block_state import FACING_INDEX

    # Build a tensor where position (0,0,0) has facing=north
    north_idx = FACING_INDEX["north"]
    east_idx  = FACING_INDEX["east"]

    t = _rand_chunk_tensors(batch=1)
    # Remove batch dim for augmentation (augmentation works on [16,16,16])
    tensors = {k: v[0] for k, v in t.items()}
    tensors["facing"][0, 0, 0] = north_idx

    # After 90° CW rotation, north→east
    r90 = augment_chunk(tensors, 90)
    # rot90 with k=3 (CCW in torch terms for CW effect) moves position (0,0,0)
    # to a different spatial location; we verify the facing VALUE at any position
    # that originally held north now holds east.
    original_north_positions = (tensors["facing"] == north_idx).nonzero(as_tuple=False)
    remapped_values = r90["facing"][
        original_north_positions[:, 0],   # x coords after rotation handled separately
    ]
    # Simpler check: count of east should increase by count of north in original
    original_north_count = (tensors["facing"] == north_idx).sum().item()
    rotated_east_count   = (r90["facing"]     == east_idx ).sum().item()
    assert rotated_east_count == original_north_count, (
        f"After 90° rotation: north count {original_north_count} "
        f"should become east count {rotated_east_count}"
    )
    ok(f"90° rotation: {original_north_count} north→east values correctly remapped")

    # 4× 90° should be identity for all tensor values
    cycled = tensors.copy()
    for _ in range(4):
        cycled = augment_chunk(cycled, 90)
    for key in tensors:
        assert torch.equal(cycled[key], tensors[key]), f"4×90° not identity for {key}"
    ok("4×90° rotations cycle back to identity")

    # 0° is a no-op
    r0 = augment_chunk(tensors, 0)
    for key in tensors:
        assert torch.equal(r0[key], tensors[key]), f"0° rotation changed {key}"
    ok("0° rotation is no-op")


# ---------------------------------------------------------------------------
# 9. mirror_chunk
# ---------------------------------------------------------------------------

def test_mirror() -> None:
    section("9. mirror_chunk — X axis flip and east↔west remap")
    from ml.data.augmentation import mirror_chunk
    from ml.model.block_state import FACING_INDEX

    east_idx = FACING_INDEX["east"]
    west_idx = FACING_INDEX["west"]

    tensors = _rand_chunk_tensors(batch=1)
    tensors = {k: v[0] for k, v in tensors.items()}

    # Force east at (0,y,z) and west at (15,y,z) slices
    tensors["facing"][0,  :, :] = east_idx
    tensors["facing"][15, :, :] = west_idx

    mirrored = mirror_chunk(tensors)

    # After X flip: position (0) was originally (15), which was west → should now be east
    assert (mirrored["facing"][0,  :, :] == east_idx).all(), \
        "After mirror: X=0 slice (originally X=15, west) should be east"
    assert (mirrored["facing"][15, :, :] == west_idx).all(), \
        "After mirror: X=15 slice (originally X=0, east) should be west"
    ok("east↔west remapped after X mirror")

    # Double mirror = identity for spatial layout (block_ids)
    double = mirror_chunk(mirrored)
    assert torch.equal(double["block_ids"], tensors["block_ids"]), \
        "Double mirror should restore block_ids"
    ok("double mirror restores spatial layout")


# ---------------------------------------------------------------------------
# 10. FactionIdentity vector arithmetic
# ---------------------------------------------------------------------------

def test_faction_identity() -> None:
    section("10. FactionIdentity — vector arithmetic")
    from ml.model.faction import FactionIdentity

    a = FactionIdentity(faction_id="faction-a", variance=0.3)
    b = FactionIdentity(faction_id="faction-b", variance=0.3)
    a_vec_orig = a.home_vector.clone()
    b_vec_orig = b.home_vector.clone()

    # cultural_exchange: both should move toward each other
    a.cultural_exchange(b, influence=0.1)
    dist_after  = (a.home_vector - b.home_vector).norm().item()
    dist_before = (a_vec_orig    - b_vec_orig).norm().item()
    assert dist_after < dist_before, "cultural_exchange should reduce distance between factions"
    ok(f"cultural_exchange: distance {dist_before:.3f} → {dist_after:.3f}")

    # conquest: conquered moves toward conqueror
    conqueror = FactionIdentity(faction_id="conqueror")
    conquered = FactionIdentity(faction_id="conquered")
    c_before  = conquered.home_vector.clone()
    conquered.conquest(conqueror, assimilation=0.3)
    moved = (conquered.home_vector - c_before).norm().item()
    assert moved > 0, "conquest should move conquered home_vector"
    ok(f"conquest: home_vector moved {moved:.3f}")

    # spawn_splinter: child variance = 1.2× parent
    parent = FactionIdentity(variance=0.4)
    child  = parent.spawn_splinter(drift=0.5)
    assert abs(child.variance - parent.variance * 1.2) < 1e-6, \
        f"Expected variance {parent.variance*1.2}, got {child.variance}"
    ok(f"spawn_splinter: variance {parent.variance:.3f} → {child.variance:.3f}")

    # merge: average home_vector and variance
    m = FactionIdentity.merge(a, b)
    expected_vec = (a.home_vector + b.home_vector) / 2
    assert torch.allclose(m.home_vector, expected_vec), "merge home_vector should be average"
    ok("merge: home_vector is average of a and b")


# ---------------------------------------------------------------------------
# 11. FactionStore — save / load / apply_event
# ---------------------------------------------------------------------------

def test_faction_store() -> None:
    section("11. FactionStore — persistence and event dispatch")
    from ml.model.faction import FactionStore

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "factions.json"

        store = FactionStore()
        fa = store.get("faction-a")
        fb = store.get("faction-b")
        fa_vec = fa.home_vector.clone()
        fb_vec = fb.home_vector.clone()

        store.save(path)
        ok(f"saved {len(store)} factions to {path}")

        store2 = FactionStore.load(path)
        assert len(store2) == 2
        assert torch.allclose(store2.get("faction-a").home_vector, fa_vec), \
            "faction-a vector changed after save/load"
        assert torch.allclose(store2.get("faction-b").home_vector, fb_vec), \
            "faction-b vector changed after save/load"
        ok("save/load round-trips faction vectors exactly")

        # apply_event cultural_exchange
        result = store.apply_event("cultural_exchange", faction_a="faction-a", faction_b="faction-b", influence=0.1)
        assert "updated" in result
        assert len(result["updated"]) == 2
        ok("apply_event 'cultural_exchange' returns updated faction IDs")

        # apply_event schism
        result = store.apply_event("schism", parent="faction-a", child_id="faction-child", drift=0.3)
        assert "created" in result
        assert result["created"]["faction_id"] == "faction-child"
        assert "faction-child" in store.all_ids()
        ok("apply_event 'schism' creates child faction")

        # apply_event alliance
        result = store.apply_event("alliance", faction_a="faction-a", faction_b="faction-b", merged_id="faction-merged")
        assert result["created"]["faction_id"] == "faction-merged"
        ok("apply_event 'alliance' creates merged faction")


# ---------------------------------------------------------------------------
# 12. SyntheticChunkDataset
# ---------------------------------------------------------------------------

def test_synthetic_dataset() -> None:
    section("12. SyntheticChunkDataset — shapes and conditioning")
    from ml.data.dataset import SyntheticChunkDataset

    ds = SyntheticChunkDataset(size=10, augment=False)
    assert len(ds) == 10
    ok("len() == 10")

    tensors, cond = ds[0]
    for key in ("block_ids", "facing", "axis", "half", "shape"):
        assert key in tensors,                         f"Missing key: {key}"
        assert tensors[key].shape == (C, C, C),        f"{key}: {tensors[key].shape}"
    assert cond == {}, f"Phase-1 condition_dict should be empty, got {cond}"
    ok("__getitem__ returns 5-key tensor dict + empty condition dict")

    # With augmentation
    ds_aug = SyntheticChunkDataset(size=5, augment=True)
    t1, _ = ds_aug[0]
    ok("augmented dataset __getitem__ succeeds")


# ---------------------------------------------------------------------------
# 13. Training smoke test (2 epochs)
# ---------------------------------------------------------------------------

def test_training_smoke() -> None:
    section("13. Training smoke test — 2 epochs with synthetic data")

    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable, "-m", "ml.train",
                "--data-dir",   "/tmp/nonexistent_chunks_xyz",
                "--output-dir", tmpdir,
                "--epochs",     "2",
                "--batch-size", "4",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent),  # repo root
        )

        if result.returncode != 0:
            print(f"  STDOUT:\n{result.stdout[-2000:]}")
            print(f"  STDERR:\n{result.stderr[-2000:]}")
            raise AssertionError(f"Training exited with code {result.returncode}")

        ckpts = sorted(Path(tmpdir).glob("checkpoint_epoch_*.pt"))
        assert len(ckpts) == 2, f"Expected 2 checkpoints, found {len(ckpts)}"
        ok(f"Training completed; {len(ckpts)} checkpoints written")

        # Verify checkpoint can be loaded
        ckpt = torch.load(ckpts[-1], map_location="cpu")
        assert "model_state"  in ckpt
        assert "optim_state"  in ckpt
        assert "scaler_state" in ckpt
        assert "epoch"        in ckpt
        assert ckpt["epoch"]  == 1
        ok(f"checkpoint epoch={ckpt['epoch']} loads cleanly")

        # stdout should mention "Falling back to Synthetic"
        assert "SyntheticChunkDataset" in result.stdout, \
            "Expected fallback message in stdout"
        ok("stdout confirms synthetic data fallback")

        # Epoch log lines present
        epoch_lines = [l for l in result.stdout.splitlines() if l.startswith("Epoch")]
        assert len(epoch_lines) == 2, f"Expected 2 epoch log lines, got {len(epoch_lines)}"
        ok(f"epoch log lines: {epoch_lines[0].strip()}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    test_block_state_encoder,
    test_encoder,
    test_decoder,
    test_vae_forward,
    test_vae_loss,
    test_condition_builder,
    test_vae_generate,
    test_augmentation_rotation,
    test_mirror,
    test_faction_identity,
    test_faction_store,
    test_synthetic_dataset,
    test_training_smoke,
]


def main() -> None:
    failed: list[str] = []

    for test_fn in TESTS:
        try:
            test_fn()
        except Exception as exc:
            section_name = test_fn.__name__
            print(f"  {FAIL}  {section_name}: {exc}")
            import traceback
            traceback.print_exc()
            failed.append(section_name)

    print(f"\n{'═'*60}")
    total = len(TESTS)
    passed = total - len(failed)
    if failed:
        print(f"  {FAIL}  {passed}/{total} passed.  Failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"  {PASS}  All {total}/{total} tests passed.")


if __name__ == "__main__":
    main()
