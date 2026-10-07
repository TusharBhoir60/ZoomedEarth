"""
tests/test_band_order_contract.py

RAD-2 band-order contract tests.

Tests:
  1. Order-sensitive stub model verifies the exact permutation contract
  2. infer_tile band-order end-to-end (assert_band_order helper)
  3. stitch band-order end-to-end
  4. bicubic band-order
  5. Three mutation tests: forward removed, reverse removed, both removed — all must fail
  6. stitch refuses legacy tiles lacking contract fields
  7. Real-model diagonal RMSE test (skipped if no GPU or weights)
"""
import json
import pytest
import numpy as np
import rasterio
import torch
from pathlib import Path
from rasterio.transform import from_origin, Affine
from unittest.mock import patch

from src.infer.infer_tile import infer_tile, INPUT_BANDS
from src.infer.run_sen2sr import permute_bgrn_to_rgbn, permute_rgbn_to_bgrn
from src.infer.stitch import stitch_scene
from src.infer.bicubic import run_bicubic

# ── Sentinel band values ──────────────────────────────────────────────────────
# These are per-band constants; any permutation error is detectable because
# no two bands share a value.
BAND_SENTINELS = {"B02": 0.1, "B03": 0.2, "B04": 0.3, "B08": 0.4}
# In model RGBN order: [B04=0.3, B03=0.2, B02=0.1, B08=0.4]
MODEL_ORDER_SENTINELS = [0.3, 0.2, 0.1, 0.4]

TILE_H, TILE_W = 32, 32
SCALE = 4


# ── Order-sensitive stub model ────────────────────────────────────────────────

class OrderSensitiveStub(torch.nn.Module):
    """
    Stub model that:
    1. Asserts the input arrives in the correct model order [B04, B03, B02, B08].
    2. Returns the input upsampled 4× (nearest-neighbour) so pixel constants are exact.

    If both permutations are deleted the input would still be [B02,B03,B04,B08]
    and channel 0 would be 0.1 (B02), not 0.3 (B04) → assertion fails.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (1, 4, H, W) — already permuted by infer_tile
        assert x.ndim == 4 and x.shape[1] == 4, f"Expected (1,4,H,W), got {x.shape}"
        tol = 1e-4

        ch0_val = float(x[0, 0].mean())
        ch2_val = float(x[0, 2].mean())

        assert abs(ch0_val - MODEL_ORDER_SENTINELS[0]) < tol, (
            f"Stub: channel 0 expected B04={MODEL_ORDER_SENTINELS[0]}, got {ch0_val:.4f}. "
            "Forward permutation (bgrn→rgbn) may be missing or wrong."
        )
        assert abs(ch2_val - MODEL_ORDER_SENTINELS[2]) < tol, (
            f"Stub: channel 2 expected B02={MODEL_ORDER_SENTINELS[2]}, got {ch2_val:.4f}. "
            "Forward permutation (bgrn→rgbn) may be missing or wrong."
        )

        # Nearest-neighbour 4× upsample — constants preserved exactly
        return x.repeat_interleave(SCALE, dim=2).repeat_interleave(SCALE, dim=3)


# ── Shared fixture helpers ────────────────────────────────────────────────────

def make_tile(root: Path, tile_id: str = "tile_r000_c000",
              item_id: str = "test_item") -> Path:
    """Build a minimal synthetic T2.5 tile directory with sentinel band values."""
    tile_dir = root / tile_id
    tile_dir.mkdir(parents=True)
    transform = from_origin(500000.0, 3500000.0, 10.0, 10.0)
    f32_profile = dict(driver="GTiff", height=TILE_H, width=TILE_W, count=1,
                       dtype=rasterio.float32, crs="EPSG:32643", transform=transform)
    u8_profile = {**f32_profile, "dtype": rasterio.uint8}

    for band, val in BAND_SENTINELS.items():
        data = np.full((TILE_H, TILE_W), val, dtype=np.float32)
        with rasterio.open(tile_dir / f"{band}.tif", "w", **f32_profile) as dst:
            dst.write(data, 1)

    with rasterio.open(tile_dir / "SCL.tif", "w", **u8_profile) as dst:
        dst.write(np.full((TILE_H, TILE_W), 4, dtype=np.uint8), 1)
    with rasterio.open(tile_dir / "cloud_mask.tif", "w", **u8_profile) as dst:
        dst.write(np.zeros((TILE_H, TILE_W), dtype=np.uint8), 1)

    meta = {
        "item_id": item_id, "tile_id": tile_id, "row": 0, "column": 0,
        "window": [0, 0, TILE_W, TILE_H], "width": TILE_W, "height": TILE_H,
        "crs": "EPSG:32643",
        "transform": [transform.a, transform.b, transform.c,
                      transform.d, transform.e, transform.f],
        "resolution": "10m", "band_order": ["B02", "B03", "B04", "B08"],
        "tile_size": TILE_W, "overlap": 0, "scene_pixel_count": TILE_W * TILE_H,
        "cloud_fraction": 0.0, "valid_fraction": 1.0, "tiling_version": "v1.0",
    }
    with open(tile_dir / "metadata.json", "w") as f:
        json.dump(meta, f)
    return tile_dir


def make_sr_tile(sr_dir: Path, tile_id: str, r_px: int, c_px: int,
                 sr_transform: Affine) -> Path:
    """Create a synthetic already-inferred SR tile in the expected layout."""
    t_dir = sr_dir / tile_id
    t_dir.mkdir(parents=True)
    out_h = TILE_H * SCALE
    out_w = TILE_W * SCALE

    # Band-sentinel values, same as input (stub model is identity under permutation)
    data = np.stack([
        np.full((out_h, out_w), BAND_SENTINELS["B02"], dtype=np.float32),
        np.full((out_h, out_w), BAND_SENTINELS["B03"], dtype=np.float32),
        np.full((out_h, out_w), BAND_SENTINELS["B04"], dtype=np.float32),
        np.full((out_h, out_w), BAND_SENTINELS["B08"], dtype=np.float32),
    ])  # (4, H, W) in canonical output order

    tile_tf = sr_transform * Affine.translation(c_px, r_px)
    profile = {"driver": "GTiff", "height": out_h, "width": out_w, "count": 4,
               "dtype": "float32", "crs": "EPSG:32643", "transform": tile_tf,
               "nodata": float("nan")}
    with rasterio.open(t_dir / "SEN2SR.tif", "w", **profile) as dst:
        dst.write(data)

    tile_meta = {
        "output_transform": [tile_tf.a, tile_tf.b, tile_tf.c,
                             tile_tf.d, tile_tf.e, tile_tf.f],
        "output_width": out_w,
        "output_height": out_h,
        "input_band_order": ["B02", "B03", "B04", "B08"],
        "model_band_order": ["B04", "B03", "B02", "B08"],
        "output_band_order": ["B02", "B03", "B04", "B08"],
        "band_order_contract_version": "v1.0",
    }
    with open(t_dir / "metadata.json", "w") as f:
        json.dump(tile_meta, f)
    return t_dir


# ── assert_band_order helper ──────────────────────────────────────────────────

def assert_band_order(result_tif: Path):
    """
    Check output band i == input band i using sentinel values.
    Raises AssertionError with a clear message if any band is wrong.
    """
    with rasterio.open(result_tif) as src:
        data = src.read()  # (4, H, W)

    expected = [BAND_SENTINELS["B02"], BAND_SENTINELS["B03"],
                BAND_SENTINELS["B04"], BAND_SENTINELS["B08"]]
    labels = list(BAND_SENTINELS.keys())

    for i, (exp_val, band) in enumerate(zip(expected, labels)):
        actual_mean = float(np.nanmean(data[i]))
        assert abs(actual_mean - exp_val) < 1e-4, (
            f"Band {i} ({band}): expected mean {exp_val:.3f}, "
            f"got {actual_mean:.4f}. Band order contract violated."
        )


# ── 1. infer_tile band-order ──────────────────────────────────────────────────

def test_infer_tile_band_order(tmp_path):
    """Output band i must equal input band i through a full infer_tile call."""
    tile_dir = make_tile(tmp_path)
    out_dir = tmp_path / "out"
    stub = OrderSensitiveStub()
    meta = infer_tile(tile_dir=tile_dir, output_dir=out_dir, model=stub)

    assert meta["output_band_order"] == ["B02", "B03", "B04", "B08"]
    assert meta["band_order_contract_version"] == "v1.0"
    assert_band_order(out_dir / "SEN2SR.tif")


# ── 2. stitch band-order ──────────────────────────────────────────────────────

def test_stitch_band_order(tmp_path):
    """Stitched scene band i must equal input band i."""
    item_id = "test_stitch_item"
    lr_w, lr_h = TILE_W, TILE_H
    sr_w, sr_h = lr_w * SCALE, lr_h * SCALE

    # Create processed scene metadata
    proc_dir = tmp_path / "processed" / item_id
    proc_dir.mkdir(parents=True)
    lr_tf = from_origin(500000.0, 3500000.0, 10.0, 10.0)
    with open(proc_dir / "metadata.json", "w") as f:
        json.dump({
            "crs": "EPSG:32643",
            "transform": [lr_tf.a, lr_tf.b, lr_tf.c, lr_tf.d, lr_tf.e, lr_tf.f],
            "width": lr_w, "height": lr_h,
        }, f)

    sr_tf = Affine(lr_tf.a / SCALE, lr_tf.b / SCALE, lr_tf.c,
                   lr_tf.d / SCALE, lr_tf.e / SCALE, lr_tf.f)
    sr_dir = tmp_path / "outputs" / item_id
    sr_dir.mkdir(parents=True)
    make_sr_tile(sr_dir, "tile_r000_c000", r_px=0, c_px=0, sr_transform=sr_tf)

    meta = stitch_scene(item_id,
                        processed_dir=str(tmp_path / "processed"),
                        sr_output_dir=str(tmp_path / "outputs"))

    assert meta["output_band_order"] == ["B02", "B03", "B04", "B08"]
    assert meta["band_order_contract_version"] == "v1.0"
    assert_band_order(sr_dir / "SEN2SR_scene.tif")


# ── 3. bicubic band-order ─────────────────────────────────────────────────────

def test_bicubic_band_order(tmp_path):
    """Bicubic output band i must equal input band i."""
    in_path = tmp_path / "in.tif"
    out_path = tmp_path / "out.tif"

    data = np.stack([
        np.full((TILE_H, TILE_W), BAND_SENTINELS["B02"], dtype=np.float32),
        np.full((TILE_H, TILE_W), BAND_SENTINELS["B03"], dtype=np.float32),
        np.full((TILE_H, TILE_W), BAND_SENTINELS["B04"], dtype=np.float32),
        np.full((TILE_H, TILE_W), BAND_SENTINELS["B08"], dtype=np.float32),
    ])
    tf = from_origin(500000.0, 3500000.0, 10.0, 10.0)
    profile = dict(driver="GTiff", height=TILE_H, width=TILE_W, count=4,
                   dtype="float32", crs="EPSG:32643", transform=tf)
    with rasterio.open(in_path, "w", **profile) as dst:
        dst.write(data)

    result = run_bicubic(in_path, out_path, band_order=["B02", "B03", "B04", "B08"])

    assert result["input_band_order"] == ["B02", "B03", "B04", "B08"]
    assert result["output_band_order"] == ["B02", "B03", "B04", "B08"]
    assert result["model_band_order"] is None
    assert result["band_order_contract_version"] == "v1.0"
    assert_band_order(out_path)


# ── 4. Mutation tests ─────────────────────────────────────────────────────────

def test_mutation_forward_permutation_removed(tmp_path):
    """If the forward (bgrn→rgbn) permutation is removed, stub raises."""
    tile_dir = make_tile(tmp_path)
    stub = OrderSensitiveStub()

    import src.infer.infer_tile as infer_tile_mod
    identity = lambda x: x

    with patch.object(infer_tile_mod, "permute_bgrn_to_rgbn", identity):
        with pytest.raises((AssertionError, ValueError)):
            infer_tile(tile_dir=tile_dir, output_dir=tmp_path / "out", model=stub)


def test_mutation_reverse_permutation_removed(tmp_path):
    """If the reverse (rgbn→bgrn) permutation is removed, assert_band_order fails."""
    tile_dir = make_tile(tmp_path)
    stub = OrderSensitiveStub()

    import src.infer.infer_tile as infer_tile_mod
    identity = lambda x: x

    with patch.object(infer_tile_mod, "permute_rgbn_to_bgrn", identity):
        # The stub passes (input to model is still correctly permuted),
        # but the output is now in model order [B04,B03,B02,B08] not canonical.
        infer_tile(tile_dir=tile_dir, output_dir=tmp_path / "out", model=stub)

    with pytest.raises(AssertionError):
        assert_band_order(tmp_path / "out" / "SEN2SR.tif")


def test_mutation_both_permutations_removed(tmp_path):
    """If BOTH permutations are removed, stub fails (input not in model order)."""
    tile_dir = make_tile(tmp_path)
    stub = OrderSensitiveStub()

    import src.infer.infer_tile as infer_tile_mod
    identity = lambda x: x

    with patch.object(infer_tile_mod, "permute_bgrn_to_rgbn", identity):
        with patch.object(infer_tile_mod, "permute_rgbn_to_bgrn", identity):
            with pytest.raises((AssertionError, ValueError)):
                infer_tile(tile_dir=tile_dir, output_dir=tmp_path / "out", model=stub)


# ── 5. stitch rejects legacy tiles ───────────────────────────────────────────

def test_stitch_rejects_legacy_tile_missing_contract(tmp_path):
    """stitch_scene must refuse tiles that lack output_band_order."""
    item_id = "legacy_item"
    lr_w, lr_h = TILE_W, TILE_H
    lr_tf = from_origin(500000.0, 3500000.0, 10.0, 10.0)

    proc_dir = tmp_path / "processed" / item_id
    proc_dir.mkdir(parents=True)
    with open(proc_dir / "metadata.json", "w") as f:
        json.dump({
            "crs": "EPSG:32643",
            "transform": [lr_tf.a, lr_tf.b, lr_tf.c, lr_tf.d, lr_tf.e, lr_tf.f],
            "width": lr_w, "height": lr_h,
        }, f)

    sr_tf = Affine(lr_tf.a / SCALE, lr_tf.b / SCALE, lr_tf.c,
                   lr_tf.d / SCALE, lr_tf.e / SCALE, lr_tf.f)
    sr_dir = tmp_path / "outputs" / item_id
    sr_dir.mkdir(parents=True)

    # Create a tile WITHOUT the new contract fields
    t_dir = sr_dir / "tile_r000_c000"
    t_dir.mkdir()
    out_h, out_w = lr_h * SCALE, lr_w * SCALE
    data = np.ones((4, out_h, out_w), dtype=np.float32)
    tile_tf = sr_tf
    profile = {"driver": "GTiff", "height": out_h, "width": out_w, "count": 4,
               "dtype": "float32", "crs": "EPSG:32643", "transform": tile_tf,
               "nodata": float("nan")}
    with rasterio.open(t_dir / "SEN2SR.tif", "w", **profile) as dst:
        dst.write(data)
    # Legacy metadata: no output_band_order, no band_order_contract_version
    with open(t_dir / "metadata.json", "w") as f:
        json.dump({
            "output_transform": [tile_tf.a, tile_tf.b, tile_tf.c,
                                 tile_tf.d, tile_tf.e, tile_tf.f],
            "output_width": out_w,
            "output_height": out_h,
        }, f)

    with pytest.raises(ValueError, match="missing input_band_order"):
        stitch_scene(item_id,
                     processed_dir=str(tmp_path / "processed"),
                     sr_output_dir=str(tmp_path / "outputs"))


# ── 6. Real-model diagonal RMSE test ─────────────────────────────────────────

WEIGHTS_DIR = Path("models/SEN2SRLite")
EXAMPLE_DATA = WEIGHTS_DIR / "example_data.safetensor"


@pytest.mark.skipif(
    not (EXAMPLE_DATA.exists() and
         (WEIGHTS_DIR / "sr_model.safetensor").exists()),
    reason="Requires SEN2SRLite weights + example_data.safetensor"
)
def test_real_model_equivalence(tmp_path):
    """
    Equivalence test: run models/SEN2SRLite example data (model order) directly
    through the model; run the same data permuted to canonical order through infer_tile;
    permute the pipeline output back to model order; assert allclose (FP32).
    """
    import safetensors.torch
    from src.infer.run_sen2sr import load_sen2sr_lite_model

    ex = safetensors.torch.load_file(str(EXAMPLE_DATA))
    # Shape (1, 10, 128, 128): first 4 are RGBN LR bands
    raw_model_in = ex["lr"][:, :4]  # (1, 4, 128, 128) in model order [B04, B03, B02, B08]
    
    # Clean input: replace NaNs with 0.0 so both paths get identical numeric input
    raw_model_in = torch.nan_to_num(raw_model_in, nan=0.0)

    # Power check: ensure the test can discriminate channels
    ch0_mean = float(raw_model_in[0, 0].mean())
    ch2_mean = float(raw_model_in[0, 2].mean())
    assert abs(ch0_mean - ch2_mean) > 1e-3, "Example data channels 0 and 2 are too similar to test permutation"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    model = load_sen2sr_lite_model(weights_dir=WEIGHTS_DIR, device=device)

    # 1. Direct model path
    with torch.no_grad():
        raw_model_out = model(raw_model_in.to(device)).cpu()  # (1, 4, 512, 512)

    # 2. Pipeline path
    from src.infer.run_sen2sr import permute_rgbn_to_bgrn, permute_bgrn_to_rgbn
    canonical_in = permute_rgbn_to_bgrn(raw_model_in.numpy()[0])  # (4, 128, 128)

    tile_dir = tmp_path / "real_tile"
    tile_dir.mkdir()
    tf = from_origin(500000.0, 3500000.0, 10.0, 10.0)
    f32_profile = dict(driver="GTiff", height=128, width=128, count=1,
                       dtype=rasterio.float32, crs="EPSG:32643", transform=tf)
    u8_profile = {**f32_profile, "dtype": rasterio.uint8}

    for i, band in enumerate(["B02", "B03", "B04", "B08"]):
        with rasterio.open(tile_dir / f"{band}.tif", "w", **f32_profile) as dst:
            dst.write(canonical_in[i], 1)
    with rasterio.open(tile_dir / "SCL.tif", "w", **u8_profile) as dst:
        dst.write(np.full((128, 128), 4, dtype=np.uint8), 1)
    with rasterio.open(tile_dir / "cloud_mask.tif", "w", **u8_profile) as dst:
        dst.write(np.zeros((128, 128), dtype=np.uint8), 1)

    meta_json = {
        "item_id": "real_test", "tile_id": "tile_r000_c000",
        "row": 0, "column": 0, "window": [0, 0, 128, 128],
        "width": 128, "height": 128, "crs": "EPSG:32643",
        "transform": [tf.a, tf.b, tf.c, tf.d, tf.e, tf.f],
        "resolution": "10m", "band_order": ["B02", "B03", "B04", "B08"],
        "tile_size": 128, "overlap": 0, "scene_pixel_count": 128 * 128,
        "cloud_fraction": 0.0, "valid_fraction": 1.0, "tiling_version": "v1.0",
    }
    with open(tile_dir / "metadata.json", "w") as f:
        json.dump(meta_json, f)

    out_dir = tmp_path / "sr_out"
    infer_tile(tile_dir=tile_dir, output_dir=out_dir, model=model, device=str(device))

    with rasterio.open(out_dir / "SEN2SR.tif") as src:
        out_canonical = src.read()  # (4, 512, 512)

    out_model_order = permute_bgrn_to_rgbn(out_canonical)

    # Note: 1e-4 tolerance due to possible floating point non-determinism on GPU,
    # though they should be very close.
    np.testing.assert_allclose(out_model_order, raw_model_out.squeeze(0).numpy(), rtol=1e-4, atol=1e-4)

    # Negative control: assert swapped permutation does not match
    wrong_perm = out_model_order[[2, 1, 0, 3], :, :]
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(wrong_perm, raw_model_out.squeeze(0).numpy(), rtol=1e-4, atol=1e-4)
