"""
Comprehensive tests for T2.6 — tile-level SEN2SR inference.

All tests are offline and deterministic. The SEN2SR model is mocked.
"""
import hashlib
import json
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin, Affine
from pathlib import Path
from unittest.mock import MagicMock, patch
import torch

from src.infer.infer_tile import (
    INPUT_BANDS,
    MODEL_BAND_ORDER,
    SCALE,
    validate_input_tile,
    validate_output,
    infer_tile,
)
from src.infer.run_sen2sr import permute_bgrn_to_rgbn, scale_affine_transform

# ─── Helpers ─────────────────────────────────────────────────────────────────

TILE_W, TILE_H = 32, 32   # small synthetic tile

def make_tile(
    root: Path,
    tile_id: str = "tile_r000_c000",
    item_id: str = "test_item",
    width: int = TILE_W,
    height: int = TILE_H,
    transform: Affine = None,
    crs: str = "EPSG:32643",
    band_vals: dict = None,
) -> Path:
    """Build a minimal synthetic T2.5 tile directory."""
    tile_dir = root / tile_id
    tile_dir.mkdir(parents=True)

    if transform is None:
        transform = from_origin(500000.0, 3500000.0, 10.0, 10.0)

    if band_vals is None:
        # Distinct value per band so permutation errors are detectable
        band_vals = {"B02": 0.1, "B03": 0.2, "B04": 0.3, "B08": 0.4}

    f32_profile = dict(driver="GTiff", height=height, width=width, count=1,
                       dtype=rasterio.float32, crs=crs, transform=transform)
    u8_profile  = f32_profile.copy()
    u8_profile["dtype"] = rasterio.uint8

    for band, val in band_vals.items():
        data = np.full((height, width), val, dtype=np.float32)
        with rasterio.open(tile_dir / f"{band}.tif", "w", **f32_profile) as dst:
            dst.write(data, 1)

    scl = np.full((height, width), 4, dtype=np.uint8)
    with rasterio.open(tile_dir / "SCL.tif", "w", **u8_profile) as dst:
        dst.write(scl, 1)

    mask = np.zeros((height, width), dtype=np.uint8)
    with rasterio.open(tile_dir / "cloud_mask.tif", "w", **u8_profile) as dst:
        dst.write(mask, 1)

    meta = {
        "item_id": item_id,
        "tile_id": tile_id,
        "row": 0,
        "column": 0,
        "window": [0, 0, width, height],
        "width": width,
        "height": height,
        "crs": crs,
        "transform": [transform.a, transform.b, transform.c,
                      transform.d, transform.e, transform.f],
        "resolution": "10m",
        "band_order": ["B02", "B03", "B04", "B08"],
        "tile_size": width,
        "overlap": 0,
        "scene_pixel_count": width * height,
        "cloud_fraction": 0.0,
        "valid_fraction": 1.0,
        "tiling_version": "v1.0",
    }
    with open(tile_dir / "metadata.json", "w") as f:
        json.dump(meta, f)

    return tile_dir


def make_mock_model(output_val: float = 0.5, nan: bool = False, inf_val: bool = False):
    """Return a callable mock that returns a deterministic 4x output tensor."""
    def _forward(tensor):
        # tensor: (1, 4, H, W) -> return (4, H*4, W*4)
        b, c, h, w = tensor.shape
        out_h = h * 4
        out_w = w * 4
        if nan:
            data = torch.full((c, out_h, out_w), float("nan"))
        elif inf_val:
            data = torch.full((c, out_h, out_w), float("inf"))
        else:
            data = torch.full((c, out_h, out_w), output_val)
        return data.unsqueeze(0)   # (1, 4, H*4, W*4)

    mock = MagicMock()
    mock.side_effect = _forward
    return mock


# ─── 1. Band-order permutation ────────────────────────────────────────────────

def test_band_order_permutation_detectable():
    """
    Distinct band values ensure a permutation error is detectable.
    [B02=0.1, B03=0.2, B04=0.3, B08=0.4] -> [B04=0.3, B03=0.2, B02=0.1, B08=0.4]
    """
    canonical = np.array([
        np.full((4, 4), 0.1, dtype=np.float32),  # B02
        np.full((4, 4), 0.2, dtype=np.float32),  # B03
        np.full((4, 4), 0.3, dtype=np.float32),  # B04
        np.full((4, 4), 0.4, dtype=np.float32),  # B08
    ])
    permuted = permute_bgrn_to_rgbn(canonical)
    np.testing.assert_allclose(permuted[0], 0.3)  # B04 -> index 0
    np.testing.assert_allclose(permuted[1], 0.2)  # B03 -> index 1
    np.testing.assert_allclose(permuted[2], 0.1)  # B02 -> index 2
    np.testing.assert_allclose(permuted[3], 0.4)  # B08 -> index 3


def test_band_order_metadata_recorded(tmp_path):
    tile_dir = make_tile(tmp_path)
    mock_model = make_mock_model()
    meta = infer_tile(tile_dir=tile_dir, output_dir=tmp_path / "out", model=mock_model)
    assert meta["input_band_order"] == ["B02", "B03", "B04", "B08"]
    assert meta["model_band_order"] == ["B04", "B03", "B02", "B08"]


# ─── 2. Input validation ──────────────────────────────────────────────────────

@pytest.mark.parametrize("missing_band", ["B02", "B03", "B04", "B08"])
def test_missing_band_raises(tmp_path, missing_band):
    tile_dir = make_tile(tmp_path)
    (tile_dir / f"{missing_band}.tif").unlink()
    with pytest.raises(FileNotFoundError, match=missing_band):
        validate_input_tile(tile_dir)


def test_mismatched_dimensions_raises(tmp_path):
    tile_dir = make_tile(tmp_path)
    # Overwrite B03 with wrong dimensions
    transform = from_origin(500000.0, 3500000.0, 10.0, 10.0)
    profile = dict(driver="GTiff", height=TILE_H + 1, width=TILE_W, count=1,
                   dtype=rasterio.float32, crs="EPSG:32643", transform=transform)
    with rasterio.open(tile_dir / "B03.tif", "w", **profile) as dst:
        dst.write(np.zeros((TILE_H + 1, TILE_W), dtype=np.float32), 1)
    with pytest.raises(ValueError, match="dimensions"):
        validate_input_tile(tile_dir)


def test_mismatched_crs_raises(tmp_path):
    tile_dir = make_tile(tmp_path)
    transform = from_origin(500000.0, 3500000.0, 10.0, 10.0)
    profile = dict(driver="GTiff", height=TILE_H, width=TILE_W, count=1,
                   dtype=rasterio.float32, crs="EPSG:4326", transform=transform)
    with rasterio.open(tile_dir / "B04.tif", "w", **profile) as dst:
        dst.write(np.zeros((TILE_H, TILE_W), dtype=np.float32), 1)
    with pytest.raises(ValueError, match="CRS"):
        validate_input_tile(tile_dir)


def test_mismatched_transform_raises(tmp_path):
    tile_dir = make_tile(tmp_path)
    wrong_transform = from_origin(600000.0, 3500000.0, 10.0, 10.0)
    profile = dict(driver="GTiff", height=TILE_H, width=TILE_W, count=1,
                   dtype=rasterio.float32, crs="EPSG:32643", transform=wrong_transform)
    with rasterio.open(tile_dir / "B08.tif", "w", **profile) as dst:
        dst.write(np.zeros((TILE_H, TILE_W), dtype=np.float32), 1)
    with pytest.raises(ValueError, match="transform"):
        validate_input_tile(tile_dir)


def test_invalid_resolution_raises(tmp_path):
    bad_transform = from_origin(500000.0, 3500000.0, 20.0, 20.0)  # 20m not 10m
    tile_dir = make_tile(tmp_path, transform=bad_transform)
    with pytest.raises(ValueError, match="pixel size"):
        validate_input_tile(tile_dir)


def test_inf_in_input_raises(tmp_path):
    tile_dir = make_tile(tmp_path)
    # Inject Inf into B02
    with rasterio.open(tile_dir / "B02.tif", "r+") as dst:
        data = dst.read(1)
        data[0, 0] = float("inf")
        dst.write(data, 1)
    with pytest.raises(ValueError, match="Inf"):
        validate_input_tile(tile_dir)


# ─── 3. 4× output dimensions ─────────────────────────────────────────────────

def test_output_dimensions_4x(tmp_path):
    tile_dir = make_tile(tmp_path)
    mock_model = make_mock_model()
    meta = infer_tile(tile_dir=tile_dir, output_dir=tmp_path / "out", model=mock_model)
    assert meta["output_width"] == TILE_W * 4
    assert meta["output_height"] == TILE_H * 4

    with rasterio.open(tmp_path / "out" / "SEN2SR.tif") as src:
        assert src.width == TILE_W * 4
        assert src.height == TILE_H * 4


# ─── 4. Affine transform ─────────────────────────────────────────────────────

def test_affine_transform_scaled_correctly(tmp_path):
    # Non-trivial transform with rotation/shear terms set to 0 but non-identity origin
    in_transform = Affine(10.0, 0.3, 712000.0, -0.3, -10.0, 2800000.0)
    tile_dir = make_tile(tmp_path, transform=in_transform)
    mock_model = make_mock_model()
    meta = infer_tile(tile_dir=tile_dir, output_dir=tmp_path / "out", model=mock_model)

    out_tf_list = meta["output_transform"]
    a_out, b_out, c_out, d_out, e_out, f_out = out_tf_list

    assert pytest.approx(a_out) == in_transform.a / 4
    assert pytest.approx(b_out) == in_transform.b / 4
    assert pytest.approx(c_out) == in_transform.c      # origin preserved
    assert pytest.approx(d_out) == in_transform.d / 4
    assert pytest.approx(e_out) == in_transform.e / 4
    assert pytest.approx(f_out) == in_transform.f      # origin preserved


def test_output_raster_transform_correct(tmp_path):
    in_transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 3500000.0)
    tile_dir = make_tile(tmp_path, transform=in_transform)
    mock_model = make_mock_model()
    infer_tile(tile_dir=tile_dir, output_dir=tmp_path / "out", model=mock_model)

    expected = scale_affine_transform(in_transform, scale=4.0)
    with rasterio.open(tmp_path / "out" / "SEN2SR.tif") as src:
        assert src.transform == expected


# ─── 5. CRS preservation ─────────────────────────────────────────────────────

def test_crs_preserved(tmp_path):
    tile_dir = make_tile(tmp_path, crs="EPSG:32644")
    mock_model = make_mock_model()
    infer_tile(tile_dir=tile_dir, output_dir=tmp_path / "out", model=mock_model)

    with rasterio.open(tmp_path / "out" / "SEN2SR.tif") as src:
        assert src.crs.to_string() == "EPSG:32644"


# ─── 6. Output footprint ─────────────────────────────────────────────────────

def test_output_footprint_matches_input(tmp_path):
    in_transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 3500000.0)
    tile_dir = make_tile(tmp_path, width=TILE_W, height=TILE_H, transform=in_transform)
    mock_model = make_mock_model()
    infer_tile(tile_dir=tile_dir, output_dir=tmp_path / "out", model=mock_model)

    in_bounds = rasterio.transform.array_bounds(TILE_H, TILE_W, in_transform)
    out_transform = scale_affine_transform(in_transform, scale=4.0)
    out_bounds = rasterio.transform.array_bounds(TILE_H * 4, TILE_W * 4, out_transform)

    assert pytest.approx(in_bounds) == out_bounds


# ─── 7. dtype and band count ─────────────────────────────────────────────────

def test_output_dtype_and_band_count(tmp_path):
    tile_dir = make_tile(tmp_path)
    mock_model = make_mock_model()
    infer_tile(tile_dir=tile_dir, output_dir=tmp_path / "out", model=mock_model)

    with rasterio.open(tmp_path / "out" / "SEN2SR.tif") as src:
        assert src.count == 4
        assert src.dtypes[0] == "float32"


# ─── 8. Invalid model output ─────────────────────────────────────────────────

def test_nan_model_output_raises(tmp_path):
    tile_dir = make_tile(tmp_path)
    mock_model = make_mock_model(nan=True)
    with pytest.raises(ValueError, match="NaN"):
        infer_tile(tile_dir=tile_dir, output_dir=tmp_path / "out", model=mock_model)


def test_inf_model_output_raises(tmp_path):
    tile_dir = make_tile(tmp_path)
    mock_model = make_mock_model(inf_val=True)
    with pytest.raises(ValueError, match="Inf"):
        infer_tile(tile_dir=tile_dir, output_dir=tmp_path / "out", model=mock_model)


# ─── 9. Determinism ──────────────────────────────────────────────────────────

def test_determinism(tmp_path):
    tile_dir = make_tile(tmp_path)

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    mock_model = make_mock_model(output_val=0.42)
    meta1 = infer_tile(tile_dir=tile_dir, output_dir=out1, model=mock_model)

    mock_model2 = make_mock_model(output_val=0.42)
    meta2 = infer_tile(tile_dir=tile_dir, output_dir=out2, model=mock_model2)

    # Dimensions must be identical
    assert meta1["output_width"] == meta2["output_width"]
    assert meta1["output_height"] == meta2["output_height"]
    assert meta1["output_transform"] == meta2["output_transform"]
    assert meta1["input_band_order"] == meta2["input_band_order"]
    assert meta1["model_band_order"] == meta2["model_band_order"]

    # Pixel data must be identical
    with rasterio.open(out1 / "SEN2SR.tif") as s1, rasterio.open(out2 / "SEN2SR.tif") as s2:
        np.testing.assert_array_equal(s1.read(), s2.read())


# ─── 10. Source protection ────────────────────────────────────────────────────

def test_source_tile_unchanged(tmp_path):
    tile_dir = make_tile(tmp_path)

    src_files = list(tile_dir.glob("*"))
    def chksum(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    before = {f: chksum(f) for f in src_files}

    mock_model = make_mock_model()
    infer_tile(tile_dir=tile_dir, output_dir=tmp_path / "out", model=mock_model)

    for f in src_files:
        assert chksum(f) == before[f], f"{f.name} was modified by infer_tile"


# ─── 11. Masking / NaN handling ───────────────────────────────────────────────

def test_nan_pixels_handled_not_propagated(tmp_path):
    """NaN-masked pixels in the input are substituted with 0.0 before model call."""
    tile_dir = make_tile(tmp_path)
    # Manually write NaN into B02 to simulate masked cloud pixels
    with rasterio.open(tile_dir / "B02.tif", "r+") as dst:
        data = dst.read(1)
        data[0, 0] = float("nan")
        dst.write(data, 1)

    # validate_input_tile should not raise; NaN is substituted with 0.0 for model
    validated = validate_input_tile(tile_dir)
    assert not np.any(np.isnan(validated["stacked_model_in"]))
    # But the original canonical array still preserves NaN
    assert np.isnan(validated["stacked_canonical"][0, 0, 0])


# ─── 12. Output validation helper ────────────────────────────────────────────

def test_validate_output_wrong_shape():
    arr = np.zeros((4, TILE_H * 4 + 1, TILE_W * 4), dtype=np.float32)
    profile = {"height": TILE_H, "width": TILE_W}
    with pytest.raises(ValueError, match="shape"):
        validate_output(arr, profile)


def test_validate_output_nan():
    arr = np.zeros((4, TILE_H * 4, TILE_W * 4), dtype=np.float32)
    arr[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        validate_output(arr, {"height": TILE_H, "width": TILE_W})


# ─── 13. Metadata completeness ───────────────────────────────────────────────

REQUIRED_META_KEYS = [
    "item_id", "tile_id", "source_tile", "model", "model_version_or_identifier",
    "input_band_order", "model_band_order",
    "input_width", "input_height", "output_width", "output_height",
    "input_resolution", "output_resolution",
    "crs", "input_transform", "output_transform",
    "dtype", "precision",
    "inference_timestamp", "inference_duration_seconds",
    "git_sha", "software_versions", "hardware",
]

def test_metadata_completeness(tmp_path):
    tile_dir = make_tile(tmp_path)
    mock_model = make_mock_model()
    meta = infer_tile(tile_dir=tile_dir, output_dir=tmp_path / "out", model=mock_model)
    for key in REQUIRED_META_KEYS:
        assert key in meta, f"Required metadata key missing: {key}"

    with open(tmp_path / "out" / "metadata.json") as f:
        saved_meta = json.load(f)
    for key in REQUIRED_META_KEYS:
        assert key in saved_meta, f"Saved metadata.json missing key: {key}"
