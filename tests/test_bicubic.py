"""
Tests for Bicubic Baseline (src/infer/bicubic.py).

Verifies:
    1. Dimension scaling: (H, W) → (H×4, W×4)
    2. Band preservation: no channel permutation, distinct per-band values detectable
    3. CRS preservation: output CRS == input CRS
    4. Transform correctness: origin, extent, pixel-size ÷4; rotated/sheared case
    5. Determinism: two identical runs produce bitwise-identical output
    6. Input validation: missing file, wrong band count, missing CRS
    7. Raster readability: write → reopen → verify all geospatial and data properties
"""

import math
import pathlib
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from src.infer.bicubic import run_bicubic


# ── Shared synthetic raster fixtures ──────────────────────────────────────────

def _write_4band_raster(
    path: pathlib.Path,
    height: int = 32,
    width: int = 32,
    dtype="uint16",
    crs: CRS = CRS.from_epsg(32643),
    transform: Affine = Affine(10.0, 0.0, 200000.0, 0.0, -10.0, 2100000.0),
    band_values=None,
) -> pathlib.Path:
    """Write a synthetic 4-band raster with distinct per-band values."""
    if band_values is None:
        # Default: distinct per-band values in uint16 (L2A BOA × 10000 scale)
        band_values = [1000, 1500, 2000, 3000]  # B02, B03, B04, B08

    data = np.zeros((4, height, width), dtype=dtype)
    for i, v in enumerate(band_values):
        data[i, :, :] = v

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 4,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


# ── Test 1: Dimension scaling ──────────────────────────────────────────────────

class TestDimensionScaling:
    """32×32 input must produce exactly 128×128 output."""

    def test_32x32_to_128x128(self, tmp_path):
        inp = _write_4band_raster(tmp_path / "in_32.tif", height=32, width=32)
        out = tmp_path / "out_128.tif"
        result = run_bicubic(inp, out)

        assert result["input_shape"] == [4, 32, 32]
        assert result["output_shape"] == [4, 128, 128]
        assert result["scale"] == 4.0

        with rasterio.open(out) as dst:
            assert dst.height == 128
            assert dst.width == 128
            assert dst.count == 4

    def test_non_square_input(self, tmp_path):
        """Non-square inputs must scale both dimensions independently."""
        inp = _write_4band_raster(tmp_path / "in_rect.tif", height=20, width=40)
        out = tmp_path / "out_rect.tif"
        result = run_bicubic(inp, out)

        assert result["output_shape"] == [4, 80, 160]
        with rasterio.open(out) as dst:
            assert dst.height == 80
            assert dst.width == 160


# ── Test 2: Band preservation ──────────────────────────────────────────────────

class TestBandPreservation:
    """
    Uses band-distinct values so accidental swaps are detectable.
    B02=0.10, B03=0.15, B04=0.20, B08=0.30 (after ÷10000 normalisation).
    """

    def test_no_channel_permutation(self, tmp_path):
        # Band values as uint16 (BOA × 10000)
        inp = _write_4band_raster(
            tmp_path / "in_bands.tif",
            band_values=[1000, 1500, 2000, 3000],  # B02, B03, B04, B08
        )
        out = tmp_path / "out_bands.tif"
        run_bicubic(inp, out)

        with rasterio.open(out) as dst:
            data = dst.read()
            # After normalisation ÷10000, expected reflectance ≈ 0.10, 0.15, 0.20, 0.30
            # Bicubic on a constant-valued band is identity → values preserved.
            assert data.shape[0] == 4
            np.testing.assert_allclose(data[0].mean(), 0.10, atol=1e-3)  # B02
            np.testing.assert_allclose(data[1].mean(), 0.15, atol=1e-3)  # B03
            np.testing.assert_allclose(data[2].mean(), 0.20, atol=1e-3)  # B04
            np.testing.assert_allclose(data[3].mean(), 0.30, atol=1e-3)  # B08

            # Verify band descriptions carry correct labels
            assert "B02" in dst.descriptions[0]
            assert "B03" in dst.descriptions[1]
            assert "B04" in dst.descriptions[2]
            assert "B08" in dst.descriptions[3]

    def test_band_ordering_distinct_values_not_swapped(self, tmp_path):
        """If channels were swapped, at least one mean would be wrong."""
        inp = _write_4band_raster(
            tmp_path / "in_swap.tif",
            band_values=[100, 200, 400, 800],  # 2× differences between each
        )
        out = tmp_path / "out_swap.tif"
        run_bicubic(inp, out)

        with rasterio.open(out) as dst:
            data = dst.read()
            means = [data[i].mean() for i in range(4)]
            # Ratios must be preserved: 100:200:400:800 = 1:2:4:8
            assert means[0] < means[1] < means[2] < means[3]
            np.testing.assert_allclose(means[1] / means[0], 2.0, rtol=1e-3)
            np.testing.assert_allclose(means[2] / means[0], 4.0, rtol=1e-3)
            np.testing.assert_allclose(means[3] / means[0], 8.0, rtol=1e-3)


# ── Test 3: CRS preservation ───────────────────────────────────────────────────

class TestCRSPreservation:
    """Output CRS must equal input CRS."""

    def test_epsg_32643(self, tmp_path):
        crs = CRS.from_epsg(32643)
        inp = _write_4band_raster(tmp_path / "in_crs.tif", crs=crs)
        out = tmp_path / "out_crs.tif"
        run_bicubic(inp, out)

        with rasterio.open(out) as dst:
            assert dst.crs == crs

    def test_epsg_4326(self, tmp_path):
        crs = CRS.from_epsg(4326)
        transform = Affine(0.0001, 0.0, 77.0, 0.0, -0.0001, 28.0)
        inp = _write_4band_raster(tmp_path / "in_4326.tif", crs=crs, transform=transform)
        out = tmp_path / "out_4326.tif"
        run_bicubic(inp, out)

        with rasterio.open(out) as dst:
            assert dst.crs == crs


# ── Test 4: Transform correctness ─────────────────────────────────────────────

class TestTransformCorrectness:
    """Origin, pixel size ÷4, spatial extent, and rotated transform."""

    def test_north_up_transform(self, tmp_path):
        in_tf = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0)
        inp = _write_4band_raster(
            tmp_path / "in_northup.tif",
            height=32, width=32,
            transform=in_tf,
        )
        out = tmp_path / "out_northup.tif"
        run_bicubic(inp, out)

        with rasterio.open(out) as dst:
            tf = dst.transform

            # Origin preserved
            assert tf.c == in_tf.c
            assert tf.f == in_tf.f

            # Pixel size ÷4
            assert math.isclose(tf.a, 2.5, rel_tol=1e-9)
            assert math.isclose(tf.e, -2.5, rel_tol=1e-9)

            # Shear terms should remain 0
            assert tf.b == 0.0
            assert tf.d == 0.0

            # Spatial extent preserved: 32 px × 10m = 128 px × 2.5m = 320m
            in_corner = in_tf * (32, 32)
            out_corner = tf * (128, 128)
            assert math.isclose(in_corner[0], out_corner[0], abs_tol=1e-6)
            assert math.isclose(in_corner[1], out_corner[1], abs_tol=1e-6)

    def test_rotated_and_sheared_transform(self, tmp_path):
        """General affine (b≠0, d≠0): full linear component must scale."""
        in_tf = Affine(8.0, 6.0, 12345.0, -6.0, 8.0, 67890.0)
        inp = _write_4band_raster(
            tmp_path / "in_rotated.tif",
            height=32, width=32,
            transform=in_tf,
        )
        out = tmp_path / "out_rotated.tif"
        run_bicubic(inp, out)

        with rasterio.open(out) as dst:
            tf = dst.transform

            # Origin preserved
            assert tf.c == in_tf.c
            assert tf.f == in_tf.f

            # Linear components scaled by 1/4
            assert math.isclose(tf.a, 2.0,  rel_tol=1e-9)
            assert math.isclose(tf.b, 1.5,  rel_tol=1e-9)
            assert math.isclose(tf.d, -1.5, rel_tol=1e-9)
            assert math.isclose(tf.e, 2.0,  rel_tol=1e-9)

            # Spatial extent preserved for an arbitrary corner
            in_corner  = in_tf * (32, 32)
            out_corner = tf    * (128, 128)
            assert math.isclose(in_corner[0], out_corner[0], abs_tol=1e-6)
            assert math.isclose(in_corner[1], out_corner[1], abs_tol=1e-6)


# ── Test 5: Determinism ────────────────────────────────────────────────────────

class TestDeterminism:
    """Two identical runs must produce bitwise-identical output."""

    def test_two_runs_identical(self, tmp_path):
        inp = _write_4band_raster(
            tmp_path / "in_det.tif",
            band_values=[1234, 2345, 3456, 4567],
        )
        out_a = tmp_path / "out_a.tif"
        out_b = tmp_path / "out_b.tif"

        run_bicubic(inp, out_a)
        run_bicubic(inp, out_b)

        with rasterio.open(out_a) as da, rasterio.open(out_b) as db:
            np.testing.assert_array_equal(da.read(), db.read())


# ── Test 6: Input validation ───────────────────────────────────────────────────

class TestInputValidation:
    """Clear actionable errors before downstream exceptions."""

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Input raster not found"):
            run_bicubic(tmp_path / "nonexistent.tif", tmp_path / "out.tif")

    def test_wrong_band_count_raises_value_error(self, tmp_path):
        bad = tmp_path / "three_band.tif"
        profile = {
            "driver": "GTiff",
            "height": 16,
            "width": 16,
            "count": 3,
            "dtype": "float32",
            "crs": CRS.from_epsg(32643),
            "transform": Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0),
        }
        with rasterio.open(bad, "w", **profile) as dst:
            dst.write(np.ones((3, 16, 16), dtype=np.float32))

        with pytest.raises(ValueError, match="exactly 4 bands"):
            run_bicubic(bad, tmp_path / "out.tif")

    def test_missing_crs_raises_value_error(self, tmp_path):
        no_crs = tmp_path / "no_crs.tif"
        profile = {
            "driver": "GTiff",
            "height": 16,
            "width": 16,
            "count": 4,
            "dtype": "float32",
        }
        with rasterio.open(no_crs, "w", **profile) as dst:
            dst.write(np.ones((4, 16, 16), dtype=np.float32))

        with pytest.raises(ValueError, match="no CRS"):
            run_bicubic(no_crs, tmp_path / "out.tif")


# ── Test 7: Raster readability ─────────────────────────────────────────────────

class TestRasterReadability:
    """Write the bicubic output and reopen it — verify all raster properties."""

    def test_full_raster_properties(self, tmp_path):
        in_tf = Affine(10.0, 0.0, 300000.0, 0.0, -10.0, 3000000.0)
        inp = _write_4band_raster(
            tmp_path / "in_full.tif",
            height=32,
            width=32,
            dtype="uint16",
            crs=CRS.from_epsg(32643),
            transform=in_tf,
            band_values=[1000, 1500, 2000, 3000],
        )
        out = tmp_path / "out_full.tif"
        result = run_bicubic(inp, out)

        assert out.exists(), "Output file was not created"

        with rasterio.open(out) as dst:
            # Dimensions
            assert dst.height == 128
            assert dst.width == 128
            assert dst.count == 4

            # CRS
            assert dst.crs == CRS.from_epsg(32643)

            # Transform: origin unchanged, pixel ÷4
            assert dst.transform.c == in_tf.c
            assert dst.transform.f == in_tf.f
            assert math.isclose(dst.transform.a, 2.5, rel_tol=1e-9)
            assert math.isclose(dst.transform.e, -2.5, rel_tol=1e-9)

            # Dtype
            assert dst.dtypes[0] == "float32"

            # Pixel data readable and in physical reflectance range
            data = dst.read()
            assert data.shape == (4, 128, 128)
            assert not np.isnan(data).any(), "Output contains NaN"
            assert not np.isinf(data).any(), "Output contains Inf"
            assert data.min() >= 0.0
            assert data.max() <= 2.0

        # Telemetry dict structure
        assert result["resampling"] == "cubic"
        assert result["output_dtype"] == "float32"
        assert result["latency_ms"] > 0
