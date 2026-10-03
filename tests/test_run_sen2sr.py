"""
Tests for SEN2SR Inference Wrapper (src/infer/run_sen2sr.py).

Verifies:
1. Band permutation [B02, B03, B04, B08] <-> [B04, B03, B02, B08]
2. General affine transform scaling (origin, extent, pixel size)
3. Input validation (band count, missing file)
4. End-to-end GeoTIFF inference, 4x spatial upscale, and georeference preservation
"""

import math
import pathlib
import tempfile
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
import torch

from src.infer.run_sen2sr import (
    permute_bgrn_to_rgbn,
    permute_rgbn_to_bgrn,
    scale_affine_transform,
    run_sen2sr,
)


class TestBandPermutation:
    """Verifies bidirectional band permutation between repository and model conventions."""

    def test_numpy_3d_permutation(self):
        # Repos order: B02=10, B03=20, B04=30, B08=40
        bgrn = np.zeros((4, 8, 8), dtype=np.float32)
        bgrn[0, :, :] = 10.0  # B02
        bgrn[1, :, :] = 20.0  # B03
        bgrn[2, :, :] = 30.0  # B04
        bgrn[3, :, :] = 40.0  # B08

        # Forward -> [B04, B03, B02, B08]
        rgbn = permute_bgrn_to_rgbn(bgrn)
        assert rgbn[0, 0, 0] == 30.0  # B04
        assert rgbn[1, 0, 0] == 20.0  # B03
        assert rgbn[2, 0, 0] == 10.0  # B02
        assert rgbn[3, 0, 0] == 40.0  # B08

        # Reverse -> [B02, B03, B04, B08]
        restored = permute_rgbn_to_bgrn(rgbn)
        np.testing.assert_array_equal(bgrn, restored)

    def test_torch_4d_permutation(self):
        bgrn = torch.zeros((2, 4, 16, 16), dtype=torch.float32)
        bgrn[:, 0, :, :] = 1.0  # B02
        bgrn[:, 1, :, :] = 2.0  # B03
        bgrn[:, 2, :, :] = 3.0  # B04
        bgrn[:, 3, :, :] = 4.0  # B08

        rgbn = permute_bgrn_to_rgbn(bgrn)
        assert (rgbn[:, 0, :, :] == 3.0).all()
        assert (rgbn[:, 1, :, :] == 2.0).all()
        assert (rgbn[:, 2, :, :] == 1.0).all()
        assert (rgbn[:, 3, :, :] == 4.0).all()

        restored = permute_rgbn_to_bgrn(rgbn)
        assert torch.equal(bgrn, restored)


class TestAffineTransformScaling:
    """Verifies general 6-parameter Affine transform scaling for 4x super-resolution."""

    def test_north_up_transform(self):
        # 10m north-up pixel
        in_transform = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 2000000.0)
        out_transform = scale_affine_transform(in_transform, scale=4.0)

        # 1. Origin unchanged
        assert (out_transform.c, out_transform.f) == (in_transform.c, in_transform.f)
        assert out_transform * (0, 0) == in_transform * (0, 0)

        # 2. Pixel size divided by 4
        assert out_transform.a == 2.5
        assert out_transform.e == -2.5

        # 3. Spatial extent preserved (100 px at 10m == 400 px at 2.5m)
        w_in, h_in = 100, 100
        w_out, h_out = 400, 400
        in_bottom_right = in_transform * (w_in, h_in)
        out_bottom_right = out_transform * (w_out, h_out)
        assert in_bottom_right == out_bottom_right

    def test_rotated_and_sheared_transform(self):
        # General non-north-up transform: b != 0, d != 0
        in_transform = Affine(8.0, 6.0, 12345.0, -6.0, 8.0, 67890.0)
        out_transform = scale_affine_transform(in_transform, scale=4.0)

        # 1. Origin preserved
        assert (out_transform.c, out_transform.f) == (in_transform.c, in_transform.f)
        assert out_transform * (0, 0) == in_transform * (0, 0)

        # 2. Linear component scaled
        assert out_transform.a == 2.0
        assert out_transform.b == 1.5
        assert out_transform.d == -1.5
        assert out_transform.e == 2.0

        # 3. Arbitrary corner point extent preserved
        pt_in = in_transform * (50, 75)
        pt_out = out_transform * (200, 300)
        assert math.isclose(pt_in[0], pt_out[0], abs_tol=1e-6)
        assert math.isclose(pt_in[1], pt_out[1], abs_tol=1e-6)


class TestInputValidation:
    """Verifies error handling for invalid input rasters."""

    def test_missing_file_raises_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_sen2sr(tmp_path / "nonexistent.tif", tmp_path / "out.tif")

    def test_invalid_band_count_raises_error(self, tmp_path):
        # Create a 3-band RGB GeoTIFF
        bad_raster = tmp_path / "three_band.tif"
        profile = {
            "driver": "GTiff",
            "height": 16,
            "width": 16,
            "count": 3,
            "dtype": "float32",
            "crs": CRS.from_epsg(32643),
            "transform": Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0),
        }
        with rasterio.open(bad_raster, "w", **profile) as dst:
            dst.write(np.ones((3, 16, 16), dtype=np.float32))

        with pytest.raises(ValueError, match="exactly 4 bands"):
            run_sen2sr(bad_raster, tmp_path / "out.tif")


class TestEndToEndInference:
    """Verifies end-to-end inference on a 4-band Sentinel-2 synthetic raster."""

    @pytest.fixture
    def sample_raster(self, tmp_path):
        raster_path = tmp_path / "sample_s2_10m.tif"
        h, w = 32, 32
        # Synthetic L2A BOA reflectance scaled by 10,000 in uint16
        data = np.zeros((4, h, w), dtype=np.uint16)
        data[0, :, :] = 1200  # B02 (0.12 reflectance)
        data[1, :, :] = 1500  # B03 (0.15 reflectance)
        data[2, :, :] = 1800  # B04 (0.18 reflectance)
        data[3, :, :] = 3500  # B08 (0.35 reflectance)

        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": 4,
            "dtype": "uint16",
            "crs": CRS.from_epsg(32643),  # UTM Zone 43N
            "transform": Affine(10.0, 0.0, 200000.0, 0.0, -10.0, 2100000.0),
        }
        with rasterio.open(raster_path, "w", **profile) as dst:
            dst.write(data)
        return raster_path

    def test_run_sen2sr_e2e(self, sample_raster, tmp_path):
        out_path = tmp_path / "output_2p5m.tif"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        result = run_sen2sr(
            input_path=sample_raster,
            output_path=out_path,
            weights_dir="models/SEN2SRLite",
            device=device,
            precision="fp32",
        )

        # 1. Output file exists
        assert out_path.exists()

        # 2. Telemetry metadata
        assert result["input_shape"] == [4, 32, 32]
        assert result["output_shape"] == [4, 128, 128]
        assert result["scale"] == 4.0
        assert result["precision"] == "fp32"
        assert result["latency_ms"] > 0

        # 3. Geospatial verification via rasterio
        with rasterio.open(out_path) as dst:
            assert dst.count == 4
            assert dst.height == 128
            assert dst.width == 128
            assert dst.crs == CRS.from_epsg(32643)

            # Transform verification
            assert dst.transform.a == 2.5
            assert dst.transform.e == -2.5
            assert dst.transform.c == 200000.0
            assert dst.transform.f == 2100000.0

            # Content verification
            out_data = dst.read()
            assert not np.isnan(out_data).any()
            assert not np.isinf(out_data).any()
            # Reflectance should be in physical range
            assert out_data.min() >= 0.0
            assert out_data.max() <= 2.0

            # Band descriptions
            assert "B02" in dst.descriptions[0]
            assert "B03" in dst.descriptions[1]
            assert "B04" in dst.descriptions[2]
            assert "B08" in dst.descriptions[3]

    def test_run_sen2sr_fp16_nan_on_this_gpu(self, sample_raster, tmp_path):
        """
        EMPIRICAL FINDING (RTX 3050 Laptop, CUDA 12.1, PyTorch 2.3.0+cu121):
        fp16 autocast causes NaN in band 4 (B08 / NIR) due to the FFT-based
        HardConstraint module using ComplexHalf arithmetic, which is experimental
        and produces NaN on this configuration.

        cuDNN warning: CUDNN_STATUS_NOT_SUPPORTED (Conv_v8.cpp:919)
        PyTorch warning: ComplexHalf support is experimental and many operators
        don't support it yet (EmptyTensor.cpp:41)

        DECISION: fp16 is UNSAFE for SEN2SRLite on this GPU.
        fp32 is the required operating mode.
        Recorded here so this is NOT silently ignored.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for FP16 test")

        out_path = tmp_path / "output_fp16.tif"
        result = run_sen2sr(
            input_path=sample_raster,
            output_path=out_path,
            weights_dir="models/SEN2SRLite",
            device="cuda:0",
            precision="fp16",
        )
        assert result["precision"] == "fp16"
        with rasterio.open(out_path) as dst:
            out_data = dst.read()
            # Band 4 (B08/NIR) is NaN under fp16 on this GPU — documented failure
            has_nan_band4 = np.isnan(out_data[3]).any()
            assert has_nan_band4, (
                "Expected NaN in B08 band under fp16 on this GPU (ComplexHalf in HardConstraint). "
                "If this passes, re-evaluate fp16 safety before enabling it."
            )
            # Bands 1-3 may be finite
            assert not np.isnan(out_data[0]).any(), "B02 has unexpected NaN in fp16"
            assert not np.isnan(out_data[1]).any(), "B03 has unexpected NaN in fp16"
            assert not np.isnan(out_data[2]).any(), "B04 has unexpected NaN in fp16"

    def test_run_sen2sr_large_image(self, tmp_path):
        # 160x160 input (exercises predict_large tiled inference)
        raster_path = tmp_path / "large_s2_160px.tif"
        h, w = 160, 160
        data = np.full((4, h, w), 2000, dtype=np.uint16)

        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": 4,
            "dtype": "uint16",
            "crs": CRS.from_epsg(32643),
            "transform": Affine(10.0, 0.0, 100000.0, 0.0, -10.0, 2000000.0),
        }
        with rasterio.open(raster_path, "w", **profile) as dst:
            dst.write(data)

        out_path = tmp_path / "output_large_640px.tif"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        result = run_sen2sr(
            input_path=raster_path,
            output_path=out_path,
            weights_dir="models/SEN2SRLite",
            device=device,
            precision="fp32",
        )

        assert out_path.exists()
        assert result["output_shape"] == [4, 640, 640]
        with rasterio.open(out_path) as dst:
            assert dst.height == 640
            assert dst.width == 640
            assert dst.transform.a == 2.5
            assert dst.transform.e == -2.5
            assert not np.isnan(dst.read()).any()

