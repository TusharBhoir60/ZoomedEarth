import numpy as np
import pytest
from pathlib import Path
import rasterio
import json

from src.eval.consistency import (
    downsample_box_4x4,
    compute_sam,
    compute_consistency_arrays,
    evaluate_scene_consistency
)

def test_downsample_box_4x4():
    # 4x4 image with ones
    sr = np.ones((1, 4, 4), dtype=np.float32)
    lr = downsample_box_4x4(sr)
    assert lr.shape == (1, 1, 1)
    assert lr[0, 0, 0] == 1.0

    # Pattern
    sr2 = np.zeros((1, 8, 8), dtype=np.float32)
    sr2[0, :4, :4] = 2.0
    sr2[0, 4:, 4:] = 4.0
    lr2 = downsample_box_4x4(sr2)
    assert lr2.shape == (1, 2, 2)
    assert lr2[0, 0, 0] == 2.0
    assert lr2[0, 1, 1] == 4.0
    assert lr2[0, 0, 1] == 0.0
    
    # Non-divisible should raise
    sr_bad = np.zeros((1, 5, 5))
    with pytest.raises(ValueError, match="not divisible by 4"):
        downsample_box_4x4(sr_bad)

def test_compute_sam():
    # Perfect match
    a = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    b = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    sam = compute_sam(a, b)
    assert np.isclose(sam, 0.0)

    # Orthogonal
    a = np.array([[[1.0]], [[0.0]]])
    b = np.array([[[0.0]], [[1.0]]])
    sam = compute_sam(a, b)
    assert np.isclose(sam, np.pi / 2)

    # Collinear (different magnitude)
    a = np.array([[[1.0]], [[1.0]]])
    b = np.array([[[2.0]], [[2.0]]])
    sam = compute_sam(a, b)
    assert np.isclose(sam, 0.0, atol=1e-7)

def test_compute_consistency_arrays():
    lr = np.ones((4, 2, 2), dtype=np.float32)
    # Perfect reconstruction
    sr = np.ones((4, 8, 8), dtype=np.float32)
    
    metrics = compute_consistency_arrays(lr, sr)
    assert "sam_rad" in metrics
    assert "rmse_global" in metrics
    assert np.isclose(metrics["sam_rad"], 0.0)
    assert np.isclose(metrics["rmse_global"], 0.0)
    for band in ["B02", "B03", "B04", "B08"]:
        assert np.isclose(metrics[f"rmse_{band}"], 0.0)
        assert np.isclose(metrics[f"rel_err_{band}"], 0.0)

    # Imperfect reconstruction
    sr2 = sr.copy()
    sr2[0, :4, :4] = 2.0 # In the first 4x4 block of B02, average becomes 2.0
    metrics2 = compute_consistency_arrays(lr, sr2)
    assert metrics2["rmse_B02"] > 0
    assert metrics2["sam_rad"] > 0

def test_evaluate_scene_consistency(tmp_path):
    lr_dir = tmp_path / "lr"
    lr_dir.mkdir()
    
    sr_path = tmp_path / "sr.tif"
    
    tf_lr = rasterio.transform.from_origin(0, 0, 10, 10)
    profile_lr = {
        "driver": "GTiff", "height": 2, "width": 2, "count": 1,
        "dtype": "float32", "transform": tf_lr, "crs": "EPSG:32643"
    }
    
    lr_arr = np.ones((4, 2, 2), dtype=np.float32)
    for i, band in enumerate(["B02", "B03", "B04", "B08"]):
        with rasterio.open(lr_dir / f"{band}.tif", "w", **profile_lr) as dst:
            dst.write(lr_arr[i], 1)
            
    # Write SCL to test masking
    scl = np.array([[4, 3], [4, 4]], dtype=np.uint8) # 4 is veg (valid), 3 is cloud shadow (invalid)
    profile_scl = profile_lr.copy()
    profile_scl["dtype"] = "uint8"
    with rasterio.open(lr_dir / "SCL.tif", "w", **profile_scl) as dst:
        dst.write(scl, 1)

    tf_sr = rasterio.transform.from_origin(0, 0, 2.5, 2.5)
    profile_sr = {
        "driver": "GTiff", "height": 8, "width": 8, "count": 4,
        "dtype": "float32", "transform": tf_sr, "crs": "EPSG:32643"
    }
    
    sr_arr = np.ones((4, 8, 8), dtype=np.float32)
    # Put a large error in the invalid pixel (row 0, col 1)
    sr_arr[:, 0:4, 4:8] = 100.0 
    
    with rasterio.open(sr_path, "w", **profile_sr) as dst:
        dst.write(sr_arr)
        
    metrics = evaluate_scene_consistency(lr_dir, sr_path)
    
    # Since the invalid pixel was masked out, metrics should still be perfect 0
    assert np.isclose(metrics["rmse_global"], 0.0)
    assert np.isclose(metrics["sam_rad"], 0.0)

def test_band_permutation():
    # Test that swapping B02 and B04 produces a measurable failure
    lr = np.ones((4, 2, 2), dtype=np.float32)
    lr[0] = 1.0 # B02
    lr[2] = 2.0 # B04
    
    sr = np.ones((4, 8, 8), dtype=np.float32)
    sr[0] = 2.0 # SR B02 is what LR B04 was (swapped)
    sr[2] = 1.0 # SR B04 is what LR B02 was
    
    metrics = compute_consistency_arrays(lr, sr)
    
    assert metrics["rmse_B02"] == 1.0
    assert metrics["rmse_B04"] == 1.0
    assert metrics["sam_rad"] > 0.0

def test_known_perturbation():
    # Deterministic perturbation: SR = LR + 0.1 uniformly
    lr = np.ones((4, 2, 2), dtype=np.float32) * 0.5
    sr = np.ones((4, 8, 8), dtype=np.float32) * 0.6
    
    metrics = compute_consistency_arrays(lr, sr)
    for b in ["B02", "B03", "B04", "B08"]:
        assert np.isclose(metrics[f"rmse_{b}"], 0.1, atol=1e-6)
        assert np.isclose(metrics[f"rel_err_{b}"], 0.2, atol=1e-6) # 0.1 / 0.5 = 0.2
    assert np.isclose(metrics["rmse_global"], 0.1, atol=1e-6)

def test_spatial_alignment_detection():
    # Test that a 1-pixel shift (in LR space) produces a measurable error
    lr = np.zeros((4, 4, 4), dtype=np.float32)
    lr[:, 2:, 2:] = 1.0
    
    # Perfect SR
    sr = np.zeros((4, 16, 16), dtype=np.float32)
    sr[:, 8:, 8:] = 1.0
    
    metrics_perfect = compute_consistency_arrays(lr, sr)
    assert np.isclose(metrics_perfect["rmse_global"], 0.0)
    
    # Shift SR by 4 pixels (1 LR pixel) down and right
    sr_shifted = np.zeros((4, 16, 16), dtype=np.float32)
    sr_shifted[:, 12:, 12:] = 1.0
    
    metrics_shifted = compute_consistency_arrays(lr, sr_shifted)
    assert metrics_shifted["rmse_global"] > 0.0
