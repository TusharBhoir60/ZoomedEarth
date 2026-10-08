import json
import math
from pathlib import Path

import numpy as np
import pytest
import rasterio

from src.eval.wald import (
    compute_psnr,
    compute_metrics,
    run_wald_experiment
)

def test_compute_psnr():
    # PSNR = 10 * log10(MAX^2 / MSE)
    # If MSE is 0.01 and MAX is 1.0, PSNR = 10 * log10(1 / 0.01) = 20
    assert math.isclose(compute_psnr(0.01), 20.0, rel_tol=1e-5)
    
    # Perfect reconstruction
    assert compute_psnr(0.0) == float('inf')

def test_compute_metrics():
    # Synthetic data
    hr = np.ones((4, 4, 4), dtype=np.float32) * 0.5
    
    # 1. Perfect reconstruction
    metrics = compute_metrics(hr, hr)
    assert np.isclose(metrics["rmse_global"], 0.0)
    assert metrics["psnr_global"] == float('inf')
    assert np.isclose(metrics["sam_rad"], 0.0)
    
    # 2. Known perturbation (+0.1)
    sr = hr + 0.1
    metrics = compute_metrics(hr, sr)
    # MSE = 0.01. RMSE = 0.1. PSNR = 20.0
    for b in ["B02", "B03", "B04", "B08"]:
        assert np.isclose(metrics[f"rmse_{b}"], 0.1, atol=1e-6)
        assert np.isclose(metrics[f"psnr_{b}"], 20.0, atol=1e-6)
        
    assert np.isclose(metrics["rmse_global"], 0.1, atol=1e-6)
    assert np.isclose(metrics["psnr_global"], 20.0, atol=1e-6)
    
    # 3. Mask behavior
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True # Only one valid pixel
    sr_bad = sr.copy()
    sr_bad[:, 1:, 1:] = 1.0 # Bad values in masked regions
    
    metrics_masked = compute_metrics(hr, sr_bad, mask=mask)
    assert np.isclose(metrics_masked["rmse_global"], 0.1, atol=1e-6)

def test_compute_metrics_nan_filtering():
    # Synthetic data
    hr = np.ones((4, 4, 4), dtype=np.float32) * 0.5
    sr = hr + 0.1
    
    # B02 (index 0) is finite, B08 (index 3) gets a NaN at (0, 0)
    hr[3, 0, 0] = np.nan
    
    # B03 (index 1) gets an Inf at (1, 1)
    hr[1, 1, 1] = np.inf
    
    # SR gets a NaN at (2, 2) on B04
    sr[2, 2, 2] = np.nan
    
    metrics = compute_metrics(hr, sr)
    
    # Since 3 pixels are filtered out, 13 pixels should remain.
    # The error should still be exactly 0.1 (since all valid pixels have difference 0.1)
    assert "error" not in metrics
    assert np.isclose(metrics["rmse_global"], 0.1, atol=1e-6)
    for b in ["B02", "B03", "B04", "B08"]:
        assert np.isclose(metrics[f"rmse_{b}"], 0.1, atol=1e-6)


def test_run_wald_experiment_synthetic(tmp_path):
    hr_dir = tmp_path / "hr"
    hr_dir.mkdir()
    
    out_dir = tmp_path / "out"
    
    # We need a small mock scene, e.g. 128x128
    hr_h, hr_w = 128, 128
    
    tf_hr = rasterio.transform.from_origin(0, 0, 10, 10)
    profile_hr = {
        "driver": "GTiff", "height": hr_h, "width": hr_w, "count": 1,
        "dtype": "float32", "transform": tf_hr, "crs": "EPSG:32643"
    }
    
    hr_arr = np.random.rand(4, hr_h, hr_w).astype(np.float32)
    for i, band in enumerate(["B02", "B03", "B04", "B08"]):
        with rasterio.open(hr_dir / f"{band}.tif", "w", **profile_hr) as dst:
            dst.write(hr_arr[i], 1)
            
    # Run the experiment
    # We override block_size to something small so we get multiple blocks (e.g., 32)
    # The device can be CPU for testing. We use the real models (bicubic and sen2sr).
    res = run_wald_experiment(hr_dir, out_dir, block_size=32, device="cpu")
    
    assert "error" not in res
    assert res["n_blocks"] == 16 # 128/32 = 4, 4*4 = 16 blocks
    assert res["n_boot"] == 1000
    assert "metrics" in res
    assert "rmse_global" in res["metrics"]
    
    # Check outputs exist
    assert (out_dir / "lr_40m" / "multiband_40m.tif").exists()
    assert (out_dir / "sen2sr_10m" / "SEN2SR.tif").exists()
    assert (out_dir / "bicubic_10m.tif").exists()
    
    # Check resolution relationship
    with rasterio.open(out_dir / "lr_40m" / "multiband_40m.tif") as src:
        assert src.width == hr_w // 4
        assert src.height == hr_h // 4
        assert abs(src.transform.a) == 40.0 # 10m * 4
        
    with rasterio.open(out_dir / "sen2sr_10m" / "SEN2SR.tif") as src:
        assert src.width == hr_w
        assert src.height == hr_h
        assert abs(src.transform.a) == 10.0 # 40m / 4
        
    # Check metrics structure
    global_rmse = res["metrics"]["rmse_global"]
    assert "bicubic" in global_rmse
    assert "sen2sr" in global_rmse
    assert "diff" in global_rmse
    assert "mean" in global_rmse["diff"]
    assert "ci_lower" in global_rmse["diff"]
    assert "ci_upper" in global_rmse["diff"]
    assert "significant" in global_rmse["diff"]
