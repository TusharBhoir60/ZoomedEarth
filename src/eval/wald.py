import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
import torch

from src.eval.consistency import downsample_box_4x4, compute_sam
from src.infer.bicubic import run_bicubic
from src.infer.infer_tile import infer_tile

INPUT_BANDS = ["B02", "B03", "B04", "B08"]

def compute_psnr(mse: float, max_val: float = 1.0) -> float:
    if mse == 0:
        return float('inf')
    return 10 * math.log10((max_val ** 2) / mse)

def compute_metrics(hr: np.ndarray, sr: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute metrics between HR (C, H, W) and SR (C, H, W).
    """
    if hr.shape != sr.shape:
        raise ValueError(f"Shape mismatch: HR {hr.shape} vs SR {sr.shape}")
        
    metrics = {}
    if mask is None:
        mask = np.ones(hr.shape[1:], dtype=bool)
        
    valid = mask & ~np.isnan(hr[0]) & ~np.isnan(sr[0])
    
    if not np.any(valid):
        return {"error": "No valid pixels"}
        
    # Prevent uint16 overflow and ensure reflectance space [0, 1]
    hr_f = hr.astype(np.float64)
    sr_f = sr.astype(np.float64)
    
    if np.max(hr_f) > 2.0:
        hr_f /= 10000.0
    if np.max(sr_f) > 2.0:
        sr_f /= 10000.0
        
    global_sq_err = 0.0
    for i, band in enumerate(INPUT_BANDS):
        hr_b = hr_f[i][valid]
        sr_b = sr_f[i][valid]
        
        mse = float(np.mean((hr_b - sr_b)**2))
        metrics[f"rmse_{band}"] = math.sqrt(mse)
        metrics[f"psnr_{band}"] = compute_psnr(mse)
        global_sq_err += np.sum((hr_b - sr_b)**2)
        
    num_valid = np.sum(valid)
    mse_global = global_sq_err / (num_valid * len(INPUT_BANDS))
    metrics["rmse_global"] = math.sqrt(mse_global)
    metrics["psnr_global"] = compute_psnr(mse_global)
    
    # sam computation expects [0, 1] as well, compute_sam handles normalisation internally? Let's check compute_sam
    metrics["sam_rad"] = compute_sam(hr_f, sr_f, mask=valid)
    
    return metrics

def run_wald_experiment(
    hr_scene_dir: Path,
    output_dir: Path,
    block_size: int = 512,
    device: str = "cuda:0"
) -> Dict[str, Any]:
    """
    Runs the Wald protocol:
    1. Reads HR (10m) scene.
    2. Downsamples to 40m LR.
    3. Super-resolves 40m LR to 10m SR using SEN2SR and Bicubic.
    4. Computes metrics against original HR 10m.
    """
    hr_scene_dir = Path(hr_scene_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Prepare 40m LR inputs
    lr_40m_dir = output_dir / "lr_40m"
    lr_40m_dir.mkdir(exist_ok=True)
    
    # Read first HR band to get profile
    b02_path = hr_scene_dir / "B02.tif"
    with rasterio.open(b02_path) as src:
        hr_profile = src.profile.copy()
        hr_transform = src.transform
        hr_h = src.height
        hr_w = src.width
        
    # We must ensure dimensions are divisible by 4
    if hr_h % 4 != 0 or hr_w % 4 != 0:
        hr_h = (hr_h // 4) * 4
        hr_w = (hr_w // 4) * 4
        
    lr_h = hr_h // 4
    lr_w = hr_w // 4
    
    lr_transform = rasterio.transform.Affine(
        hr_transform.a * 4, hr_transform.b, hr_transform.c,
        hr_transform.d, hr_transform.e * 4, hr_transform.f
    )
    
    # Load all HR bands and downsample
    hr_arrays = []
    for b in INPUT_BANDS:
        with rasterio.open(hr_scene_dir / f"{b}.tif") as src:
            arr = src.read(1, window=Window(0, 0, hr_w, hr_h))
            hr_arrays.append(arr)
            
    hr_stacked = np.stack(hr_arrays, axis=0) # (4, hr_h, hr_w)
    lr_stacked = downsample_box_4x4(hr_stacked) # (4, lr_h, lr_w)
    
    # Write 40m multi-band for bicubic
    multiband_40m_path = lr_40m_dir / "multiband_40m.tif"
    multiband_profile = hr_profile.copy()
    multiband_profile.update({
        "height": lr_h,
        "width": lr_w,
        "count": 4,
        "transform": lr_transform,
        "tiled": True
    })
    
    with rasterio.open(multiband_40m_path, "w", **multiband_profile) as dst:
        dst.write(lr_stacked)
        for i, b in enumerate(INPUT_BANDS):
            dst.set_band_description(i + 1, b)
            
    # Write 40m tile dir for SEN2SR
    sen2sr_40m_tile_dir = lr_40m_dir / "sen2sr_tile"
    sen2sr_40m_tile_dir.mkdir(exist_ok=True)
    single_band_profile = multiband_profile.copy()
    single_band_profile["count"] = 1
    
    for i, b in enumerate(INPUT_BANDS):
        with rasterio.open(sen2sr_40m_tile_dir / f"{b}.tif", "w", **single_band_profile) as dst:
            dst.write(lr_stacked[i], 1)
            
    # Copy SCL if present, downsampled by nearest neighbor
    scl_path = hr_scene_dir / "SCL.tif"
    scl_hr = None
    if scl_path.exists():
        with rasterio.open(scl_path) as src:
            scl_hr = src.read(1, window=Window(0, 0, hr_w, hr_h))
            # simple nearest neighbor for mask
            scl_lr = scl_hr[::4, ::4]
            
        scl_profile = single_band_profile.copy()
        scl_profile["dtype"] = "uint8"
        if "nodata" in scl_profile:
            del scl_profile["nodata"]
        with rasterio.open(sen2sr_40m_tile_dir / "SCL.tif", "w", **scl_profile) as dst:
            dst.write(scl_lr, 1)
            
    # Delete stacked arrays to save memory before inference
    del hr_stacked
    del lr_stacked

    # Write metadata for sen2sr tile
    with open(sen2sr_40m_tile_dir / "metadata.json", "w") as f:
        json.dump({"item_id": "wald_proxy", "tile_id": "0"}, f)
        
    # 2. Run inference
    bicubic_out_path = output_dir / "bicubic_10m.tif"
    run_bicubic(
        input_path=multiband_40m_path,
        output_path=bicubic_out_path,
        scale=4.0,
        band_order=INPUT_BANDS
    )
    
    sen2sr_out_dir = output_dir / "sen2sr_10m"
    infer_tile(
        tile_dir=sen2sr_40m_tile_dir,
        output_dir=sen2sr_out_dir,
        device=device,
        require_10m=False
    )
    
    sen2sr_out_path = sen2sr_out_dir / "SEN2SR.tif"
    
    # 3. Compute Metrics via block bootstrap
    print("Collecting block metrics...")
    blocks = []
    n_y = hr_h // block_size
    n_x = hr_w // block_size
    
    print(f"Total blocks to process: {n_y * n_x}")
    
    # Open dataset readers
    hr_srcs = {b: rasterio.open(hr_scene_dir / f"{b}.tif") for b in INPUT_BANDS}
    scl_src = rasterio.open(scl_path) if scl_path.exists() else None
    bic_src = rasterio.open(bicubic_out_path)
    sen_src = rasterio.open(sen2sr_out_path)
    
    for iy in range(n_y):
        for ix in range(n_x):
            y0 = iy * block_size
            x0 = ix * block_size
            window = Window(x0, y0, block_size, block_size)
            
            # Read HR block
            hr_b = np.stack([hr_srcs[b].read(1, window=window) for b in INPUT_BANDS], axis=0)
            
            # Read SR blocks
            bic_b = bic_src.read(window=window)
            sen_b = sen_src.read(window=window)
            
            # Read SCL block
            if scl_src:
                scl_b = scl_src.read(1, window=window)
                msk_b = np.isin(scl_b, [4, 5, 6, 7])
            else:
                msk_b = np.ones((block_size, block_size), dtype=bool)
            
            if np.sum(msk_b) < (block_size * block_size * 0.1): # Need at least 10% valid pixels
                continue
                
            m_bic = compute_metrics(hr_b, bic_b, msk_b)
            m_sen = compute_metrics(hr_b, sen_b, msk_b)
            
            if "error" not in m_bic and "error" not in m_sen:
                blocks.append({
                    "bicubic": m_bic,
                    "sen2sr": m_sen
                })
                
    # Close datasets
    for src in hr_srcs.values(): src.close()
    if scl_src: scl_src.close()
    bic_src.close()
    sen_src.close()
    
    print(f"Valid blocks collected: {len(blocks)}")
                
    print(f"Valid blocks collected: {len(blocks)}")
                
    # 4. Bootstrap CI
    n_boot = 1000
    rng = np.random.default_rng(42) # fixed seed
    
    if len(blocks) == 0:
        return {"error": "No valid blocks found for evaluation"}
        
    metrics_keys = list(blocks[0]["bicubic"].keys())
    
    results = {}
    
    for k in metrics_keys:
        bic_vals = np.array([b["bicubic"][k] for b in blocks])
        sen_vals = np.array([b["sen2sr"][k] for b in blocks])
        diff_vals = sen_vals - bic_vals
        
        # Resampling blocks with replacement
        boot_idx = rng.choice(len(blocks), size=(n_boot, len(blocks)), replace=True)
        boot_bic = bic_vals[boot_idx].mean(axis=1)
        boot_sen = sen_vals[boot_idx].mean(axis=1)
        boot_diff = diff_vals[boot_idx].mean(axis=1)
        
        results[k] = {
            "bicubic": {
                "mean": float(np.mean(bic_vals)),
                "ci_lower": float(np.percentile(boot_bic, 2.5)),
                "ci_upper": float(np.percentile(boot_bic, 97.5))
            },
            "sen2sr": {
                "mean": float(np.mean(sen_vals)),
                "ci_lower": float(np.percentile(boot_sen, 2.5)),
                "ci_upper": float(np.percentile(boot_sen, 97.5))
            },
            "diff": {
                "mean": float(np.mean(diff_vals)),
                "ci_lower": float(np.percentile(boot_diff, 2.5)),
                "ci_upper": float(np.percentile(boot_diff, 97.5)),
                "significant": not (np.percentile(boot_diff, 2.5) <= 0 <= np.percentile(boot_diff, 97.5))
            }
        }
        
    return {
        "dataset": str(hr_scene_dir),
        "n_blocks": len(blocks),
        "block_size": block_size,
        "n_boot": n_boot,
        "seed": 42,
        "metrics": results,
        "blocked_metrics": ["SSIM", "LPIPS"]
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    
    res = run_wald_experiment(args.hr_dir, args.out_dir, device=args.device)
    
    with open(Path(args.out_dir) / "wald_results.json", "w") as f:
        json.dump(res, f, indent=2)
        
    print(json.dumps(res, indent=2))
