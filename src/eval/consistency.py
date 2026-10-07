"""
T3.1 — Consistency Metrics.

Evaluates SR output against the 10m LR input by downsampling the SR back to 10m
using a fixed 4x4 box-average kernel.
Metrics computed:
- Per-band RMSE
- Spectral Angle Mapper (SAM)
- Relative Error (norm(diff) / norm(lr))
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import rasterio

# Canonical S2 bands in project order
INPUT_BANDS = ["B02", "B03", "B04", "B08"]

def downsample_box_4x4(sr_arr: np.ndarray) -> np.ndarray:
    """
    Downsample a 2.5m array to 10m using a 4x4 box average.
    Expected shape: (C, H, W) where H and W are multiples of 4.
    """
    C, H, W = sr_arr.shape
    if H % 4 != 0 or W % 4 != 0:
        raise ValueError(f"SR shape {sr_arr.shape} not divisible by 4.")
    
    # Reshape and mean over the 4x4 blocks
    return sr_arr.reshape(C, H // 4, 4, W // 4, 4).mean(axis=(2, 4))

def compute_sam(lr: np.ndarray, sr_down: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Compute Spectral Angle Mapper (SAM) in radians.
    Expects arrays of shape (C, H, W).
    """
    # Dot product along channel dimension
    dot = np.sum(lr * sr_down, axis=0)
    
    # Norms along channel dimension
    norm_lr = np.linalg.norm(lr, axis=0)
    norm_sr = np.linalg.norm(sr_down, axis=0)
    
    # Cosine of angle
    denominator = norm_lr * norm_sr
    
    # Avoid division by zero
    valid_denom = denominator > 1e-8
    
    cosine = np.zeros_like(dot)
    cosine[valid_denom] = dot[valid_denom] / denominator[valid_denom]
    
    # Clip to valid range for arccos
    cosine = np.clip(cosine, -1.0, 1.0)
    sam_map = np.arccos(cosine)
    
    if mask is not None:
        valid_mask = valid_denom & mask
        if not np.any(valid_mask):
            return np.nan
        return float(np.mean(sam_map[valid_mask]))
    else:
        if not np.any(valid_denom):
            return np.nan
        return float(np.mean(sam_map[valid_denom]))

def compute_consistency_arrays(lr: np.ndarray, sr: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute consistency metrics between LR (C, H, W) and SR (C, H*4, W*4).
    """
    if lr.shape[0] != len(INPUT_BANDS):
        raise ValueError(f"Expected {len(INPUT_BANDS)} bands, got {lr.shape[0]}")
    
    sr_down = downsample_box_4x4(sr)
    
    if sr_down.shape != lr.shape:
        raise ValueError(f"Shape mismatch after downsampling: LR {lr.shape} vs SR↓ {sr_down.shape}")
        
    metrics = {}
    
    if mask is None:
        mask = np.ones(lr.shape[1:], dtype=bool)
        
    # Mask out NaNs
    valid = mask & ~np.isnan(lr[0]) & ~np.isnan(sr_down[0])
    
    if not np.any(valid):
        return {"error": "No valid pixels"}
        
    # Per-band RMSE and Relative Error
    for i, band in enumerate(INPUT_BANDS):
        lr_b = lr[i][valid]
        sr_b = sr_down[i][valid]
        
        # RMSE
        rmse = float(np.sqrt(np.mean((lr_b - sr_b)**2)))
        metrics[f"rmse_{band}"] = rmse
        
        # Relative error (norm of diff / norm of truth)
        norm_diff = np.linalg.norm(lr_b - sr_b)
        norm_lr = np.linalg.norm(lr_b)
        rel_err = float(norm_diff / norm_lr) if norm_lr > 1e-8 else np.nan
        metrics[f"rel_err_{band}"] = rel_err
        
    # SAM
    metrics["sam_rad"] = compute_sam(lr, sr_down, mask=valid)
    
    # Global RMSE
    valid_3d = np.broadcast_to(valid, lr.shape)
    metrics["rmse_global"] = float(np.sqrt(np.mean((lr[valid_3d] - sr_down[valid_3d])**2)))
    
    return metrics

def evaluate_scene_consistency(lr_dir: Path, sr_raster_path: Path) -> Dict[str, float]:
    """
    Evaluate consistency of a full SR scene against its LR input directory by processing in blocks.
    """
    lr_dir = Path(lr_dir)
    sr_raster_path = Path(sr_raster_path)
    
    if not sr_raster_path.exists():
        raise FileNotFoundError(f"SR raster not found: {sr_raster_path}")
        
    metrics = {f"rmse_{b}": 0.0 for b in INPUT_BANDS}
    metrics["rmse_global"] = 0.0
    metrics["sam_rad"] = 0.0
    for b in INPUT_BANDS:
        metrics[f"rel_err_{b}"] = 0.0
        
    total_valid_pixels = 0
    total_sam_pixels = 0
    sum_sam = 0.0
    
    # We will accumulate squared errors for RMSE, and norms for rel_err
    sq_errs = {b: 0.0 for b in INPUT_BANDS}
    norm_diffs = {b: 0.0 for b in INPUT_BANDS}
    norm_lrs = {b: 0.0 for b in INPUT_BANDS}
    global_sq_err = 0.0
    
    with rasterio.open(sr_raster_path) as sr_src:
        lr_srcs = []
        for band in INPUT_BANDS:
            b_path = lr_dir / f"{band}.tif"
            if not b_path.exists():
                raise FileNotFoundError(f"LR band not found: {b_path}")
            lr_srcs.append(rasterio.open(b_path))
            
        scl_src = None
        scl_path = lr_dir / "SCL.tif"
        if scl_path.exists():
            scl_src = rasterio.open(scl_path)
            
        # Process block by block based on LR grid
        # e.g. 512x512 LR blocks -> 2048x2048 SR blocks
        from rasterio.windows import Window
        
        lr_w = lr_srcs[0].width
        lr_h = lr_srcs[0].height
        block_size = 512
        
        for y0 in range(0, lr_h, block_size):
            for x0 in range(0, lr_w, block_size):
                h = min(block_size, lr_h - y0)
                w = min(block_size, lr_w - x0)
                
                lr_window = Window(x0, y0, w, h)
                sr_window = Window(x0*4, y0*4, w*4, h*4)
                
                sr_arr = sr_src.read(window=sr_window)
                sr_down = downsample_box_4x4(sr_arr)
                
                lr_arr = np.stack([s.read(1, window=lr_window) for s in lr_srcs], axis=0)
                
                mask = np.ones((h, w), dtype=bool)
                if scl_src is not None:
                    scl = scl_src.read(1, window=lr_window)
                    mask = np.isin(scl, [4, 5, 6, 7])
                    
                valid = mask & ~np.isnan(lr_arr[0]) & ~np.isnan(sr_down[0])
                if not np.any(valid):
                    continue
                    
                num_valid = np.sum(valid)
                total_valid_pixels += num_valid
                
                for i, band in enumerate(INPUT_BANDS):
                    l = lr_arr[i][valid]
                    s = sr_down[i][valid]
                    
                    sq_errs[band] += np.sum((l - s)**2)
                    norm_diffs[band] += np.linalg.norm(l - s)**2
                    norm_lrs[band] += np.linalg.norm(l)**2
                    
                valid_3d = np.broadcast_to(valid, lr_arr.shape)
                global_sq_err += np.sum((lr_arr[valid_3d] - sr_down[valid_3d])**2)
                
                # SAM
                dot = np.sum(lr_arr * sr_down, axis=0)
                n_lr = np.linalg.norm(lr_arr, axis=0)
                n_sr = np.linalg.norm(sr_down, axis=0)
                denom = n_lr * n_sr
                valid_denom = (denom > 1e-8) & valid
                
                if np.any(valid_denom):
                    cosine = np.clip(dot[valid_denom] / denom[valid_denom], -1.0, 1.0)
                    sam = np.arccos(cosine)
                    sum_sam += np.sum(sam)
                    total_sam_pixels += np.sum(valid_denom)
                    
        for s in lr_srcs:
            s.close()
        if scl_src:
            scl_src.close()
            
    if total_valid_pixels == 0:
        return {"error": "No valid pixels found in scene"}
        
    # Finalize metrics
    for b in INPUT_BANDS:
        metrics[f"rmse_{b}"] = float(np.sqrt(sq_errs[b] / total_valid_pixels))
        metrics[f"rel_err_{b}"] = float(np.sqrt(norm_diffs[b]) / np.sqrt(norm_lrs[b])) if norm_lrs[b] > 0 else np.nan
        
    metrics["rmse_global"] = float(np.sqrt(global_sq_err / (total_valid_pixels * 4)))
    metrics["sam_rad"] = float(sum_sam / total_sam_pixels) if total_sam_pixels > 0 else np.nan
    
    return metrics

if __name__ == '__main__':
    import argparse
    import logging
    parser = argparse.ArgumentParser(description='Compute Consistency Metrics')
    parser.add_argument('--lr-dir', required=True, help='Path to prepared LR scene directory')
    parser.add_argument('--sr-raster', required=True, help='Path to SR raster (e.g., SEN2SR_scene.tif)')
    parser.add_argument('--output', default='consistency_metrics.json', help='Path to save JSON metrics')
    
    args = parser.parse_args()
    
    try:
        metrics = evaluate_scene_consistency(args.lr_dir, args.sr_raster)
        print(json.dumps(metrics, indent=2))
        
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
    except Exception as e:
        logging.error(f'Failed to compute consistency: {e}')
        raise
