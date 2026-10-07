"""
scripts/diag_consistency.py

Read-only diagnostic script for RAD-2b consistency checks.
Compares SR output and Bicubic output to the LR input using various downsampling kernels
and SR pixel shifts (-4 to +4).
"""

import sys
import time
import numpy as np
import rasterio
from rasterio.enums import Resampling
import torch
import torch.nn.functional as F
import scipy.ndimage
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infer.run_sen2sr import load_sen2sr_lite_model, permute_rgbn_to_bgrn

def downsample_box(sr):
    # sr is (B, H, W). H and W are multiples of 4.
    B, H, W = sr.shape
    return sr.reshape(B, H//4, 4, W//4, 4).mean(axis=(2, 4))

def downsample_strided(sr):
    return sr[:, ::4, ::4]

def downsample_gaussian(sr):
    out = np.zeros((sr.shape[0], sr.shape[1]//4, sr.shape[2]//4), dtype=sr.dtype)
    for b in range(sr.shape[0]):
        blurred = scipy.ndimage.gaussian_filter(sr[b], sigma=1.0) # sigma to be tuned, e.g. 1.0
        out[b] = blurred[::4, ::4]
    return out

def downsample_fft(sr):
    """
    FFT low-pass filter: transform, crop high frequencies, inverse transform.
    Assumes sr is (B, H, W) and we want (B, H/4, W/4).
    """
    B, H, W = sr.shape
    lr_H, lr_W = H // 4, W // 4
    
    out = np.zeros((B, lr_H, lr_W), dtype=sr.dtype)
    for b in range(B):
        # 2D FFT
        F_sr = np.fft.fftshift(np.fft.fft2(sr[b]))
        
        # Crop to central lr_H x lr_W
        start_y = (H - lr_H) // 2
        start_x = (W - lr_W) // 2
        F_lr = F_sr[start_y:start_y+lr_H, start_x:start_x+lr_W]
        
        # IFFT
        lr = np.fft.ifft2(np.fft.ifftshift(F_lr)).real
        
        # Adjust scale (FFT power conservation)
        # Ratio of pixels is 1/16
        out[b] = lr * (1 / 16.0)
    return out

MODES = {
    "box": downsample_box,
    "strided": downsample_strided,
    "gaussian": downsample_gaussian,
    "fft": downsample_fft
}

def shift_image(img, dy, dx):
    """Shift an image (B, H, W) by dy, dx in pixels. Pad with nan."""
    B, H, W = img.shape
    out = np.full_like(img, np.nan)
    
    src_y0 = max(0, -dy)
    src_y1 = min(H, H - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(W, W - dx)
    
    dst_y0 = max(0, dy)
    dst_y1 = min(H, H + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(W, W + dx)
    
    if dst_y0 < dst_y1 and dst_x0 < dst_x1:
        out[:, dst_y0:dst_y1, dst_x0:dst_x1] = img[:, src_y0:src_y1, src_x0:src_x1]
        
    return out

def run_bicubic_np(lr, scale=4):
    """Simple GDAL-based bicubic upsample for testing."""
    B, H, W = lr.shape
    out_shape = (B, H * scale, W * scale)
    
    # We can use rasterio memory file to do proper gdal cubic
    profile = {
        "driver": "GTiff",
        "height": H,
        "width": W,
        "count": B,
        "dtype": lr.dtype.name,
    }
    
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(lr)
        with memfile.open() as dataset:
            return dataset.read(out_shape=out_shape, resampling=Resampling.cubic)

def main():
    weights_dir = Path("models/SEN2SRLite")
    ex_data_path = weights_dir / "example_data.safetensor"
    
    if not ex_data_path.exists():
        print(f"Example data not found at {ex_data_path}")
        return
        
    import safetensors.torch
    ex = safetensors.torch.load_file(str(ex_data_path))
    
    # Model order is B04, B03, B02, B08
    lr_rgbn = ex["lr"][0, :4].numpy() # (4, 128, 128)
    # Convert to canonical for analysis [B02, B03, B04, B08]
    lr = permute_rgbn_to_bgrn(lr_rgbn)
    
    print(f"Loaded example data: shape {lr.shape}, dtype {lr.dtype}")
    for i, band in enumerate(["B02", "B03", "B04", "B08"]):
        print(f"  Input {band} - Min: {lr[i].min():.4f}, Max: {lr[i].max():.4f}, Mean: {lr[i].mean():.4f}")
        
    # Load model
    print("\nLoading SEN2SRLite model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_sen2sr_lite_model(weights_dir=weights_dir, device=device)
    
    has_hc = hasattr(model, "hard_constraint")
    print(f"Model loaded on {device}. Has hard_constraint layer: {has_hc}")
    
    # Run SR
    print("Running SR model...")
    with torch.no_grad():
        tensor_in = ex["lr"][:, :4].to(device) # RGBN order
        # HardConstraint needs sizes multiple of 128 usually, but example_data is 128x128
        out_tensor = model(tensor_in)
        sr_rgbn = out_tensor[0].cpu().numpy()
        
    sr = permute_rgbn_to_bgrn(sr_rgbn)
    print("SR Output generated.")
    for i, band in enumerate(["B02", "B03", "B04", "B08"]):
        print(f"  SR {band} - Min: {sr[i].min():.4f}, Max: {sr[i].max():.4f}, Mean: {sr[i].mean():.4f}")
        
    # Run Bicubic
    print("\nRunning Bicubic Baseline...")
    bic = run_bicubic_np(lr)
    for i, band in enumerate(["B02", "B03", "B04", "B08"]):
        print(f"  Bic {band} - Min: {bic[i].min():.4f}, Max: {bic[i].max():.4f}, Mean: {bic[i].mean():.4f}")
        
    # Consistency Tests
    print("\n--- Downsampling Consistency Test ---")
    
    shifts = range(-4, 5)
    
    for method_name, method_fn in MODES.items():
        print(f"\nMethod: {method_name.upper()}")
        
        best_sr_rmse = np.inf
        best_sr_shift = None
        best_bic_rmse = np.inf
        best_bic_shift = None
        
        # Test shifts
        for dy in shifts:
            for dx in shifts:
                sr_shifted = shift_image(sr, dy, dx)
                bic_shifted = shift_image(bic, dy, dx)
                
                # Downsample
                sr_ds = method_fn(sr_shifted)
                bic_ds = method_fn(bic_shifted)
                
                # Compare to LR (masking nan caused by shift)
                valid = ~np.isnan(sr_ds) & ~np.isnan(bic_ds)
                
                if not np.any(valid):
                    continue
                    
                sr_rmse = np.sqrt(np.mean((sr_ds[valid] - lr[valid])**2))
                bic_rmse = np.sqrt(np.mean((bic_ds[valid] - lr[valid])**2))
                
                if sr_rmse < best_sr_rmse:
                    best_sr_rmse = sr_rmse
                    best_sr_shift = (dy, dx)
                    
                if bic_rmse < best_bic_rmse:
                    best_bic_rmse = bic_rmse
                    best_bic_shift = (dy, dx)
                    
        print(f"  Best SR Shift: dy={best_sr_shift[0]}, dx={best_sr_shift[1]} | Min RMSE: {best_sr_rmse:.5f}")
        print(f"  Best Bicubic Shift: dy={best_bic_shift[0]}, dx={best_bic_shift[1]} | Min RMSE: {best_bic_rmse:.5f}")
        
        # Report per-band RMSE for best shift
        print("  Per-band RMSE at best shift:")
        sr_best = method_fn(shift_image(sr, *best_sr_shift))
        bic_best = method_fn(shift_image(bic, *best_bic_shift))
        valid_sr = ~np.isnan(sr_best)
        valid_bic = ~np.isnan(bic_best)
        
        for i, band in enumerate(["B02", "B03", "B04", "B08"]):
            v = valid_sr[i]
            sr_err = np.sqrt(np.mean((sr_best[i][v] - lr[i][v])**2))
            bic_err = np.sqrt(np.mean((bic_best[i][v] - lr[i][v])**2))
            print(f"    {band}: SR={sr_err:.5f}, Bicubic={bic_err:.5f}")

if __name__ == "__main__":
    main()
