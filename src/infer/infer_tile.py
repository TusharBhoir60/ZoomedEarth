"""
T2.6 — Tile-level SEN2SR Inference.

Consumes one T2.5 10m tile directory and produces a validated 2.5m SEN2SR output.

Project: Sentinel-2 Super-Resolution + Trust Harness (SIH26142, NTRO)
"""
import argparse
import datetime
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import rasterio
import torch

from src.infer.run_sen2sr import (
    permute_bgrn_to_rgbn,
    permute_rgbn_to_bgrn,
    load_sen2sr_lite_model,
    scale_affine_transform,
)

# Canonical T2.5 tile band files in project order [B02, B03, B04, B08]
INPUT_BANDS = ["B02", "B03", "B04", "B08"]
# Official SEN2SR model band order
MODEL_BAND_ORDER = ["B04", "B03", "B02", "B08"]
SCALE = 4.0
OUTPUT_RESOLUTION_M = 2.5


def _get_git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _get_software_versions() -> Dict[str, str]:
    import rasterio as _rio
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "rasterio": _rio.__version__,
        "numpy": np.__version__,
    }
    try:
        import sen2sr
        versions["sen2sr"] = getattr(sen2sr, "__version__", "unknown")
    except ImportError:
        versions["sen2sr"] = "unavailable"
    return versions


def _get_hardware_info() -> Dict[str, str]:
    if torch.cuda.is_available():
        return {
            "device": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda or "unknown",
        }
    return {"device": "cpu", "cuda_version": "N/A"}


def validate_input_tile(tile_dir: Path) -> Dict[str, Any]:
    """
    Validate all pre-inference requirements against the T2.5 tile.

    Returns a dict of validated metadata (profile, dimensions, transform, arrays).
    """
    for band in INPUT_BANDS:
        p = tile_dir / f"{band}.tif"
        if not p.exists():
            raise FileNotFoundError(f"Missing required band file: {p}")

    reference_profile = None
    arrays = {}

    for band in INPUT_BANDS:
        with rasterio.open(tile_dir / f"{band}.tif") as src:
            if reference_profile is None:
                reference_profile = {
                    "crs": src.crs,
                    "transform": src.transform,
                    "width": src.width,
                    "height": src.height,
                    "dtype": src.dtypes[0],
                }
            else:
                if src.crs != reference_profile["crs"]:
                    raise ValueError(
                        f"{band}: CRS {src.crs} does not match reference {reference_profile['crs']}"
                    )
                if src.transform != reference_profile["transform"]:
                    raise ValueError(
                        f"{band}: transform does not match reference"
                    )
                if src.width != reference_profile["width"] or src.height != reference_profile["height"]:
                    raise ValueError(
                        f"{band}: dimensions ({src.width}×{src.height}) do not match "
                        f"reference ({reference_profile['width']}×{reference_profile['height']})"
                    )

            # Verify 10m resolution
            if abs(src.transform.a) != 10.0 or abs(src.transform.e) != 10.0:
                raise ValueError(
                    f"{band}: pixel size is not 10m "
                    f"(got {abs(src.transform.a)}m × {abs(src.transform.e)}m)"
                )

            arr = src.read(1)
            arrays[band] = arr

    # Check dtype — T2.4 prepared scenes are float32
    if reference_profile["dtype"] not in ("float32",):
        raise ValueError(
            f"Input dtype must be float32, got {reference_profile['dtype']}"
        )

    # Stack canonical band order [B02, B03, B04, B08]
    stacked = np.stack([arrays[b] for b in INPUT_BANDS], axis=0)  # (4, H, W)

    # Check for Inf in input (NaN is expected for masked pixels — handled next)
    if np.any(np.isinf(stacked)):
        raise ValueError("Input raster contains Inf values — cannot pass to model.")

    # Replace NaN masked pixels with 0.0 for model input.
    # This follows the established T2.4/T2.5 masking contract: masked pixels are NaN
    # in the spectral files. Passing NaN to a CNN would propagate NaN through convolutions.
    # We substitute 0.0, which the HardConstraint will reconstruct from the LR input.
    nan_mask = np.isnan(stacked)
    stacked_model_in = stacked.copy()
    stacked_model_in[nan_mask] = 0.0

    return {
        "profile": reference_profile,
        "stacked_canonical": stacked,         # original with NaN for masked
        "stacked_model_in": stacked_model_in, # NaN→0 substituted, safe for model
        "nan_mask": nan_mask,
    }


def validate_output(
    output_arr: np.ndarray,
    input_profile: Dict[str, Any],
) -> None:
    """Validate the SEN2SR output array before writing."""
    H_in = input_profile["height"]
    W_in = input_profile["width"]
    expected_H = int(H_in * SCALE)
    expected_W = int(W_in * SCALE)

    if output_arr.shape != (4, expected_H, expected_W):
        raise ValueError(
            f"Output shape {output_arr.shape} != expected (4, {expected_H}, {expected_W})"
        )

    if np.any(np.isnan(output_arr)):
        raise ValueError("Model produced NaN values in output — failing explicitly.")

    if np.any(np.isinf(output_arr)):
        raise ValueError("Model produced Inf values in output — failing explicitly.")


def infer_tile(
    tile_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    weights_dir: Union[str, Path] = "models/SEN2SRLite",
    device: Optional[str] = None,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run SEN2SR inference on one T2.5 tile directory.

    Parameters
    ----------
    tile_dir : path to the T2.5 tile directory containing B02–B08/SCL/cloud_mask/metadata.json
    output_dir : where to write SEN2SR.tif + metadata.json (defaults to
                 data/outputs/sen2sr/<item_id>/<tile_id>/)
    weights_dir : SEN2SRLite model weights directory
    device : torch device string; defaults to CUDA if available
    model : pre-loaded model (for testing); if None, loads from weights_dir

    Returns
    -------
    dict of audit/telemetry metadata
    """
    tile_dir = Path(tile_dir)
    if not tile_dir.exists():
        raise FileNotFoundError(f"Tile directory not found: {tile_dir}")

    # Load tile-level provenance
    meta_path = tile_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Tile metadata.json not found: {meta_path}")
    with open(meta_path) as f:
        tile_meta = json.load(f)

    item_id = tile_meta.get("item_id", "unknown")
    tile_id = tile_meta.get("tile_id", tile_dir.name)

    # Determine output directory
    if output_dir is None:
        output_dir = Path("data/outputs/sen2sr") / item_id / tile_id
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pre-inference validation
    validated = validate_input_tile(tile_dir)
    profile = validated["profile"]
    stacked_model_in = validated["stacked_model_in"]

    H_in = profile["height"]
    W_in = profile["width"]
    in_transform = profile["transform"]
    in_crs = profile["crs"]

    # 2. Select device
    if device is None:
        target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    # 3. Load model (or accept injected mock)
    if model is None:
        model = load_sen2sr_lite_model(weights_dir=weights_dir, device=target_device)

    # 4. Permute [B02,B03,B04,B08] → [B04,B03,B02,B08]
    permuted = permute_bgrn_to_rgbn(stacked_model_in)  # (4, H, W) in RGBN order

    input_tensor = torch.from_numpy(permuted).unsqueeze(0).to(target_device)  # (1, 4, H, W)

    # 5. Inference — FP32 only, no FP16
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
        torch.cuda.synchronize(target_device)

    t0 = time.perf_counter()

    with torch.no_grad():
        if H_in > 128 or W_in > 128:
            from sen2sr.utils import predict_large
            output_tensor = predict_large(
                permuted_tensor := torch.from_numpy(permuted).to(target_device),
                model=model,
                overlap=32,
            ).cpu()
        elif H_in == 128 and W_in == 128:
            output_tensor = model(input_tensor).squeeze(0).cpu()
        else:
            # Small tile — pad to 128×128, run model, crop
            pad_h = max(0, 128 - H_in)
            pad_w = max(0, 128 - W_in)
            padded = torch.nn.functional.pad(input_tensor, (0, pad_w, 0, pad_h), mode="replicate")
            out_padded = model(padded)
            output_tensor = out_padded[0, :, :int(H_in * SCALE), :int(W_in * SCALE)].cpu()

    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
        peak_vram_mb = round(torch.cuda.max_memory_allocated(target_device) / (1024 ** 2), 2)
    else:
        peak_vram_mb = 0.0

    inference_duration_s = time.perf_counter() - t0

    # 6. Convert output tensor to numpy and unpermute from RGBN to BGRN
    output_arr_rgbn = output_tensor.float().numpy()  # (4, H*4, W*4)
    output_arr = np.ascontiguousarray(permute_rgbn_to_bgrn(output_arr_rgbn))

    # 7. Output validation
    validate_output(output_arr, profile)

    # 8. Compute output geospatial contract
    out_H = int(H_in * SCALE)
    out_W = int(W_in * SCALE)
    out_transform = scale_affine_transform(in_transform, scale=SCALE)

    # 9. Write SEN2SR.tif
    out_raster_path = output_dir / "SEN2SR.tif"
    out_profile = {
        "driver": "GTiff",
        "height": out_H,
        "width": out_W,
        "count": 4,
        "dtype": "float32",
        "crs": in_crs,
        "transform": out_transform,
        "tiled": True,
        "blockxsize": min(256, out_W),
        "blockysize": min(256, out_H),
        "compress": "deflate",
    }
    with rasterio.open(out_raster_path, "w", **out_profile) as dst:
        dst.write(output_arr)
        for i, desc in enumerate(INPUT_BANDS):
            dst.set_band_description(i + 1, desc)

    # 10. Write metadata.json
    in_tf = in_transform
    out_tf = out_transform
    inference_meta = {
        "item_id": item_id,
        "tile_id": tile_id,
        "source_tile": str(tile_dir),
        "model": "SEN2SRLite",
        "model_version_or_identifier": "NonReference_RGBN_x4",
        "input_band_order": INPUT_BANDS,
        "model_band_order": MODEL_BAND_ORDER,
        "output_band_order": INPUT_BANDS,
        "band_order_contract_version": "v1.0",
        "input_width": W_in,
        "input_height": H_in,
        "output_width": out_W,
        "output_height": out_H,
        "input_resolution": "10m",
        "output_resolution": "2.5m",
        "scale_factor": SCALE,
        "nodata_policy": "NaN values replaced with 0.0 before inference",
        "normalization_policy": "not_documented",
        "clipping_policy": "not_documented",
        "tile_size": W_in,
        "tile_overlap": "not_documented",
        "tile_stride": "not_documented",
        "acquisition_date": "not_documented",
        "acquisition_datetime": "not_documented",
        "crs": in_crs.to_string() if in_crs else "unknown",
        "input_transform": [in_tf.a, in_tf.b, in_tf.c, in_tf.d, in_tf.e, in_tf.f],
        "output_transform": [out_tf.a, out_tf.b, out_tf.c, out_tf.d, out_tf.e, out_tf.f],
        "dtype": "float32",
        "precision": "fp32",
        "inference_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "inference_duration_seconds": round(inference_duration_s, 4),
        "peak_vram_mb": peak_vram_mb,
        "git_sha": _get_git_sha(),
        "software_versions": _get_software_versions(),
        "hardware": _get_hardware_info(),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(inference_meta, f, indent=2)

    return inference_meta


def cli_main():
    parser = argparse.ArgumentParser(
        description="Run SEN2SR inference on a single T2.5 tile."
    )
    parser.add_argument("--tile", required=True, help="Path to T2.5 tile directory")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument(
        "--weights", default="models/SEN2SRLite", help="SEN2SRLite weights dir"
    )
    parser.add_argument("--device", default=None, help="Torch device (cuda/cpu)")
    args = parser.parse_args()

    t0 = time.perf_counter()
    meta = infer_tile(
        tile_dir=args.tile,
        output_dir=args.output_dir,
        weights_dir=args.weights,
        device=args.device,
    )
    elapsed = time.perf_counter() - t0

    print(f"Output: {meta['source_tile']} -> {args.output_dir or 'data/outputs/sen2sr/...'}")
    print(f"Input:  {meta['input_width']}×{meta['input_height']} @ 10m")
    print(f"Output: {meta['output_width']}×{meta['output_height']} @ 2.5m")
    print(f"Inference time: {meta['inference_duration_seconds']:.2f}s")
    if meta["peak_vram_mb"]:
        print(f"Peak VRAM: {meta['peak_vram_mb']:.1f} MB")


if __name__ == "__main__":
    cli_main()
