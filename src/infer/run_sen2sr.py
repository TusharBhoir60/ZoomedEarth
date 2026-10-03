"""
SEN2SR 10m -> 2.5m Super-Resolution Inference Wrapper.

Project: Sentinel-2 Super-Resolution + Trust Harness (SIH26142, NTRO)
Model: Pretrained SEN2SRLite (NonReference_RGBN_x4)
Bands: B02, B03, B04, B08 (10m) -> 4x upscale -> 2.5m grid
"""

import argparse
import os
import pathlib
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.transform import Affine
import safetensors.torch
import torch
import torch.nn as nn

from sen2sr.models.opensr_baseline.cnn import CNNSR
from sen2sr.models.tricks import HardConstraint
from sen2sr.nonreference import srmodel as rgbn_model

# Repository band order: [B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)]
# SEN2SR model band order: [B04 (Red), B03 (Green), B02 (Blue), B08 (NIR)]
PERMUTE_BGRN_TO_RGBN = [2, 1, 0, 3]
PERMUTE_RGBN_TO_BGRN = [2, 1, 0, 3]


def permute_bgrn_to_rgbn(data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Permute bands from repository convention [B02, B03, B04, B08]
    to official SEN2SRLite RGBN convention [B04, B03, B02, B08].
    """
    if isinstance(data, torch.Tensor):
        if data.ndim == 3:
            return data[PERMUTE_BGRN_TO_RGBN, :, :]
        elif data.ndim == 4:
            return data[:, PERMUTE_BGRN_TO_RGBN, :, :]
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got ndim={data.ndim}")
    elif isinstance(data, np.ndarray):
        if data.ndim == 3:
            return data[PERMUTE_BGRN_TO_RGBN, :, :]
        elif data.ndim == 4:
            return data[:, PERMUTE_BGRN_TO_RGBN, :, :]
        else:
            raise ValueError(f"Expected 3D or 4D array, got ndim={data.ndim}")
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def permute_rgbn_to_bgrn(data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Permute bands from official SEN2SRLite RGBN convention [B04, B03, B02, B08]
    back to repository convention [B02, B03, B04, B08].
    """
    if isinstance(data, torch.Tensor):
        if data.ndim == 3:
            return data[PERMUTE_RGBN_TO_BGRN, :, :]
        elif data.ndim == 4:
            return data[:, PERMUTE_RGBN_TO_BGRN, :, :]
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got ndim={data.ndim}")
    elif isinstance(data, np.ndarray):
        if data.ndim == 3:
            return data[PERMUTE_RGBN_TO_BGRN, :, :]
        elif data.ndim == 4:
            return data[:, PERMUTE_RGBN_TO_BGRN, :, :]
        else:
            raise ValueError(f"Expected 3D or 4D array, got ndim={data.ndim}")
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def scale_affine_transform(transform: Affine, scale: float = 4.0) -> Affine:
    """
    Scale the full 6-parameter linear component of an Affine transform for super-resolution.

    The 2.5m grid must preserve the exact spatial origin (c, f) and extent while
    scaling the pixel dimensions by 1/scale.
    """
    if scale <= 0:
        raise ValueError(f"Scale must be positive, got {scale}")
    return Affine(
        transform.a / scale,
        transform.b / scale,
        transform.c,
        transform.d / scale,
        transform.e / scale,
        transform.f,
    )


def load_sen2sr_lite_model(
    weights_dir: Union[str, pathlib.Path] = "models/SEN2SRLite",
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Load the pretrained SEN2SRLite NonReference RGBN x4 model with its HardConstraint.
    """
    weights_path = pathlib.Path(weights_dir)
    sr_model_file = weights_path / "sr_model.safetensor"
    hard_constraint_file = weights_path / "sr_hard_constraint.safetensor"

    if not sr_model_file.exists() or not hard_constraint_file.exists():
        raise FileNotFoundError(
            f"Model weights missing in {weights_path}. Required files: "
            f"{sr_model_file.name}, {hard_constraint_file.name}."
        )

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Construct CNNSR base network
    sr_rgbn_weights = safetensors.torch.load_file(sr_model_file)
    sr_rgbn_model = CNNSR(4, 4, 24, 4, True, False, 6)
    sr_rgbn_model.load_state_dict(sr_rgbn_weights)
    sr_rgbn_model.to(device)
    sr_rgbn_model.eval()

    for param in sr_rgbn_model.parameters():
        param.requires_grad = False

    # Construct and wrap with HardConstraint module
    hard_constraint_weights = safetensors.torch.load_file(hard_constraint_file)
    sr_rgbn_hard_constraint = HardConstraint(
        low_pass_mask=hard_constraint_weights["weights"].to(device),
        device=str(device),
    )

    model = rgbn_model(
        sr_model=sr_rgbn_model,
        hard_constraint=sr_rgbn_hard_constraint,
        device=str(device),
    )
    return model


def run_sen2sr(
    input_path: Union[str, pathlib.Path],
    output_path: Union[str, pathlib.Path],
    weights_dir: Union[str, pathlib.Path] = "models/SEN2SRLite",
    device: Optional[str] = None,
    precision: str = "fp32",
    scale: float = 4.0,
) -> Dict[str, Any]:
    """
    Execute 4x super-resolution on a 4-band Sentinel-2 GeoTIFF.

    Parameters
    ----------
    input_path : str or Path
        Input GeoTIFF containing bands [B02, B03, B04, B08] at 10m resolution.
    output_path : str or Path
        Target GeoTIFF path for the 2.5m super-resolved output.
    weights_dir : str or Path
        Directory where SEN2SRLite model weights are stored.
    device : str, optional
        Target device ('cuda', 'cuda:0', 'cpu'). Defaults to CUDA if available.
    precision : str
        Inference precision ('fp32' or 'fp16'). Default is 'fp32'.
    scale : float
        Super-resolution factor. Must be 4.0 for SEN2SRLite.

    Returns
    -------
    dict
        Execution telemetry and metadata.
    """
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input raster not found: {input_path}")

    if device is None:
        target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    # 1. Read input raster and validate geospatial profile
    with rasterio.open(input_path) as src:
        if src.count != 4:
            raise ValueError(
                f"Input raster must contain exactly 4 bands [B02, B03, B04, B08], but found {src.count} bands."
            )
        
        in_crs = src.crs
        in_transform = src.transform
        in_height = src.height
        in_width = src.width
        raw_data = src.read()  # Shape: (4, H, W)
        dtype_str = str(src.dtypes[0])

    # 2. Normalize reflectance
    # Sentinel-2 L2A BOA reflectance is typically uint16 scaled by 10,000
    if np.issubdtype(raw_data.dtype, np.integer):
        normalized_data = (raw_data.astype(np.float32) / 10000.0)
    else:
        normalized_data = raw_data.astype(np.float32)

    # 3. Permute bands: Repository [B02, B03, B04, B08] -> Model [B04, B03, B02, B08]
    permuted_data = permute_bgrn_to_rgbn(normalized_data)
    input_tensor = torch.from_numpy(permuted_data).unsqueeze(0).to(target_device)

    # 4. Load model
    model = load_sen2sr_lite_model(weights_dir=weights_dir, device=target_device)

    # 5. Measure inference & peak memory
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
        torch.cuda.synchronize(target_device)

    t0 = time.perf_counter()

    # The official HardConstraint mask is fixed at 512x512 Fourier space (128x128 LR).
    # Handle inputs < 128 with replicate padding and > 128 using official predict_large.
    with torch.no_grad():
        if in_height == 128 and in_width == 128:
            input_tensor = torch.from_numpy(permuted_data).unsqueeze(0).to(target_device)
            if precision == "fp16":
                if target_device.type != "cuda":
                    raise ValueError("FP16 autocast requires CUDA device")
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    output_tensor = model(input_tensor)
                output_tensor = output_tensor.float()
            elif precision == "fp32":
                output_tensor = model(input_tensor)
            else:
                raise ValueError(f"Unsupported precision '{precision}'. Must be 'fp32' or 'fp16'.")
            output_rgbn = output_tensor.squeeze(0).cpu()

        elif in_height < 128 or in_width < 128:
            pad_h = max(0, 128 - in_height)
            pad_w = max(0, 128 - in_width)
            tensor_in = torch.from_numpy(permuted_data).unsqueeze(0).to(target_device)
            tensor_padded = torch.nn.functional.pad(tensor_in, (0, pad_w, 0, pad_h), mode="replicate")
            if precision == "fp16":
                if target_device.type != "cuda":
                    raise ValueError("FP16 autocast requires CUDA device")
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out_padded = model(tensor_padded)
                out_padded = out_padded.float()
            elif precision == "fp32":
                out_padded = model(tensor_padded)
            else:
                raise ValueError(f"Unsupported precision '{precision}'. Must be 'fp32' or 'fp16'.")
            output_rgbn = out_padded[0, :, :int(in_height * scale), :int(in_width * scale)].cpu()

        else:
            from sen2sr.utils import predict_large
            tensor_in = torch.from_numpy(permuted_data).to(target_device)
            if precision == "fp16":
                if target_device.type != "cuda":
                    raise ValueError("FP16 autocast requires CUDA device")
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    output_rgbn = predict_large(tensor_in, model=model, overlap=32).cpu().float()
            elif precision == "fp32":
                output_rgbn = predict_large(tensor_in, model=model, overlap=32).cpu()
            else:
                raise ValueError(f"Unsupported precision '{precision}'. Must be 'fp32' or 'fp16'.")

    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
        peak_vram_mb = torch.cuda.max_memory_allocated(target_device) / (1024 ** 2)
    else:
        peak_vram_mb = 0.0

    latency_ms = (time.perf_counter() - t0) * 1000.0

    # 6. Unpermute output: Model [B04, B03, B02, B08] -> Repository [B02, B03, B04, B08]
    output_bgrn = permute_rgbn_to_bgrn(output_rgbn).numpy()

    # 7. Construct output georeference
    out_height = int(in_height * scale)
    out_width = int(in_width * scale)
    out_transform = scale_affine_transform(in_transform, scale=scale)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 8. Write GeoTIFF with tiled profile
    profile = {
        "driver": "GTiff",
        "height": out_height,
        "width": out_width,
        "count": 4,
        "dtype": "float32",
        "crs": in_crs,
        "transform": out_transform,
        "tiled": True,
        "blockxsize": min(256, out_width),
        "blockysize": min(256, out_height),
        "compress": "deflate",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output_bgrn)
        dst.set_band_description(1, "B02 - Blue (2.5m)")
        dst.set_band_description(2, "B03 - Green (2.5m)")
        dst.set_band_description(3, "B04 - Red (2.5m)")
        dst.set_band_description(4, "B08 - NIR (2.5m)")

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_shape": list(raw_data.shape),
        "output_shape": list(output_bgrn.shape),
        "input_dtype": dtype_str,
        "output_dtype": "float32",
        "device": str(target_device),
        "precision": precision,
        "scale": scale,
        "peak_vram_mb": round(peak_vram_mb, 2),
        "latency_ms": round(latency_ms, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Run SEN2SR 4x super-resolution on a 4-band Sentinel-2 GeoTIFF.")
    parser.add_argument("--input", "-i", required=True, help="Input 4-band GeoTIFF [B02, B03, B04, B08]")
    parser.add_argument("--output", "-o", required=True, help="Output 4-band 2.5m GeoTIFF")
    parser.add_argument("--weights", "-w", default="models/SEN2SRLite", help="Directory with SEN2SRLite weights")
    parser.add_argument("--device", "-d", default=None, help="Device to use ('cuda', 'cpu')")
    parser.add_argument("--precision", "-p", default="fp32", choices=["fp32", "fp16"], help="Inference precision")

    args = parser.parse_args()

    result = run_sen2sr(
        input_path=args.input,
        output_path=args.output,
        weights_dir=args.weights,
        device=args.device,
        precision=args.precision,
    )

    print("--- Inference Complete ---")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
