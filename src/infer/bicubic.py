"""
Bicubic 10m → 2.5m Baseline Resampling.

Project: Sentinel-2 Super-Resolution + Trust Harness (SIH26142, NTRO)
Method:  Genuine bicubic interpolation via rasterio / GDAL (Resampling.cubic)
Bands:   B02, B03, B04, B08 (10m) → 4× bicubic upsample → 2.5m grid

Preprocessing convention:
    Identical to T1.1 (run_sen2sr.py):
    - Integer dtype (e.g. uint16 L2A BOA × 10000): divide by 10000.0 → float32
    - Float dtype: cast to float32 unchanged

Band order:
    Input and output maintain [B02, B03, B04, B08].
    No channel permutation is applied (bicubic is band-agnostic).

Affine transform:
    Uses scale_affine_transform() from run_sen2sr to guarantee identical
    geospatial contract as the SEN2SR baseline. The full 6-parameter linear
    component is scaled; no north-up assumption is made.
"""

import argparse
import pathlib
import time
from typing import Any, Dict, Union

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.crs import CRS
from rasterio.transform import Affine

# NOTE: scale_affine_transform is intentionally inlined here so bicubic.py
# is self-contained for both `python -m pytest` and direct CLI invocation.
# The formula is identical to run_sen2sr.scale_affine_transform (see docs/DECISIONS.md D004).
# If the formula ever changes in T1.1, it must be updated here too.
def scale_affine_transform(transform: Affine, scale: float = 4.0) -> Affine:
    """
    Scale the full 6-parameter linear component of an Affine transform.

    Identical to src.infer.run_sen2sr.scale_affine_transform.
    Origin (c, f) is invariant; no north-up assumption.
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

# Required band count for all Sentinel-2 RGBN inputs
REQUIRED_BAND_COUNT = 4

# Band descriptions written to output (same format as T1.1)
BAND_DESCRIPTIONS = [
    "B02 - Blue (2.5m)",
    "B03 - Green (2.5m)",
    "B04 - Red (2.5m)",
    "B08 - NIR (2.5m)",
]


def run_bicubic(
    input_path: Union[str, pathlib.Path],
    output_path: Union[str, pathlib.Path],
    scale: float = 4.0,
) -> Dict[str, Any]:
    """
    Execute 4× bicubic resampling on a 4-band Sentinel-2 GeoTIFF.

    Parameters
    ----------
    input_path : str or Path
        Input GeoTIFF containing bands [B02, B03, B04, B08] at 10m resolution.
    output_path : str or Path
        Target GeoTIFF path for the 2.5m bicubic output.
    scale : float
        Upsampling factor. Default 4.0 (10m → 2.5m).

    Returns
    -------
    dict
        Execution telemetry and metadata, in the same format as T1.1's
        run_sen2sr() for later comparison.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the input raster does not have exactly 4 bands, or is missing
        CRS/georeferencing.
    """
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input raster not found: {input_path}")

    # ── 1. Read input and validate ────────────────────────────────────────────
    with rasterio.open(input_path) as src:
        if src.count != REQUIRED_BAND_COUNT:
            raise ValueError(
                f"Input raster must contain exactly 4 bands [B02, B03, B04, B08], "
                f"but found {src.count} bands."
            )
        if src.crs is None:
            raise ValueError(
                f"Input raster has no CRS. A valid coordinate reference system "
                f"is required for georeferenced bicubic resampling."
            )
        if src.transform is None or src.transform == Affine.identity():
            raise ValueError(
                f"Input raster has no valid affine geotransform. "
                f"Georeferencing is required."
            )

        in_crs = src.crs
        in_transform = src.transform
        in_height = src.height
        in_width = src.width
        dtype_str = str(src.dtypes[0])

        # ── 2. Normalise reflectance — EXACT T1.1 convention ─────────────────
        # T1.1 (run_sen2sr.py lines 199-203):
        #   Integer dtype → divide by 10000.0 → float32
        #   Float dtype   → cast to float32 unchanged
        raw_data = src.read()  # shape: (4, H, W)

        out_height = int(in_height * scale)
        out_width = int(in_width * scale)

        # ── 3. Bicubic upsample via GDAL cubic kernel ─────────────────────────
        # rasterio.read(out_shape=...) delegates to gdal_warp under the hood.
        # Resampling.cubic is the genuine cubic convolution kernel, the same
        # method used by gdalwarp -r cubic and QGIS "Cubic" resampling.
        t0 = time.perf_counter()
        resampled_raw = src.read(
            out_shape=(REQUIRED_BAND_COUNT, out_height, out_width),
            resampling=Resampling.cubic,
        )  # dtype = same as source

    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Apply the same normalisation rule as T1.1, post-resampling.
    # Resampling first (in native integer space) then normalising preserves
    # the numeric scale that T1.1 operates in after loading.
    if np.issubdtype(resampled_raw.dtype, np.integer):
        normalized_data = resampled_raw.astype(np.float32) / 10000.0
    else:
        normalized_data = resampled_raw.astype(np.float32)

    # ── 4. Construct output georeference ─────────────────────────────────────
    # Reuse scale_affine_transform from T1.1: scales the full 6-parameter
    # linear component, preserving origin, spatial extent, and rotation/shear.
    out_transform = scale_affine_transform(in_transform, scale=scale)

    # ── 5. Write output GeoTIFF ───────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": out_height,
        "width": out_width,
        "count": REQUIRED_BAND_COUNT,
        "dtype": "float32",
        "crs": in_crs,
        "transform": out_transform,
        "tiled": True,
        "blockxsize": min(256, out_width),
        "blockysize": min(256, out_height),
        "compress": "deflate",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(normalized_data)
        for idx, desc in enumerate(BAND_DESCRIPTIONS, start=1):
            dst.set_band_description(idx, desc)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_shape": list(raw_data.shape),
        "output_shape": list(normalized_data.shape),
        "input_dtype": dtype_str,
        "output_dtype": "float32",
        "resampling": "cubic",
        "scale": scale,
        "latency_ms": round(latency_ms, 2),
        "in_transform": {
            "a": in_transform.a,
            "b": in_transform.b,
            "c": in_transform.c,
            "d": in_transform.d,
            "e": in_transform.e,
            "f": in_transform.f,
        },
        "out_transform": {
            "a": out_transform.a,
            "b": out_transform.b,
            "c": out_transform.c,
            "d": out_transform.d,
            "e": out_transform.e,
            "f": out_transform.f,
        },
        "crs": str(in_crs),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run 4× bicubic resampling on a 4-band Sentinel-2 GeoTIFF."
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input 4-band GeoTIFF [B02, B03, B04, B08] at 10m")
    parser.add_argument("--output", "-o", required=True,
                        help="Output 4-band 2.5m bicubic GeoTIFF")
    parser.add_argument("--scale", "-s", type=float, default=4.0,
                        help="Upsampling scale factor (default: 4.0)")

    args = parser.parse_args()

    result = run_bicubic(
        input_path=args.input,
        output_path=args.output,
        scale=args.scale,
    )

    print("--- Bicubic Resampling Complete ---")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
