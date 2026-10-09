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

CANONICAL_BAND_ORDER = ["B02", "B03", "B04", "B08"]
BAND_ORDER_CONTRACT_VERSION = "v1.0"

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

# Band descriptions omitted, derived dynamically


def run_bicubic(
    input_path: Union[str, pathlib.Path],
    output_path: Union[str, pathlib.Path],
    scale: float = 4.0,
    metadata_path: Union[str, pathlib.Path, None] = None,
    band_order: list = None,
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

    if band_order is None and metadata_path is None:
        raise ValueError("Must provide either metadata_path or explicit band_order.")

    if band_order is None:
        import json
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        band_order = meta.get("band_order")
        if band_order is None:
            raise ValueError(f"metadata_path {metadata_path} lacks 'band_order'.")

    if not input_path.exists():
        raise FileNotFoundError(f"Input raster not found: {input_path}")

    # ── 1. Read input and validate ────────────────────────────────────────────
    if input_path.is_dir():
        if band_order is None:
            raise ValueError("Must provide explicit band_order or metadata_path when input is a directory.")
        source_paths = [input_path / f"{b}.tif" for b in band_order]
        for p in source_paths:
            if not p.exists():
                raise FileNotFoundError(f"Required band file not found: {p}")
        open_kwargs = {}
    else:
        source_paths = [input_path] * REQUIRED_BAND_COUNT
        open_kwargs = {}

    with rasterio.open(source_paths[0], **open_kwargs) as src0:
        if not input_path.is_dir() and src0.count != REQUIRED_BAND_COUNT:
            raise ValueError(
                f"Input raster must contain exactly 4 bands [B02, B03, B04, B08], "
                f"but found {src0.count} bands."
            )
        if src0.crs is None:
            raise ValueError(
                f"Input raster has no CRS. A valid coordinate reference system "
                f"is required for georeferenced bicubic resampling."
            )
        if src0.transform is None or src0.transform == Affine.identity():
            raise ValueError(
                f"Input raster has no valid affine geotransform. "
                f"Georeferencing is required."
            )

        in_crs = src0.crs
        in_transform = src0.transform
        in_height = src0.height
        in_width = src0.width
        dtype_str = str(src0.dtypes[0])
        src_nodata = src0.nodata

    out_height = int(in_height * scale)
    out_width = int(in_width * scale)

    # ── 2. Construct output georeference ─────────────────────────────────────
    out_transform = scale_affine_transform(in_transform, scale=scale)

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
        "blockxsize": min(512, out_width),
        "blockysize": min(512, out_height),
        "compress": "deflate",
        "bigtiff": "YES"
    }
    if src_nodata is not None:
        profile["nodata"] = src_nodata

    # ── 3. Bicubic upsample via GDAL out-of-core VRT ────────────────────────
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window
    
    t0 = time.perf_counter()

    with rasterio.open(output_path, "w", **profile) as dst:
        for idx, (path, desc) in enumerate(zip(source_paths, band_order), start=1):
            dst.set_band_description(idx, desc)
            src_band_idx = 1 if input_path.is_dir() else idx
            
            with rasterio.open(path) as src:
                with WarpedVRT(src, resampling=Resampling.cubic, crs=in_crs, 
                               transform=out_transform, height=out_height, width=out_width) as vrt:
                    
                    for _, window in dst.block_windows(1):
                        data = vrt.read(src_band_idx, window=window)
                        
                        # Apply normalisation identical to T1.1
                        if np.issubdtype(src.dtypes[src_band_idx-1], np.integer):
                            normalized = data.astype(np.float32) / 10000.0
                        else:
                            normalized = data.astype(np.float32)
                            
                        dst.write(normalized, idx, window=window)

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_shape": [REQUIRED_BAND_COUNT, in_height, in_width],
        "output_shape": [REQUIRED_BAND_COUNT, out_height, out_width],
        "input_dtype": dtype_str,
        "output_dtype": "float32",
        "resampling": "cubic",
        "scale": scale,
        "latency_ms": round(latency_ms, 2),
        "input_band_order": band_order,
        "model_band_order": None,  # bicubic is band-agnostic; no permutation applied
        "output_band_order": band_order,
        "band_order_contract_version": BAND_ORDER_CONTRACT_VERSION,
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
    parser.add_argument("--band-order", nargs="+", help="Explicit band order e.g. B02 B03 B04 B08")
    parser.add_argument("--metadata-path", help="Path to prepared-scene metadata.json")

    args = parser.parse_args()

    result = run_bicubic(
        input_path=args.input,
        output_path=args.output,
        scale=args.scale,
        band_order=args.band_order,
        metadata_path=args.metadata_path,
    )

    print("--- Bicubic Resampling Complete ---")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
