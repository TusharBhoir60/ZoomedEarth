"""
Cloud Masking + 10m Scene Preparation for Sentinel-2 L2A.

Project: Sentinel-2 Super-Resolution + Trust Harness (SIH26142, NTRO)
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import rasterio

from src.ingest.preprocess import CANONICAL_BANDS, to_reflectance

# SCL Masking Policy
# The following classes represent invalid pixels for clear-surface observations.
# 0: No Data
# 1: Saturated or defective
# 3: Cloud Shadows
# 8: Cloud Medium Probability
# 9: Cloud High Probability
# 10: Thin Cirrus
# 11: Snow / Ice
MASK_CLASSES = [0, 1, 3, 8, 9, 10, 11]

def load_t23_metadata(item_id: str, cache_dir: str = "data/raw/sentinel2") -> Dict[str, Any]:
    """Load T2.3 provenance metadata to extract provider URL and collection."""
    metadata_path = Path(cache_dir) / item_id / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"T2.3 cache metadata not found for {item_id}. Run T2.3 download first.")
    with open(metadata_path, "r") as f:
        return json.load(f)

def get_product_metadata(item_id: str, cache_dir: str = "data/raw/sentinel2") -> Tuple[float, float]:
    """Extract BOA_ADD_OFFSET and QUANTIFICATION_VALUE from local cache metadata."""
    metadata_path = Path(cache_dir) / item_id / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"T2.3 cache metadata not found for {item_id}.")
        
    with open(metadata_path, "r") as f:
        meta = json.load(f)
        
    if "boa_add_offset" not in meta or "quantification_value" not in meta:
        raise ValueError(
            f"Reflectance metadata (boa_add_offset, quantification_value) is missing from "
            f"T2.3 cache metadata for {item_id}. Network STAC access is prohibited in T2.4. "
            "Please ensure T2.3 cache includes this metadata."
        )
        
    return float(meta["boa_add_offset"]), float(meta["quantification_value"])

def validate_and_read_grids(item_id: str, cache_dir: str) -> Tuple[Dict[str, np.ndarray], Any]:
    """Read cached assets and validate their strict 10m spatial compatibility."""
    base_dir = Path(cache_dir) / item_id
    
    required_files = ["B02.tif", "B03.tif", "B04.tif", "B08.tif", "SCL.tif"]
    for f in required_files:
        if not (base_dir / f).exists():
            raise FileNotFoundError(f"Missing required asset {f} in cache. Run T2.3 download first.")
            
    reference_profile = None
    raster_data = {}
    
    for f in required_files:
        band_name = f.replace(".tif", "")
        with rasterio.open(base_dir / f) as src:
            profile = src.profile
            if reference_profile is None:
                reference_profile = profile
            else:
                if src.crs != reference_profile["crs"]:
                    raise ValueError(f"{f} CRS {src.crs} does not match reference CRS")
                if src.transform != reference_profile["transform"]:
                    raise ValueError(f"{f} transform does not match reference transform")
                if src.width != reference_profile["width"] or src.height != reference_profile["height"]:
                    raise ValueError(f"{f} dimensions do not match reference dimensions")
                
            # Verify exactly 10m grid
            if abs(profile["transform"].a) != 10.0 or abs(profile["transform"].e) != 10.0:
                raise ValueError(f"{f} pixel size is not 10m.")
                
            raster_data[band_name] = src.read(1)
            
    return raster_data, reference_profile

def prepare_scene(item_id: str, raw_cache_dir: str = "data/raw/sentinel2", processed_dir: str = "data/processed/sentinel2") -> None:
    """
    Produce a geospatially consistent, masked, 10m prepared-scene layer.
    """
    # 1. Load metadata and extract product conversion parameters
    t23_metadata = load_t23_metadata(item_id, raw_cache_dir)
    boa_add_offset, quantification_value = get_product_metadata(item_id, raw_cache_dir)
    
    # 2. Validate grids and read arrays
    raster_data, profile = validate_and_read_grids(item_id, raw_cache_dir)
    
    # 3. Create boolean cloud mask from SCL
    scl = raster_data["SCL"]
    cloud_mask = np.isin(scl, MASK_CLASSES)
    
    # 4. Prepare canonical stack
    band_array = np.stack([raster_data[b] for b in CANONICAL_BANDS], axis=0)
    
    # Convert to reflectance using T2.1 contract
    # DN=0 becomes np.nan inside to_reflectance
    reflectance = to_reflectance(band_array, boa_add_offset, quantification_value, nodata_value=0.0)
    
    # Apply explicit representation (NaN) for SCL-masked pixels
    reflectance[:, cloud_mask] = np.nan
    
    # 5. Output
    out_dir = Path(processed_dir) / item_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_profile = profile.copy()
    out_profile.update(dtype=rasterio.float32, nodata=np.nan, count=1)
    
    # Write canonical spectral bands
    for i, band in enumerate(CANONICAL_BANDS):
        out_path = out_dir / f"{band}.tif"
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(reflectance[i], 1)
            
    # Write unadulterated SCL and the explicit binary cloud_mask
    mask_profile = profile.copy()
    mask_profile.update(dtype=rasterio.uint8, nodata=None, count=1)
    
    with rasterio.open(out_dir / "SCL.tif", "w", **mask_profile) as dst:
        dst.write(scl, 1)
        
    with rasterio.open(out_dir / "cloud_mask.tif", "w", **mask_profile) as dst:
        dst.write(cloud_mask.astype(np.uint8), 1)
        
    # 6. Generate provenance and statistics
    valid_pixels = int(np.sum(~cloud_mask))
    masked_pixels = int(np.sum(cloud_mask))
    total_pixels = cloud_mask.size
    
    metadata = {
        "item_id": item_id,
        "source_cache": str(Path(raw_cache_dir) / item_id),
        "bands": CANONICAL_BANDS,
        "band_order": CANONICAL_BANDS,
        "crs": profile["crs"].to_string() if profile["crs"] else None,
        "transform": [profile["transform"].a, profile["transform"].b, profile["transform"].c,
                      profile["transform"].d, profile["transform"].e, profile["transform"].f],
        "width": profile["width"],
        "height": profile["height"],
        "resolution": "10m",
        "mask_policy": {
            "description": "SCL-based masking (NoData, Defective, Cloud Shadows, Cloud Medium/High, Thin Cirrus, Snow)",
            "masked_classes": MASK_CLASSES,
            "invalid_pixel_representation": "NaN"
        },
        "valid_pixel_count": valid_pixels,
        "masked_pixel_count": masked_pixels,
        "mask_fraction": masked_pixels / total_pixels if total_pixels > 0 else 0,
        "preparation_version": "v1.0"
    }
    
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

def cli_main():
    import argparse
    from src.ingest.scene_selection import select_best_scene
    
    parser = argparse.ArgumentParser(description="Prepare a clean 10m scene for an AOI.")
    parser.add_argument("--aoi", required=True, help="AOI ID")
    args = parser.parse_args()
    
    print(f"Resolving item for AOI '{args.aoi}'...")
    selection_metadata = select_best_scene(args.aoi)
    item_id = selection_metadata["selected_item_id"]
    
    print(f"Preparing scene {item_id}...")
    prepare_scene(item_id)
    print("Done.")

if __name__ == "__main__":
    cli_main()
