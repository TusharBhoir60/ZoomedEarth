"""
Download and Cache module for Sentinel-2 L2A assets.

Project: Sentinel-2 Super-Resolution + Trust Harness (SIH26142, NTRO)
"""
import os
import json
import urllib.request
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

import pystac_client

from src.ingest.stac import get_asset_mapping, verify_required_assets, REQUIRED_BANDS

def validate_cached_file(path: Path) -> bool:
    """Check if file exists and has size > 0."""
    return path.exists() and path.stat().st_size > 0

def download_file(url: str, dest_path: Path) -> None:
    """Download a file safely to a temporary path, then rename it."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    
    try:
        urllib.request.urlretrieve(url, temp_path)
        if not validate_cached_file(temp_path):
            raise ValueError(f"Downloaded file is empty: {url}")
        os.rename(temp_path, dest_path)
    except Exception as e:
        if temp_path.exists():
            os.remove(temp_path)
        raise RuntimeError(f"Failed to download {url}: {str(e)}")

def extract_reflectance_metadata(item: Any) -> Tuple[float, float]:
    """
    Extract BOA_ADD_OFFSET and QUANTIFICATION_VALUE from STAC item properties or assets.
    """
    props = getattr(item, "properties", {}) or {}
    
    # Quantification value
    qv = props.get("s2:quantification_value") or props.get("quantification_value")
    if qv is None:
        blue_asset = item.assets.get("blue") or item.assets.get("B02")
        if blue_asset:
            extra = getattr(blue_asset, "extra_fields", {}) or {}
            raster_bands = extra.get("raster:bands", [])
            if raster_bands and isinstance(raster_bands[0], dict) and "scale" in raster_bands[0]:
                scale = raster_bands[0]["scale"]
                if scale > 0:
                    qv = 1.0 / scale
    if qv is None:
        qv = 10000.0
        
    # BOA add offset
    offset = props.get("s2:boa_add_offset") or props.get("boa_add_offset")
    if offset is None:
        blue_asset = item.assets.get("blue") or item.assets.get("B02")
        if blue_asset:
            extra = getattr(blue_asset, "extra_fields", {}) or {}
            raster_bands = extra.get("raster:bands", [])
            if raster_bands and isinstance(raster_bands[0], dict) and "offset" in raster_bands[0]:
                offset_val = raster_bands[0]["offset"]
                offset = offset_val * float(qv)
    if offset is None:
        if props.get("earthsearch:boa_offset_applied") or (str(props.get("s2:processing_baseline", "00.00")) >= "04.00"):
            offset = -1000.0
        else:
            offset = 0.0
            
    return float(offset), float(qv)

def resolve_and_download_assets(selection_metadata: Dict[str, Any], cache_dir: str = "data/raw/sentinel2") -> Dict[str, Any]:
    """
    Given the T2.2 selection metadata, download the required assets.
    """
    provider_url = selection_metadata["provider_url"]
    collection = selection_metadata["stac_collection"]
    item_id = selection_metadata["selected_item_id"]
    
    # 1. Re-fetch STAC item to get asset hrefs
    client = pystac_client.Client.open(provider_url)
    search = client.search(collections=[collection], ids=[item_id])
    items = list(search.items())
    if not items:
        raise ValueError(f"STAC Item '{item_id}' not found at {provider_url}")
    item = items[0]
    
    # 2. Setup deterministic cache directory
    item_cache_dir = Path(cache_dir) / item_id
    item_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Verify and resolve required assets
    verify_required_assets(item, provider_url)
    
    # 4. Extract reflectance metadata
    boa_add_offset, quantification_value = extract_reflectance_metadata(item)
    
    mapping = get_asset_mapping(provider_url)
    asset_records = []
    
    for band in REQUIRED_BANDS:
        expected_key = mapping[band]
        asset = item.assets[expected_key]
        href = asset.href
        
        local_path = item_cache_dir / f"{band}.tif"
        
        # 5. Cache behavior
        if not validate_cached_file(local_path):
            print(f"  Downloading band {band} to {local_path}...", flush=True)
            download_file(href, local_path)
        else:
            print(f"  Using cached band {band} at {local_path}", flush=True)
            
        # Record metadata
        asset_records.append({
            "band": band,
            "local_path": str(local_path),
            "source_href": href,
            "file_size_bytes": local_path.stat().st_size
        })
        
    # 6. Save metadata.json
    download_timestamp = datetime.now(timezone.utc).isoformat()
    
    props = getattr(item, "properties", {})
    s2_processing_baseline = props.get("s2:processing_baseline")
    earthsearch_boa_offset_applied = props.get("earthsearch:boa_offset_applied")
    updated = props.get("updated")
    
    # Extract scale/offset for informational purposes
    raster_bands_info = {}
    for band_key in ["red", "nir"]:
        asset = item.assets.get(band_key)
        if asset:
            extra = getattr(asset, "extra_fields", {}) or {}
            rb = extra.get("raster:bands", [{}])[0]
            raster_bands_info[band_key] = {"scale": rb.get("scale"), "offset": rb.get("offset")}
    
    cache_metadata = {
        "item_id": item_id,
        "collection": collection,
        "provider_url": provider_url,
        "acquisition_datetime": selection_metadata["acquisition_datetime"],
        "download_timestamp": download_timestamp,
        "legacy_boa_add_offset": boa_add_offset,
        "legacy_offset_source": "legacy_unverified",
        "quantification_value": quantification_value,
        "s2:processing_baseline": s2_processing_baseline,
        "earthsearch:boa_offset_applied": earthsearch_boa_offset_applied,
        "raster_bands_info": raster_bands_info,
        "updated": updated,
        "assets": asset_records,
        "download_version": "v1.0"
    }
    
    metadata_path = item_cache_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(cache_metadata, f, indent=2)
        
    return cache_metadata

def cli_main():
    import argparse
    from src.ingest.scene_selection import select_best_scene
    
    parser = argparse.ArgumentParser(description="Download cached assets for a selected Sentinel-2 scene.")
    parser.add_argument("--aoi", required=True, help="AOI ID to select and download")
    args = parser.parse_args()
    
    print(f"Selecting best scene for AOI '{args.aoi}'...")
    selection_metadata = select_best_scene(args.aoi)
    
    print(f"Selected item: {selection_metadata['selected_item_id']}")
    print("Resolving and downloading assets...")
    cache_metadata = resolve_and_download_assets(selection_metadata)
    
    print(f"Download complete. Metadata saved at {Path('data/raw/sentinel2') / selection_metadata['selected_item_id'] / 'metadata.json'}")

if __name__ == "__main__":
    cli_main()
