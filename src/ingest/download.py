"""
Download and Cache module for Sentinel-2 L2A assets.

Project: Sentinel-2 Super-Resolution + Trust Harness (SIH26142, NTRO)
"""
import os
import json
import urllib.request
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

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
    
    mapping = get_asset_mapping(provider_url)
    asset_records = []
    
    for band in REQUIRED_BANDS:
        expected_key = mapping[band]
        asset = item.assets[expected_key]
        href = asset.href
        
        local_path = item_cache_dir / f"{band}.tif"
        
        # 4. Cache behavior
        if not validate_cached_file(local_path):
            download_file(href, local_path)
            
        # Record metadata
        asset_records.append({
            "band": band,
            "local_path": str(local_path),
            "source_href": href,
            "file_size_bytes": local_path.stat().st_size
        })
        
    # 5. Save metadata.json
    download_timestamp = datetime.now(timezone.utc).isoformat()
    
    cache_metadata = {
        "item_id": item_id,
        "collection": collection,
        "provider_url": provider_url,
        "acquisition_datetime": selection_metadata["acquisition_datetime"],
        "download_timestamp": download_timestamp,
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
