import json
import argparse
import sys
import shutil
from pathlib import Path
from datetime import datetime, timezone
import pystac_client
import rasterio
import numpy as np

def backfill_metadata(dry_run=False):
    cache_dir = Path("data/raw/sentinel2")
    if not cache_dir.exists():
        print(f"{cache_dir} does not exist.")
        return
        
    for meta_file in cache_dir.glob("*/metadata.json"):
        with open(meta_file, "r") as f:
            meta = json.load(f)
            
        # Already backfilled?
        if "s2:processing_baseline" in meta and "legacy_boa_add_offset" in meta:
            print(f"Item {meta['item_id']} is already backfilled.")
            continue
            
        print(f"Backfilling {meta['item_id']}...")
        
        provider_url = meta.get("provider_url", "https://earth-search.aws.element84.com/v1")
        collection = meta.get("collection", "sentinel-2-l2a")
        
        # Test mock handling for dry runs
        client = pystac_client.Client.open(provider_url)
        search = client.search(collections=[collection], ids=[meta["item_id"]])
        items = list(search.items())
        if not items:
            print(f"Item {meta['item_id']} not found!")
            continue
            
        item = items[0]
        props = getattr(item, "properties", {})
        
        meta["s2:processing_baseline"] = props.get("s2:processing_baseline")
        meta["earthsearch:boa_offset_applied"] = props.get("earthsearch:boa_offset_applied")
        meta["updated"] = props.get("updated")
        
        raster_bands_info = {}
        for band_key in ["red", "nir"]:
            asset = item.assets.get(band_key)
            if asset:
                extra = getattr(asset, "extra_fields", {}) or {}
                rb = extra.get("raster:bands", [{}])[0]
                raster_bands_info[band_key] = {"scale": rb.get("scale"), "offset": rb.get("offset")}
        meta["raster_bands_info"] = raster_bands_info
        
        # Rename boa_add_offset to legacy_boa_add_offset
        if "boa_add_offset" in meta:
            meta["legacy_boa_add_offset"] = meta.pop("boa_add_offset")
            meta["legacy_offset_source"] = "legacy_unverified"
            
        meta["backfill_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        if not dry_run:
            meta_bak = meta_file.with_suffix(".json.bak")
            if not meta_bak.exists():
                shutil.copy2(meta_file, meta_bak)
                
            with open(meta_file, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"Updated {meta_file}")
        else:
            print(f"[Dry Run] Would update {meta_file} with {meta['earthsearch:boa_offset_applied']} and PB {meta['s2:processing_baseline']}")
            
        # Optional: real-data dry check
        if meta["earthsearch:boa_offset_applied"] is not None:
            from src.ingest.preprocess import resolve_boa_offset, dn_to_reflectance
            eff_offset, rule_id = resolve_boa_offset(
                meta["earthsearch:boa_offset_applied"], 
                meta["s2:processing_baseline"]
            )
            print(f"  Resolver gave effective offset: {eff_offset} (rule: {rule_id})")
            
            for b in ["B02", "B04", "B08"]:
                tif_path = meta_file.parent / f"{b}.tif"
                if tif_path.exists():
                    with rasterio.open(tif_path) as src:
                        # 1/10 decimation
                        data = src.read(1, out_shape=(src.height//10, src.width//10))
                        refl = dn_to_reflectance(
                            data, 
                            boa_add_offset=eff_offset, 
                            quantification_value=meta.get("quantification_value", 10000.0)
                        )
                        valid_refl = refl[~np.isnan(refl)]
                        if valid_refl.size > 0:
                            neg_frac = np.mean(valid_refl < 0)
                            print(f"    {b} negative fraction: {neg_frac:.4%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    backfill_metadata(args.dry_run)
