"""
Deterministic LR Tile Generation and Manifest Creation.

Project: Sentinel-2 Super-Resolution + Trust Harness (SIH26142, NTRO)
"""
import csv
import json
from pathlib import Path
from typing import Dict, Any, List
import yaml
import numpy as np
import rasterio
from rasterio.windows import Window

BANDS = ["B02", "B03", "B04", "B08"]

def load_tiling_config(config_path: str = "configs/tiling.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {"tile_size": 128, "overlap": 16, "edge_policy": "pad"}
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("tiling", {"tile_size": 128, "overlap": 16, "edge_policy": "pad"})

def get_scene_split(item_id: str) -> str:
    """Attempt to find the split from AOI configuration or return 'unknown'."""
    return "unknown"

def create_tiles(
    item_id: str,
    processed_dir: str = "data/processed/sentinel2",
    tiles_dir: str = "data/tiles/sentinel2",
    config_path: str = "configs/tiling.yaml"
) -> None:
    config = load_tiling_config(config_path)
    tile_size = config["tile_size"]
    overlap = config["overlap"]
    
    if not isinstance(tile_size, int) or tile_size <= 0:
        raise ValueError(f"Invalid tile_size: {tile_size}")
    if not isinstance(overlap, int) or not (0 <= overlap < tile_size):
        raise ValueError(f"Invalid overlap: {overlap}")
        
    stride = tile_size - overlap
    
    in_dir = Path(processed_dir) / item_id
    out_dir = Path(tiles_dir) / item_id
    
    if not in_dir.exists():
        raise FileNotFoundError(f"Prepared scene not found: {in_dir}")
        
    with rasterio.open(in_dir / "B02.tif") as src:
        src_width = src.width
        src_height = src.height
        src_crs = src.crs.to_string() if src.crs else "unknown"
        src_profile = src.profile.copy()
        
    n_rows = int(np.ceil((src_height - overlap) / stride)) if src_height > overlap else 1
    n_cols = int(np.ceil((src_width - overlap) / stride)) if src_width > overlap else 1
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_rows = []
    split_val = get_scene_split(item_id)
    
    tiles_info = []
    for r in range(n_rows):
        for c in range(n_cols):
            y_offset = r * stride
            x_offset = c * stride
            window = Window(x_offset, y_offset, tile_size, tile_size)
            tile_transform = rasterio.windows.transform(window, src_profile["transform"])
            tile_id = f"tile_r{r:03d}_c{c:03d}"
            tile_out_dir = out_dir / tile_id
            tile_out_dir.mkdir(exist_ok=True)
            
            tiles_info.append({
                "row": r,
                "col": c,
                "y_offset": y_offset,
                "x_offset": x_offset,
                "tile_transform": tile_transform,
                "tile_id": tile_id,
                "tile_out_dir": tile_out_dir
            })

    # 1. Process cloud_mask first to compute valid_fraction and create metadata
    with rasterio.open(in_dir / "cloud_mask.tif") as src:
        cloud_mask_full = src.read(1)
        cm_profile = src.profile.copy()
        
    for tile in tiles_info:
        r, c = tile["row"], tile["col"]
        x_offset, y_offset = tile["x_offset"], tile["y_offset"]
        
        # pad with 1s (clouds)
        cm_tile = np.ones((tile_size, tile_size), dtype=cloud_mask_full.dtype)
        
        scene_cols_in_tile = min(tile_size, max(0, src_width - x_offset))
        scene_rows_in_tile = min(tile_size, max(0, src_height - y_offset))
        
        if scene_rows_in_tile > 0 and scene_cols_in_tile > 0:
            cm_tile[:scene_rows_in_tile, :scene_cols_in_tile] = cloud_mask_full[
                y_offset:y_offset+scene_rows_in_tile,
                x_offset:x_offset+scene_cols_in_tile
            ]
            
        scene_pixel_count = scene_rows_in_tile * scene_cols_in_tile
        if scene_pixel_count == 0:
            cloud_fraction = float("nan")
            valid_fraction = float("nan")
        else:
            scene_mask = cm_tile[:scene_rows_in_tile, :scene_cols_in_tile]
            masked_scene_pixels = int(np.sum(scene_mask == 1))
            valid_scene_pixels = scene_pixel_count - masked_scene_pixels
            cloud_fraction = masked_scene_pixels / scene_pixel_count
            valid_fraction = valid_scene_pixels / scene_pixel_count
            
        out_profile = cm_profile.copy()
        out_profile.update(
            width=tile_size,
            height=tile_size,
            transform=tile["tile_transform"]
        )
        with rasterio.open(tile["tile_out_dir"] / "cloud_mask.tif", "w", **out_profile) as dst:
            dst.write(cm_tile, 1)
            
        bounds = rasterio.transform.array_bounds(tile_size, tile_size, tile["tile_transform"])
        tile_transform = tile["tile_transform"]
        
        meta = {
            "item_id": item_id,
            "tile_id": tile["tile_id"],
            "source_scene": str(in_dir),
            "row": r,
            "column": c,
            "window": [x_offset, y_offset, tile_size, tile_size],
            "width": tile_size,
            "height": tile_size,
            "crs": src_crs,
            "transform": [tile_transform.a, tile_transform.b, tile_transform.c,
                          tile_transform.d, tile_transform.e, tile_transform.f],
            "resolution": "10m",
            "band_order": BANDS,
            "tile_size": tile_size,
            "overlap": overlap,
            "scene_pixel_count": scene_pixel_count,
            "cloud_fraction": cloud_fraction,
            "valid_fraction": valid_fraction,
            "tiling_version": "v1.0"
        }
        with open(tile["tile_out_dir"] / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
            
        manifest_rows.append({
            "tile_id": tile["tile_id"],
            "item_id": item_id,
            "row": r,
            "column": c,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "width": tile_size,
            "height": tile_size,
            "minx": bounds[0],
            "miny": bounds[1],
            "maxx": bounds[2],
            "maxy": bounds[3],
            "cloud_fraction": cloud_fraction,
            "valid_fraction": valid_fraction,
            "crs": src_crs,
            "split": split_val
        })
        
    del cloud_mask_full # free memory before next bands
    
    # 2. Process other bands band-by-band
    files_to_tile = BANDS + ["SCL"]
    for f in files_to_tile:
        src_path = in_dir / f"{f}.tif"
        with rasterio.open(src_path) as src:
            full_data = src.read(1)
            profile = src.profile.copy()
            
        is_float = profile["dtype"] == rasterio.float32
        fill_val = np.nan if is_float else 0
        
        for tile in tiles_info:
            x_offset, y_offset = tile["x_offset"], tile["y_offset"]
            
            tile_data = np.full((tile_size, tile_size), fill_val, dtype=full_data.dtype)
            scene_cols_in_tile = min(tile_size, max(0, src_width - x_offset))
            scene_rows_in_tile = min(tile_size, max(0, src_height - y_offset))
            
            if scene_rows_in_tile > 0 and scene_cols_in_tile > 0:
                tile_data[:scene_rows_in_tile, :scene_cols_in_tile] = full_data[
                    y_offset:y_offset+scene_rows_in_tile,
                    x_offset:x_offset+scene_cols_in_tile
                ]
                
            out_profile = profile.copy()
            out_profile.update(
                width=tile_size,
                height=tile_size,
                transform=tile["tile_transform"]
            )
            with rasterio.open(tile["tile_out_dir"] / f"{f}.tif", "w", **out_profile) as dst:
                dst.write(tile_data, 1)
                
        del full_data
            
    manifest_rows.sort(key=lambda x: (x["row"], x["column"], x["tile_id"]))
    
    manifest_path = out_dir / "manifest.csv"
    fieldnames = [
        "tile_id", "item_id", "row", "column", "x_offset", "y_offset", 
        "width", "height", "minx", "miny", "maxx", "maxy", 
        "cloud_fraction", "valid_fraction", "crs", "split"
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

def cli_main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate deterministic LR tiles.")
    parser.add_argument("--item-id", required=True, help="STAC Item ID of prepared scene")
    parser.add_argument("--config", default="configs/tiling.yaml", help="Path to config")
    args = parser.parse_args()
    
    print(f"Tiling scene {args.item_id}...")
    create_tiles(args.item_id, config_path=args.config)
    print("Done.")

if __name__ == "__main__":
    cli_main()
