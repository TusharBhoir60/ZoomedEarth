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
        src_transform = src.transform
        
    n_rows = int(np.ceil((src_height - overlap) / stride)) if src_height > overlap else 1
    n_cols = int(np.ceil((src_width - overlap) / stride)) if src_width > overlap else 1
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_rows = []
    split_val = get_scene_split(item_id)
    
    for r in range(n_rows):
        for c in range(n_cols):
            y_offset = r * stride
            x_offset = c * stride
            
            window = Window(x_offset, y_offset, tile_size, tile_size)
            
            tile_id = f"tile_r{r:03d}_c{c:03d}"
            tile_out_dir = out_dir / tile_id
            tile_out_dir.mkdir(exist_ok=True)
            
            cloud_mask_data = None
            tile_transform = None
            
            files_to_tile = BANDS + ["SCL", "cloud_mask"]
            for f in files_to_tile:
                src_path = in_dir / f"{f}.tif"
                with rasterio.open(src_path) as src:
                    if src.dtypes[0] == rasterio.float32:
                        fill_val = np.nan
                    else:
                        fill_val = 1 if f == "cloud_mask" else 0
                        
                    data = src.read(1, window=window, boundless=True, fill_value=fill_val)
                    
                    if tile_transform is None:
                        tile_transform = src.window_transform(window)
                        
                    out_profile = src.profile.copy()
                    out_profile.update(
                        width=tile_size,
                        height=tile_size,
                        transform=tile_transform
                    )
                    
                    if f == "cloud_mask":
                        cloud_mask_data = data
                        
                    with rasterio.open(tile_out_dir / f"{f}.tif", "w", **out_profile) as dst:
                        dst.write(data, 1)
                        
            # Compute cloud/valid fractions only over genuine source-scene pixels.
            # Pixels whose (row, col) in the tile window fall outside the source
            # raster extent are padding and must not count as observations.
            scene_cols_in_tile = min(tile_size, max(0, src_width - x_offset))
            scene_rows_in_tile = min(tile_size, max(0, src_height - y_offset))
            scene_pixel_count = scene_rows_in_tile * scene_cols_in_tile
            
            if scene_pixel_count == 0:
                cloud_fraction = float("nan")
                valid_fraction = float("nan")
            else:
                scene_mask = cloud_mask_data[:scene_rows_in_tile, :scene_cols_in_tile]
                masked_scene_pixels = int(np.sum(scene_mask == 1))
                valid_scene_pixels = scene_pixel_count - masked_scene_pixels
                cloud_fraction = masked_scene_pixels / scene_pixel_count
                valid_fraction = valid_scene_pixels / scene_pixel_count
            
            bounds = rasterio.transform.array_bounds(tile_size, tile_size, tile_transform)
            
            meta = {
                "item_id": item_id,
                "tile_id": tile_id,
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
            with open(tile_out_dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)
                
            manifest_rows.append({
                "tile_id": tile_id,
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
