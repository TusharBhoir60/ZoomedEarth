"""
T2.8 - SEN2SR Tile Stitching + Geospatial Output Contract
"""
import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import rasterio
from rasterio.windows import Window
import time

from src.infer.run_sen2sr import scale_affine_transform
from src.infer.infer_tile import _get_git_sha, _get_software_versions

logger = logging.getLogger(__name__)

def create_feather_weight(height: int, width: int) -> np.ndarray:
    """Create a 2D separable linear feather weight (Bartlett-like window)."""
    y = np.minimum(np.arange(height) + 0.5, height - np.arange(height) - 0.5)
    x = np.minimum(np.arange(width) + 0.5, width - np.arange(width) - 0.5)
    w = np.outer(y, x)
    return w.astype(np.float32)

def stitch_scene(item_id: str,
                 processed_dir: str = "data/processed/sentinel2",
                 sr_output_dir: str = "data/outputs/sen2sr") -> Dict[str, Any]:
    
    source_scene_meta_path = Path(processed_dir) / item_id / "metadata.json"
    if not source_scene_meta_path.exists():
        raise FileNotFoundError(f"Source scene metadata not found: {source_scene_meta_path}")
        
    with open(source_scene_meta_path, "r") as f:
        source_meta = json.load(f)
        
    in_crs = source_meta["crs"]
    lr_transform = rasterio.Affine(*source_meta["transform"])
    lr_width = source_meta["width"]
    lr_height = source_meta["height"]
    
    sr_transform = scale_affine_transform(lr_transform, 4.0)
    sr_width = lr_width * 4
    sr_height = lr_height * 4
    
    # Locate all SR tiles
    sr_dir = Path(sr_output_dir) / item_id
    tile_paths = list(sr_dir.glob("tile_*/SEN2SR.tif"))
    if not tile_paths:
        raise FileNotFoundError(f"No SR tiles found in {sr_dir}")
        
    tiles_info = []
    sr_transform_inv = ~sr_transform
    for p in tile_paths:
        meta_path = p.parent / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path, "r") as f:
            tile_meta = json.load(f)
            
        out_tf = tile_meta["output_transform"]
        # Determine exact placement via geotransforms
        x_geo, y_geo = out_tf[2], out_tf[5]
        col_offset, row_offset = sr_transform_inv * (x_geo, y_geo)
        
        c = int(round(col_offset))
        r = int(round(row_offset))
        
        tiles_info.append({
            "path": p,
            "x_offset": c,
            "y_offset": r,
            "width": tile_meta["output_width"],
            "height": tile_meta["output_height"]
        })
        
    # Sort tiles top-to-bottom, left-to-right
    tiles_info.sort(key=lambda t: (t["y_offset"], t["x_offset"]))
    
    output_path = sr_dir / "SEN2SR_scene.tif"
    out_profile = {
        "driver": "GTiff",
        "height": sr_height,
        "width": sr_width,
        "count": 4,
        "dtype": "float32",
        "crs": in_crs,
        "transform": sr_transform,
        "tiled": True,
        "blockxsize": min(512, sr_width),
        "blockysize": min(512, sr_height),
        "compress": "deflate",
        "nodata": np.nan,
        "bigtiff": "YES", # Explicit BigTIFF for full scenes
    }
    
    buffer_capacity = 2048 # Holds ~4 tile rows simultaneously
    val_buffer = np.zeros((4, buffer_capacity, sr_width), dtype=np.float32)
    wt_buffer = np.zeros((4, buffer_capacity, sr_width), dtype=np.float32)
    buffer_start_y = 0
    
    feather_weights = {}
    
    t0 = time.perf_counter()
    
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.set_band_description(1, "B02 - Blue (2.5m SR)")
        dst.set_band_description(2, "B03 - Green (2.5m SR)")
        dst.set_band_description(3, "B04 - Red (2.5m SR)")
        dst.set_band_description(4, "B08 - NIR (2.5m SR)")

        def flush_buffer(end_y: int):
            nonlocal buffer_start_y
            write_h = end_y - buffer_start_y
            if write_h <= 0: return buffer_start_y
            
            # extract ready block
            val_out = val_buffer[:, :write_h, :]
            wt_out = wt_buffer[:, :write_h, :]
            
            # compute valid pixels
            mask = wt_out > 0
            final_out = np.full_like(val_out, np.nan)
            final_out[mask] = val_out[mask] / wt_out[mask]
            
            window = Window(0, buffer_start_y, sr_width, write_h)
            dst.write(final_out, window=window)
            
            # shift
            remaining = buffer_capacity - write_h
            val_buffer[:, :remaining, :] = val_buffer[:, write_h:, :]
            val_buffer[:, remaining:, :] = 0.0
            wt_buffer[:, :remaining, :] = wt_buffer[:, write_h:, :]
            wt_buffer[:, remaining:, :] = 0.0
            
            return end_y

        for t in tiles_info:
            r = t["y_offset"]
            c = t["x_offset"]
            w = t["width"]
            h = t["height"]
            
            # If the current tile exceeds our active buffer, flush the completed rows
            if r + h > buffer_start_y + buffer_capacity:
                buffer_start_y = flush_buffer(r)
                
            with rasterio.open(t["path"]) as src:
                data = src.read() # (4, H, W)
                
            shape_key = (h, w)
            if shape_key not in feather_weights:
                feather_weights[shape_key] = create_feather_weight(h, w)
            fw = feather_weights[shape_key]
            
            valid_mask = ~np.isnan(data)
            
            # Tile feather weights mapped perfectly to valid pixels
            wt_to_add = np.broadcast_to(fw[np.newaxis, ...], data.shape).copy()
            wt_to_add[~valid_mask] = 0.0
            
            val_to_add = data.copy()
            val_to_add[~valid_mask] = 0.0
            val_to_add *= wt_to_add
            
            buf_y = r - buffer_start_y
            
            # Clip bounds in case tiles exceed scene margins
            scene_w = min(w, sr_width - c)
            scene_h = min(h, sr_height - r)
            
            if scene_w > 0 and scene_h > 0:
                val_buffer[:, buf_y:buf_y+scene_h, c:c+scene_w] += val_to_add[:, :scene_h, :scene_w]
                wt_buffer[:, buf_y:buf_y+scene_h, c:c+scene_w] += wt_to_add[:, :scene_h, :scene_w]
            
        flush_buffer(sr_height)
        
    elapsed = time.perf_counter() - t0
    
    # Metadata
    scene_meta = {
        "item_id": item_id,
        "source_tile_directory": str(sr_dir),
        "source_scene": str(Path(processed_dir) / item_id),
        "band_order": ["B02", "B03", "B04", "B08"],
        "input_resolution": "10m",
        "output_resolution": "2.5m",
        "crs": in_crs,
        "width": sr_width,
        "height": sr_height,
        "transform": [sr_transform.a, sr_transform.b, sr_transform.c, 
                      sr_transform.d, sr_transform.e, sr_transform.f],
        "bounds": [
            sr_transform.c,
            sr_transform.f + sr_transform.e * sr_height,
            sr_transform.c + sr_transform.a * sr_width,
            sr_transform.f
        ],
        "tile_count": len(tiles_info),
        "tile_size": 512,
        "lr_overlap": 16,
        "sr_overlap": 64,
        "blend_method": "Separable linear feather (Bartlett)",
        "stitching_version": "v1.0",
        "git_sha": _get_git_sha(),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "software_versions": _get_software_versions(),
        "elapsed_seconds": round(elapsed, 2)
    }
    
    meta_path = sr_dir / "scene_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(scene_meta, f, indent=2)
        
    return scene_meta

def cli_main():
    parser = argparse.ArgumentParser(description="Stitch SEN2SR tiles into a full scene.")
    parser.add_argument("--item-id", required=True, help="Sentinel-2 Item ID")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Stitching scene {args.item_id}...")
    meta = stitch_scene(args.item_id)
    logger.info(f"Stitching complete. Output: {meta['width']}x{meta['height']} in {meta['elapsed_seconds']}s")

if __name__ == "__main__":
    cli_main()
