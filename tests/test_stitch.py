import pytest
import numpy as np
import rasterio
from rasterio.transform import Affine
from src.infer.stitch import create_feather_weight, stitch_scene
import json
import shutil
from pathlib import Path

def test_feather_weights():
    w = create_feather_weight(4, 4)
    # [0.5, 1.5, 1.5, 0.5] outer [0.5, 1.5, 1.5, 0.5]
    assert w.shape == (4, 4)
    assert np.allclose(w[0, :], [0.25, 0.75, 0.75, 0.25])
    assert np.allclose(w[1, :], [0.75, 2.25, 2.25, 0.75])

def test_stitch_scene_synthetic(tmp_path):
    item_id = "test_item"
    processed_dir = tmp_path / "processed"
    sr_output_dir = tmp_path / "outputs"
    
    # Setup mock processed metadata
    proc_scene_dir = processed_dir / item_id
    proc_scene_dir.mkdir(parents=True)
    with open(proc_scene_dir / "metadata.json", "w") as f:
        json.dump({
            "crs": "EPSG:32643",
            "transform": [10.0, 0.0, 700000.0, 0.0, -10.0, 3100000.0],
            "width": 256,
            "height": 256
        }, f)
        
    # Setup mock SR tiles
    sr_dir = sr_output_dir / item_id
    
    tf = Affine(2.5, 0.0, 700000.0, 0.0, -2.5, 3100000.0)
    
    def create_mock_tile(tile_id, r, c, val):
        t_dir = sr_dir / tile_id
        t_dir.mkdir(parents=True)
        # 512x512 tile
        data = np.full((4, 512, 512), val, dtype=np.float32)
        
        # Tile 1: Introduce NaN padding
        if val == 1.0:
            data[:, :10, :] = np.nan # top padding
            
        tile_tf = tf * Affine.translation(c, r)
        profile = {
            "driver": "GTiff", "height": 512, "width": 512, "count": 4,
            "dtype": "float32", "crs": "EPSG:32643", "transform": tile_tf,
            "nodata": np.nan
        }
        with rasterio.open(t_dir / "SEN2SR.tif", "w", **profile) as dst:
            dst.write(data)
            dst.set_band_description(1, "B02")
            dst.set_band_description(2, "B03")
            dst.set_band_description(3, "B04")
            dst.set_band_description(4, "B08")
            
        with open(t_dir / "metadata.json", "w") as f:
            json.dump({
                "output_transform": [tile_tf.a, tile_tf.b, tile_tf.c, tile_tf.d, tile_tf.e, tile_tf.f],
                "output_width": 512,
                "output_height": 512,
                "input_band_order": ["B02", "B03", "B04", "B08"],
                "model_band_order": ["B04", "B03", "B02", "B08"],
                "output_band_order": ["B02", "B03", "B04", "B08"],
                "band_order_contract_version": "v1.0"
            }, f)

    # Tile 1: top-left (r=0, c=0)
    create_mock_tile("tile_1", 0, 0, 1.0)
    # Tile 2: right (r=0, c=448) -> overlap of 64
    create_mock_tile("tile_2", 0, 448, 2.0)
    
    # 1. Determinism check: run twice, must be identical
    meta1 = stitch_scene(item_id, processed_dir=str(processed_dir), sr_output_dir=str(sr_output_dir))
    
    with rasterio.open(sr_dir / "SEN2SR_scene.tif") as src:
        out_data1 = src.read()
    
    # Move and run again
    shutil.move(sr_dir / "SEN2SR_scene.tif", sr_dir / "SEN2SR_scene_1.tif")
    meta2 = stitch_scene(item_id, processed_dir=str(processed_dir), sr_output_dir=str(sr_output_dir))
    
    with rasterio.open(sr_dir / "SEN2SR_scene.tif") as src:
        out_data2 = src.read()
        
    assert np.all((out_data1 == out_data2) | (np.isnan(out_data1) & np.isnan(out_data2)))
        
    meta = meta2
    out_data = out_data2
    
    # 2. Check metadata dimensions and CRS
    assert meta["width"] == 1024 # 256 * 4
    assert meta["height"] == 1024
    assert meta["tile_count"] == 2
    assert meta["crs"] == "EPSG:32643"
    assert meta["output_resolution"] == "2.5m"
    
    with rasterio.open(sr_dir / "SEN2SR_scene.tif") as src:
        assert src.shape == (1024, 1024)
        assert src.count == 4
        assert src.transform == tf
        assert src.crs.to_string() == "EPSG:32643"
        assert src.bounds.left == 700000.0
        
        # 3. Band Preservation
        assert src.descriptions == ('B02', 'B03', 'B04', 'B08')
        
        # 4. NaN handling and padding
        # Check non-overlap region of tile 1 (x=0 to 447)
        # Note: top 10 rows were NaN in tile 1, so they should be NaN in output
        assert np.all(np.isnan(out_data[:, :10, :448]))
        assert np.allclose(out_data[:, 10:512, :448], 1.0)
        
        # Check non-overlap region of tile 2 (x=512 to 960)
        # It covers y=0..512, but tile 1 was only x=0..512, so tile 2's top-left corner x=448..511, y=0..10
        # Wait, for y < 10, tile 1 is NaN. Tile 2 is NOT NaN.
        # Thus, in the overlap region y=0..10, x=448..511, tile 2 provides valid data, tile 1 is NaN.
        # The output should NOT be NaN. It should equal tile 2's value exactly!
        # "valid coverage remains valid"
        overlap_top = out_data[:, :10, 448:512]
        assert np.allclose(overlap_top, 2.0)
        
        # Check pure non-overlap of tile 2
        assert np.allclose(out_data[:, 0:512, 512:960], 2.0)
        
        # 5. Overlap blending
        # For y >= 10, x=448 to 511, both tiles have valid data.
        # Should be a blend of 1.0 and 2.0
        blend = out_data[0, 10:512, 448:512]
        assert np.all((blend > 1.0) & (blend < 2.0))
        
        # 6. Uncovered regions (y >= 512 or x >= 960) should be NaN
        assert np.all(np.isnan(out_data[:, 512:, :]))
        assert np.all(np.isnan(out_data[:, :, 960:]))
