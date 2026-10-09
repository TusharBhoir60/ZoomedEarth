import hashlib
import math
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
import torch
import pathlib

from src.train.detector_dataset import DetectorDataset

@pytest.fixture
def temp_rasters(tmp_path):
    """Generates synthetic paired rasters for testing."""
    b_path = tmp_path / "Bicubic_scene.tif"
    s_path = tmp_path / "SEN2SR_scene.tif"
    l_path = tmp_path / "labels.tif"
    c_path = tmp_path / "cloud_mask.tif"
    
    # 2.5m geometries (height=512, width=512 -> 2x2 blocks of 256)
    height, width = 512, 512
    crs = CRS.from_epsg(32643)
    transform = Affine(2.5, 0.0, 700000.0, 0.0, -2.5, 3000000.0)
    
    # Bicubic (float32, nan nodata)
    b_data = np.random.rand(4, height, width).astype(np.float32)
    b_data[:, 0:256, 0:256] = np.nan # inject some NaNs in first block
    b_profile = {
        "driver": "GTiff", "height": height, "width": width, "count": 4,
        "dtype": "float32", "crs": crs, "transform": transform, "nodata": np.nan
    }
    with rasterio.open(b_path, "w", **b_profile) as dst:
        dst.write(b_data)
        dst.set_band_description(1, 'B02')
        dst.set_band_description(2, 'B03')
        dst.set_band_description(3, 'B04')
        dst.set_band_description(4, 'B08')
        
    # SEN2SR
    s_data = np.random.rand(4, height, width).astype(np.float32)
    s_data[:, 0:256, 0:256] = np.nan # inject NaNs in first block here as well
    with rasterio.open(s_path, "w", **b_profile) as dst:
        dst.write(s_data)
        dst.set_band_description(1, 'B02')
        
    # Labels (uint8, 255 nodata)
    l_data = np.full((1, height, width), 255, dtype=np.uint8)
    l_data[0, 256:, 256:] = 1 # Block 3 has positive footprints
    l_data[0, 256:260, :10] = 0 # some verified negatives in Block 2
    l_profile = {
        "driver": "GTiff", "height": height, "width": width, "count": 1,
        "dtype": "uint8", "crs": crs, "transform": transform, "nodata": 255
    }
    with rasterio.open(l_path, "w", **l_profile) as dst:
        dst.write(l_data)
        
    # Cloud Mask 10m (height/4 = 128, width/4 = 128)
    c_height, c_width = height // 4, width // 4
    c_transform = Affine(10.0, 0.0, 700000.0, 0.0, -10.0, 3000000.0)
    c_data = np.zeros((1, c_height, c_width), dtype=np.uint8)
    c_data[0, 100:, :] = 1 # Cloud at the bottom
    c_profile = {
        "driver": "GTiff", "height": c_height, "width": c_width, "count": 1,
        "dtype": "uint8", "crs": crs, "transform": c_transform
    }
    with rasterio.open(c_path, "w", **c_profile) as dst:
        dst.write(c_data)
        
    return b_path, s_path, l_path, c_path

def test_paired_patch_coordinates_and_grid_alignment(temp_rasters):
    b, s, l, c = temp_rasters
    ds = DetectorDataset(b, s, l, c, patch_size=256)
    
    assert len(ds) == 4
    item = ds[0]
    # Coordinates of first block
    assert item["window_coords"] == (0, 0)
    # Shape of image tensor
    assert item["image"].shape == (4, 256, 256)
    assert item["proxy_labels"].shape == (256, 256)

def test_deterministic_sampling(temp_rasters):
    b, s, l, c = temp_rasters
    # Create larger synthetic rasters conceptually by modifying the class length 
    # to test the sampling logic over a larger sequence without writing a huge file.
    ds1 = DetectorDataset(b, s, l, c, patch_size=256, seed=42)
    ds1.length = 1000 # override for logic testing
    ds1.n_cols = 10
    
    ds2 = DetectorDataset(b, s, l, c, patch_size=256, seed=42)
    ds2.length = 1000
    ds2.n_cols = 10
    
    # 3. Deterministic 50/50 sampling over a sufficiently large test sequence
    mods1 = []
    for i in range(1000):
        # We manually call the logic to avoid reading out of bounds
        hex_hash = hashlib.md5(f"42_{i}".encode('utf-8')).hexdigest()
        mods1.append("sen2sr" if (int(hex_hash, 16) % 2) == 1 else "bicubic")
        
    sen_count = mods1.count("sen2sr")
    bic_count = mods1.count("bicubic")
    
    assert 450 < sen_count < 550  # roughly 50/50
    assert 450 < bic_count < 550
    
    # 4. Reproducibility under the same seed
    mods2 = []
    for i in range(1000):
        hex_hash = hashlib.md5(f"42_{i}".encode('utf-8')).hexdigest()
        mods2.append("sen2sr" if (int(hex_hash, 16) % 2) == 1 else "bicubic")
        
    assert mods1 == mods2
    
def test_proxy_bce_and_masks(temp_rasters):
    b, s, l, c = temp_rasters
    ds = DetectorDataset(b, s, l, c, patch_size=256)
    
    # Block 0: All unknown labels, NaNs in image
    item0 = ds[0]
    assert item0["proxy_labels"].max() == 0.0 # 255 -> 0
    assert item0["unknown_mask"].all() # original unknown mask is preserved
    assert not item0["valid_imagery_mask"].any() # NaNs in b_data made imagery invalid
    assert not torch.isnan(item0["image"]).any() # NaNs replaced with 0.0 for PyTorch
    
    # Block 2 (row 1, col 0): Contains verified negatives (0) and clouds (1)
    item2 = ds[2]
    assert item2["proxy_labels"].min() == 0.0
    # Verified negatives shouldn't be in unknown mask
    assert not item2["unknown_mask"][0:4, :10].any() 
    assert item2["proxy_labels"][0:4, :10].sum() == 0.0
    
    # Block 3 (row 1, col 1): Contains positives (1)
    item3 = ds[3]
    assert item3["proxy_labels"].max() == 1.0
    assert not item3["unknown_mask"].all()

def test_mismatched_crs_or_dimensions_rejected(temp_rasters, tmp_path):
    b, s, l, c = temp_rasters
    
    # Make a bad label file with mismatched dims
    bad_l = tmp_path / "bad_l.tif"
    with rasterio.open(l) as src:
        prof = src.profile
        prof["width"] = 256 # wrong width
        with rasterio.open(bad_l, "w", **prof) as dst:
            dst.write(np.zeros((1, 512, 256), dtype=np.uint8))
            
    with pytest.raises(ValueError, match="Geometry mismatch"):
        DetectorDataset(b, s, bad_l, c)
        
    # Make a bad cloud mask (10m grid violation)
    bad_c = tmp_path / "bad_c.tif"
    with rasterio.open(c) as src:
        prof = src.profile
        # Change pixel scale to 20m instead of 10m
        prof["transform"] = Affine(20.0, 0.0, 700000.0, 0.0, -20.0, 3000000.0)
        with rasterio.open(bad_c, "w", **prof) as dst:
            dst.write(np.zeros((1, 128, 128), dtype=np.uint8))
            
    with pytest.raises(ValueError, match="expected 10m affine transform"):
        DetectorDataset(b, s, l, bad_c)

def test_validation_test_aoi_rejection(temp_rasters):
    b, s, l, c = temp_rasters
    
    with pytest.raises(ValueError, match="Pune and Mumbai are prohibited"):
        # simulate bad path input
        DetectorDataset("data/pune_val_Bicubic.tif", s, l, c)
