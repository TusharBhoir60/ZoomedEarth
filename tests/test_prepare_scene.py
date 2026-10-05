import os
import json
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path

from src.ingest.prepare_scene import (
    MASK_CLASSES,
    validate_and_read_grids,
    prepare_scene,
    get_product_metadata
)

@pytest.fixture
def mock_cache_dir(tmp_path):
    cache_dir = tmp_path / "data" / "raw" / "sentinel2"
    item_id = "test_item"
    item_dir = cache_dir / item_id
    item_dir.mkdir(parents=True)
    
    meta = {
        "item_id": item_id,
        "legacy_boa_add_offset": -1000.0,
        "quantification_value": 10000.0,
        "earthsearch:boa_offset_applied": True,
        "s2:processing_baseline": "05.09"
    }
    with open(item_dir / "metadata.json", "w") as f:
        json.dump(meta, f)
        
    transform = from_origin(100.0, 100.0, 10.0, 10.0)
    profile = {
        "driver": "GTiff",
        "height": 10,
        "width": 10,
        "count": 1,
        "dtype": rasterio.uint16,
        "crs": "EPSG:32643",
        "transform": transform
    }
    
    for band in ["B02", "B03", "B04", "B08"]:
        data = np.ones((10, 10), dtype=np.uint16) * 1500
        data[0, 0] = 0 # NoData DN
        with rasterio.open(item_dir / f"{band}.tif", "w", **profile) as dst:
            dst.write(data, 1)
            
    scl_profile = profile.copy()
    scl_profile["dtype"] = rasterio.uint8
    scl_data = np.full((10, 10), 4, dtype=np.uint8) # 4=Vegetation (Valid)
    scl_data[0, 1] = 0 # NoData -> Masked
    scl_data[0, 2] = 9 # Cloud High -> Masked
    scl_data[0, 3] = 3 # Cloud Shadow -> Masked
    scl_data[0, 4] = 10 # Cirrus -> Masked
    scl_data[0, 5] = 1 # Defective -> Masked
    scl_data[0, 6] = 8 # Cloud Medium -> Masked
    scl_data[0, 7] = 11 # Snow -> Masked
    
    with rasterio.open(item_dir / "SCL.tif", "w", **scl_profile) as dst:
        dst.write(scl_data, 1)
        
    return str(cache_dir), item_id

def test_get_product_metadata_missing(mock_cache_dir):
    cache_dir, item_id = mock_cache_dir
    meta_path = Path(cache_dir) / item_id / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump({"item_id": item_id}, f)
        
    with pytest.raises(ValueError, match="Missing required STAC fields"):
        get_product_metadata(item_id, cache_dir)

@pytest.mark.parametrize("missing_band", ["B02", "B03", "B04", "B08", "SCL"])
def test_validate_and_read_grids_missing(mock_cache_dir, missing_band):
    cache_dir, item_id = mock_cache_dir
    os.remove(Path(cache_dir) / item_id / f"{missing_band}.tif")
    with pytest.raises(FileNotFoundError, match=f"Missing required asset {missing_band}"):
        validate_and_read_grids(item_id, cache_dir)

def test_validate_and_read_grids_incompatible_transform(mock_cache_dir):
    cache_dir, item_id = mock_cache_dir
    with rasterio.open(Path(cache_dir) / item_id / "B03.tif", "r+") as src:
        src.transform = from_origin(101.0, 100.0, 10.0, 10.0)
        
    with pytest.raises(ValueError, match="transform does not match reference"):
        validate_and_read_grids(item_id, cache_dir)

def test_validate_and_read_grids_incompatible_crs(mock_cache_dir):
    cache_dir, item_id = mock_cache_dir
    with rasterio.open(Path(cache_dir) / item_id / "B03.tif") as src:
        prof = src.profile.copy()
    prof.update(crs="EPSG:4326")
    with rasterio.open(Path(cache_dir) / item_id / "B03.tif", "w", **prof) as dst:
        dst.write(np.zeros((10, 10), dtype=np.uint16), 1)
        
    with pytest.raises(ValueError, match="CRS EPSG:4326 does not match reference CRS"):
        validate_and_read_grids(item_id, cache_dir)

def test_validate_and_read_grids_incompatible_dimensions(mock_cache_dir):
    cache_dir, item_id = mock_cache_dir
    with rasterio.open(Path(cache_dir) / item_id / "B04.tif") as src:
        prof = src.profile.copy()
    prof.update(width=9)
    with rasterio.open(Path(cache_dir) / item_id / "B04.tif", "w", **prof) as dst:
        dst.write(np.zeros((10, 9), dtype=np.uint16), 1)
        
    with pytest.raises(ValueError, match="dimensions do not match reference"):
        validate_and_read_grids(item_id, cache_dir)

def test_prepare_scene(mock_cache_dir, tmp_path):
    cache_dir, item_id = mock_cache_dir
    processed_dir = tmp_path / "processed"
    
    # Store mtimes of raw files to check they remain unmodified
    raw_files = list((Path(cache_dir) / item_id).glob("*"))
    mtimes_before = {f: f.stat().st_mtime for f in raw_files}
    
    prepare_scene(item_id, raw_cache_dir=cache_dir, processed_dir=str(processed_dir))
    
    # Verify raw files are exactly the same
    for f in raw_files:
        assert f.exists()
        assert f.stat().st_mtime == mtimes_before[f]
    
    out_dir = processed_dir / item_id
    assert out_dir.exists()
    
    for band in ["B02", "B03", "B04", "B08", "SCL", "cloud_mask"]:
        assert (out_dir / f"{band}.tif").exists()
        
    assert (out_dir / "metadata.json").exists()
    
    with rasterio.open(out_dir / "B02.tif") as src:
        b02 = src.read(1)
        # Check explicit masks
        assert np.isnan(b02[0, 0])  # DN=0
        assert np.isnan(b02[0, 1])  # SCL=0
        assert np.isnan(b02[0, 2])  # SCL=9
        assert np.isnan(b02[0, 3])  # SCL=3
        assert np.isnan(b02[0, 4])  # SCL=10
        assert np.isnan(b02[0, 5])  # SCL=1
        assert np.isnan(b02[0, 6])  # SCL=8
        assert np.isnan(b02[0, 7])  # SCL=11
        
        # Check valid pixel conversion (1500 / 10000 = 0.15, offset is 0 because applied=True)
        assert np.isclose(b02[1, 1], 0.15)
        
        # Check geotransform preserved
        assert src.transform.a == 10.0
        assert src.crs.to_string() == "EPSG:32643"
        assert src.width == 10
        assert src.height == 10
        
    with rasterio.open(out_dir / "cloud_mask.tif") as src:
        cm = src.read(1)
        # SCL masked exactly 7 pixels in the first row
        for j in range(1, 8):
            assert cm[0, j] == 1
        assert cm[1, 1] == 0
        
    with open(out_dir / "metadata.json") as f:
        meta = json.load(f)
        assert meta["resolution"] == "10m"
        assert meta["band_order"] == ["B02", "B03", "B04", "B08"]
        
        assert meta["masked_pixel_count"] == 7
        assert meta["valid_pixel_count"] == 93
        assert meta["mask_fraction"] == 0.07

import unittest.mock
import src.ingest.prepare_scene

def test_prepare_scene_calls_dn_to_reflectance(mock_cache_dir, tmp_path):
    cache_dir, item_id = mock_cache_dir
    processed_dir = tmp_path / "processed"
    
    with unittest.mock.patch("src.ingest.prepare_scene.dn_to_reflectance", wraps=src.ingest.prepare_scene.dn_to_reflectance) as mock_dn:
        prepare_scene(item_id, raw_cache_dir=cache_dir, processed_dir=str(processed_dir))
        
    assert mock_dn.call_count == 4

def test_prepare_scene_guard_failure(mock_cache_dir, tmp_path):
    cache_dir, item_id = mock_cache_dir
    
    # Force a failure: earthsearch:boa_offset_applied=False, pb=05.09 -> -1000.0 offset
    # Data DN = 200 -> (200 - 1000) / 10000 = -0.08 -> 100% negative
    for band in ["B02", "B03", "B04", "B08"]:
        with rasterio.open(Path(cache_dir) / item_id / f"{band}.tif", "r+") as src:
            data = src.read(1)
            data[data > 0] = 200
            src.write(data, 1)
            
    meta_path = Path(cache_dir) / item_id / "metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    meta["earthsearch:boa_offset_applied"] = False
    with open(meta_path, "w") as f:
        json.dump(meta, f)
        
    processed_dir = tmp_path / "processed"
    with pytest.raises(ValueError, match="Reflectance Guard Failed"):
        prepare_scene(item_id, raw_cache_dir=cache_dir, processed_dir=str(processed_dir))
        
    # Ensure no partial directory left
    assert not (processed_dir / item_id).exists()
    assert not (processed_dir / f"{item_id}.tmp").exists()

def test_prepare_scene_legacy_offset_ignored(mock_cache_dir, tmp_path):
    """Legacy boa_add_offset=-1000 is present but applied=True -> effective offset must be 0."""
    cache_dir, item_id = mock_cache_dir
    processed_dir = tmp_path / "processed"
    
    # Metadata has legacy_boa_add_offset=-1000 and applied=True (set in fixture)
    prepare_scene(item_id, raw_cache_dir=cache_dir, processed_dir=str(processed_dir))
    
    out_dir = processed_dir / item_id
    with open(out_dir / "metadata.json") as f:
        meta = json.load(f)
    
    rc = meta["reflectance_conversion"]
    assert rc["legacy_boa_add_offset"] == -1000.0
    assert rc["effective_boa_add_offset"] == 0.0
    assert rc["rule_id"] == "applied_true"
    
    # Valid pixel value: DN=1500, offset=0, quant=10000 -> 0.15
    with rasterio.open(out_dir / "B02.tif") as src:
        b02 = src.read(1)
        assert np.isclose(b02[1, 1], 0.15)

def test_prepare_scene_correct_data_passes(mock_cache_dir, tmp_path):
    """Flag True with corrected data should pass the guard cleanly."""
    cache_dir, item_id = mock_cache_dir
    processed_dir = tmp_path / "processed"
    
    prepare_scene(item_id, raw_cache_dir=cache_dir, processed_dir=str(processed_dir))
    
    out_dir = processed_dir / item_id
    assert out_dir.exists()
    with open(out_dir / "metadata.json") as f:
        meta = json.load(f)
    
    # All negative fractions should be 0
    for band, frac in meta["reflectance_conversion"]["negative_fraction_by_band"].items():
        assert frac == 0.0, f"{band} has unexpected negative fraction {frac}"

def test_prepare_scene_metadata_fields(mock_cache_dir, tmp_path):
    """Verify all required reflectance_conversion metadata fields are present."""
    cache_dir, item_id = mock_cache_dir
    processed_dir = tmp_path / "processed"
    
    prepare_scene(item_id, raw_cache_dir=cache_dir, processed_dir=str(processed_dir))
    
    with open(processed_dir / item_id / "metadata.json") as f:
        meta = json.load(f)
    
    assert meta["preparation_version"] == "v1.1"
    rc = meta["reflectance_conversion"]
    assert "effective_boa_add_offset" in rc
    assert "rule_id" in rc
    assert "boa_offset_applied" in rc
    assert "processing_baseline" in rc
    assert "quantification_value" in rc
    assert "negative_fraction_by_band" in rc
    assert "p01_by_band" in rc
    assert "guard_threshold" in rc
    assert rc["guard_threshold"] == 0.05
