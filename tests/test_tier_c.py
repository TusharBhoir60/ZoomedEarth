import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
import json
from src.eval.tier_c import score_tier_c, TierCEvalError

def setup_files(tmp_path, preds, labels, cmask, budget):
    p_path = tmp_path / "pred.tif"
    l_path = tmp_path / "label.tif"
    c_path = tmp_path / "cmask.tif"
    b_path = tmp_path / "budget.json"
    
    p_transform = from_origin(0.0, 100.0, 2.5, 2.5)
    c_transform = from_origin(0.0, 100.0, 10.0, 10.0)
    
    with rasterio.open(p_path, 'w', driver='GTiff', height=preds.shape[0], width=preds.shape[1], count=1, dtype='float32', crs='EPSG:32643', transform=p_transform) as dst:
        dst.write(preds.astype('float32'), 1)
        
    with rasterio.open(l_path, 'w', driver='GTiff', height=labels.shape[0], width=labels.shape[1], count=1, dtype='uint8', crs='EPSG:32643', transform=p_transform) as dst:
        dst.write(labels.astype('uint8'), 1)
        
    with rasterio.open(c_path, 'w', driver='GTiff', height=cmask.shape[0], width=cmask.shape[1], count=1, dtype='uint8', crs='EPSG:32643', transform=c_transform) as dst:
        dst.write(cmask.astype('uint8'), 1)
        
    with open(b_path, 'w') as f:
        json.dump({"arithmetic_mean_budget": budget}, f)
        
    return str(p_path), str(l_path), str(c_path), str(b_path)

def test_tier_c_selection_and_tie_breaking(tmp_path):
    # 4x4 array = 16 pixels. 2.5m.
    preds = np.full((4, 4), 0.1)
    preds[0, 0] = 0.9
    preds[0, 1] = 0.5
    preds[0, 2] = 0.5
    preds[1, 0] = 0.2
    preds[1, 1] = 0.5
    
    labels = np.full((4, 4), 255)
    labels[0, 0] = 1
    labels[0, 2] = 1
    labels[1, 1] = 1
    
    # 1x1 array = 1 pixel at 10m. (Covers exactly the 4x4 2.5m area)
    cmask = np.zeros((1, 1))
    
    # Budget = 0.1875 (3 / 16 pixels should be selected)
    # Top 3 scores: 0.9 (0,0), 0.5 (0,1), 0.5 (0,2). 
    # 0.5 at (1,1) is left out due to row-major tie-breaking.
    
    p, l, c, b = setup_files(tmp_path, preds, labels, cmask, 0.1875)
    result = score_tier_c(p, l, c, b, "test_aoi")
    
    assert result["eligible_pool_pixels"] == 16
    assert result["selected_pixels"] == 3
    # Top 3 pixels: (0,0)->1, (0,1)->255, (0,2)->1. Total TP = 2.
    assert result["true_positives_recovered"] == 2
    # Total positive labels = 3
    assert result["total_positive_labels"] == 3
    assert result["recall"] == 2.0 / 3.0

def test_tier_c_cloud_mask_and_nodata_exclusion(tmp_path):
    # 4x8 array
    preds = np.full((4, 8), 0.7)
    preds[0, 0] = 0.9
    preds[0, 1] = 0.8
    preds[3, 0] = np.nan # invalid
    
    labels = np.ones((4, 8))
    
    # 1x2 array (10m)
    cmask = np.array([[0, 1]]) # right half is cloudy
    # right half covers cols 4..7.
    # Total px = 32. 
    # Cloud covers 16 px (right half).
    # valid_mask excludes 16 cloudy + 1 nan = 17 excluded. N = 15.
    # Budget = 0.1, round(0.1 * 15) = 2.
    
    p, l, c, b = setup_files(tmp_path, preds, labels, cmask, 0.1)
    result = score_tier_c(p, l, c, b, "test_aoi")
    
    assert result["eligible_pool_pixels"] == 15
    assert result["selected_pixels"] == 2
    
def test_tier_c_invalid_inputs(tmp_path):
    p, l, c, b = setup_files(tmp_path, np.ones((4,4)), np.ones((4,4)), np.ones((1,1)), -0.1)
    with pytest.raises(TierCEvalError, match="must be between 0 and 1"):
        score_tier_c(p, l, c, b, "test_aoi")
        
    p, l, c, b = setup_files(tmp_path, np.ones((4,4)), np.ones((4,4)), np.ones((1,1)), 0.5)
    with pytest.raises(TierCEvalError, match="Eligible candidate pool is empty"):
        score_tier_c(p, l, c, b, "test_aoi")

def test_tier_c_hash_mismatch(tmp_path):
    p_path, l_path, c_path, b_path = setup_files(tmp_path, np.ones((4,4)), np.ones((4,4)), np.zeros((1,1)), 0.5)
    
    with open(b_path, 'w') as f:
        json.dump({
            "arithmetic_mean_budget": 0.5,
            "ratios": {
                "hash_test_aoi": {
                    "input_hashes": {
                        "eval_labels_tif": "invalid_hash"
                    }
                }
            }
        }, f)
        
    with pytest.raises(TierCEvalError, match="Hash mismatch: Evaluation labels raster does not match"):
        score_tier_c(p_path, l_path, c_path, b_path, "hash_test_aoi")

def test_tier_c_zero_positives(tmp_path):
    preds = np.ones((4,4))
    labels = np.full((4,4), 255)
    cmask = np.zeros((1,1))
    
    p, l, c, b = setup_files(tmp_path, preds, labels, cmask, 0.5)
    with pytest.raises(TierCEvalError, match="Zero positive labels"):
        score_tier_c(p, l, c, b, "test_aoi")

def test_tier_c_grid_mismatch(tmp_path):
    p, l, c, b = setup_files(tmp_path, np.ones((4,4)), np.ones((4,4)), np.zeros((1,1)), 0.5)
    
    mismatched_p = tmp_path / "mismatched_pred.tif"
    transform = from_origin(0.0, 100.0, 5.0, 5.0) 
    with rasterio.open(mismatched_p, 'w', driver='GTiff', height=4, width=4, count=1, dtype='float32', crs='EPSG:32643', transform=transform) as dst:
        dst.write(np.ones((4,4), dtype='float32'), 1)
        
    with pytest.raises(TierCEvalError, match="Prediction raster does not perfectly match label raster grid."):
        score_tier_c(str(mismatched_p), l, c, b, "test_aoi")

def test_tier_c_cloud_mask_mismatch(tmp_path):
    p, l, c, b = setup_files(tmp_path, np.ones((4,4)), np.ones((4,4)), np.zeros((1,1)), 0.5)
    
    # 1. Wrong CRS
    mismatched_c = tmp_path / "mismatched_cmask_crs.tif"
    transform = from_origin(0.0, 100.0, 10.0, 10.0) 
    with rasterio.open(mismatched_c, 'w', driver='GTiff', height=1, width=1, count=1, dtype='uint8', crs='EPSG:4326', transform=transform) as dst:
        dst.write(np.zeros((1,1), dtype='uint8'), 1)
        
    with pytest.raises(TierCEvalError, match="Cloud mask raster does not perfectly match prediction raster bounds and CRS"):
        score_tier_c(p, l, str(mismatched_c), b, "test_aoi")
        
    # 2. Wrong pixel resolution (e.g. 5m instead of 10m)
    mismatched_c2 = tmp_path / "mismatched_cmask_res.tif"
    transform2 = from_origin(0.0, 100.0, 5.0, 5.0)
    with rasterio.open(mismatched_c2, 'w', driver='GTiff', height=2, width=2, count=1, dtype='uint8', crs='EPSG:32643', transform=transform2) as dst:
        dst.write(np.zeros((2,2), dtype='uint8'), 1)
        
    with pytest.raises(TierCEvalError, match="Cloud mask raster does not have the expected 10m affine transform or orientation"):
        score_tier_c(p, l, str(mismatched_c2), b, "test_aoi")
        
    # 3. Origin shift (bounds mismatch will trigger first)
    mismatched_c3 = tmp_path / "mismatched_cmask_origin.tif"
    transform3 = from_origin(10.0, 100.0, 10.0, 10.0) # shifted by 10m
    with rasterio.open(mismatched_c3, 'w', driver='GTiff', height=1, width=1, count=1, dtype='uint8', crs='EPSG:32643', transform=transform3) as dst:
        dst.write(np.zeros((1,1), dtype='uint8'), 1)
        
    with pytest.raises(TierCEvalError, match="Cloud mask raster does not perfectly match prediction raster bounds and CRS"):
        score_tier_c(p, l, str(mismatched_c3), b, "test_aoi")
        
    # 4. Rotation/skew mismatch
    mismatched_c4 = tmp_path / "mismatched_cmask_skew.tif"
    from rasterio import Affine
    # a, b, c, d, e, f -> 10.0, 1.0, 0.0, 1.0, -10.0, 100.0
    transform4 = Affine(10.0, 1.0, 0.0, 1.0, -10.0, 100.0)
    with rasterio.open(mismatched_c4, 'w', driver='GTiff', height=1, width=1, count=1, dtype='uint8', crs='EPSG:32643', transform=transform4) as dst:
        dst.write(np.zeros((1,1), dtype='uint8'), 1)
        
    with pytest.raises(TierCEvalError, match="Cloud mask raster does not perfectly match prediction raster bounds and CRS"):
        # Bounds will likely mismatch due to skew, but if they didn't, transform check would catch it.
        score_tier_c(p, l, str(mismatched_c4), b, "test_aoi")
        
    # 5. Dimensions/Bounds mismatch (e.g. right origin and resolution, but shape is 2x1 instead of 1x1)
    mismatched_c5 = tmp_path / "mismatched_cmask_dim.tif"
    transform5 = from_origin(0.0, 100.0, 10.0, 10.0) 
    with rasterio.open(mismatched_c5, 'w', driver='GTiff', height=2, width=1, count=1, dtype='uint8', crs='EPSG:32643', transform=transform5) as dst:
        dst.write(np.zeros((2,1), dtype='uint8'), 1)
        
    with pytest.raises(TierCEvalError, match="Cloud mask raster does not perfectly match prediction raster bounds and CRS"):
        score_tier_c(p, l, str(mismatched_c5), b, "test_aoi")
