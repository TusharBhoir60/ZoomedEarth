import json
import rasterio
from rasterio.enums import Resampling
import numpy as np
from pathlib import Path
import hashlib

class TierCEvalError(Exception):
    pass

def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def load_budget_and_hashes(budget_path: str, aoi_id: str) -> tuple[float, dict]:
    try:
        with open(budget_path, 'r') as f:
            data = json.load(f)
            budget = float(data['arithmetic_mean_budget'])
            if not (0.0 < budget < 1.0):
                raise TierCEvalError(f"Budget must be between 0 and 1, got {budget}")
                
            # Extract hashes if available for the training AOIs
            hashes = {}
            if 'ratios' in data and aoi_id in data['ratios']:
                hashes = data['ratios'][aoi_id].get('input_hashes', {})
                
            return budget, hashes
    except FileNotFoundError:
        raise TierCEvalError(f"Budget artifact not found at {budget_path}")
    except KeyError:
        raise TierCEvalError("Budget artifact missing 'arithmetic_mean_budget' key")
    except Exception as e:
        raise TierCEvalError(f"Failed to load budget artifact: {e}")

def score_tier_c(pred_path: str, label_path: str, cloud_mask_path: str, budget_path: str, aoi_id: str) -> dict:
    b, expected_hashes = load_budget_and_hashes(budget_path, aoi_id)
    
    # Verify hashes if they were recorded for this AOI
    if 'eval_labels_tif' in expected_hashes:
        actual_label_hash = sha256_file(label_path)
        if actual_label_hash != expected_hashes['eval_labels_tif']:
            raise TierCEvalError("Hash mismatch: Evaluation labels raster does not match the frozen budget artifact.")
            
    if 'cloud_mask_tif' in expected_hashes:
        actual_mask_hash = sha256_file(cloud_mask_path)
        if actual_mask_hash != expected_hashes['cloud_mask_tif']:
            raise TierCEvalError("Hash mismatch: Cloud mask raster does not match the frozen budget artifact.")
    
    with rasterio.open(pred_path) as p_src, rasterio.open(label_path) as l_src, rasterio.open(cloud_mask_path) as c_src:
        if p_src.dtypes[0] != 'float32':
            raise TierCEvalError(f"Prediction raster must be float32, got {p_src.dtypes[0]}")
            
        if p_src.shape != l_src.shape or p_src.transform != l_src.transform or p_src.crs != l_src.crs:
            raise TierCEvalError("Prediction raster does not perfectly match label raster grid.")
            
        if p_src.bounds != c_src.bounds or p_src.crs != c_src.crs:
            raise TierCEvalError("Cloud mask raster does not perfectly match prediction raster bounds and CRS.")
            
        # Explicit 10m to 2.5m affine transform scaling and orientation validation
        if (c_src.transform.c != p_src.transform.c or 
            c_src.transform.f != p_src.transform.f or
            c_src.transform.a != p_src.transform.a * 4 or
            c_src.transform.e != p_src.transform.e * 4 or
            c_src.transform.b != 0.0 or
            c_src.transform.d != 0.0):
            raise TierCEvalError("Cloud mask raster does not have the expected 10m affine transform or orientation.")
            
        height, width = p_src.shape
        
        preds = p_src.read(1)
        labels = l_src.read(1)
        # Cloud mask is upsampled to 2.5m using nearest
        cmask = c_src.read(1, out_shape=(height, width), resampling=Resampling.nearest)
        
    # Valid imagery pixels: not cloudy, finite predictions
    valid_mask = (cmask == 0) & np.isfinite(preds)
    
    N = int(np.sum(valid_mask))
    if N == 0:
        raise TierCEvalError("Eligible candidate pool is empty.")
        
    K = int(np.round(b * N))
    if K == 0:
        raise TierCEvalError(f"Budget {b} resulted in exactly 0 selected pixels from pool of {N}.")
    if K > N:
        raise TierCEvalError(f"Selection count {K} exceeds eligible pool {N}.")
        
    # Extract eligible scores and their original flat indices
    flat_valid_indices = np.where(valid_mask.ravel())[0]
    eligible_scores = preds.ravel()[flat_valid_indices]
    
    # Sort descending with stable sort to preserve row-major original order (tie-breaking)
    # np.argsort on -scores ascending gives descending order of scores
    sort_idx = np.argsort(-eligible_scores, kind='stable')
    
    # Take top K
    top_k_indices_in_eligible = sort_idx[:K]
    selected_flat_indices = flat_valid_indices[top_k_indices_in_eligible]
    
    # Create binary prediction mask
    pred_mask = np.zeros(height * width, dtype=bool)
    pred_mask[selected_flat_indices] = True
    pred_mask = pred_mask.reshape((height, width))
    
    # Recall calculation against label == 1
    # Note: labels==255 are strictly ignored in ground truth denominator
    true_positives = int(np.sum(pred_mask & (labels == 1) & valid_mask))
    total_positives = int(np.sum((labels == 1) & valid_mask))
    
    if total_positives == 0:
        raise TierCEvalError("Zero positive labels in the valid evaluation area. Cannot compute recall.")
        
    recall = true_positives / total_positives
    realized_area_fraction = K / N
    
    return {
        "budget_used": b,
        "eligible_pool_pixels": N,
        "selected_pixels": K,
        "true_positives_recovered": true_positives,
        "total_positive_labels": total_positives,
        "recall": recall,
        "realized_predicted_positive_area_fraction": realized_area_fraction
    }
