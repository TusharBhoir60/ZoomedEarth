"""
Deterministic Scene Selection for Sentinel-2 L2A.

Project: Sentinel-2 Super-Resolution + Trust Harness (SIH26142, NTRO)
"""

import yaml
from pathlib import Path
from typing import Dict, Any

from src.ingest.stac import discover_items, EARTH_SEARCH_URL

def load_aoi_config(aoi_id: str, config_path: str = "configs/aois.yaml") -> Dict[str, Any]:
    """
    Load and validate the AOI configuration for a given AOI ID.
    
    Args:
        aoi_id: The ID of the AOI.
        config_path: Path to the AOI YAML configuration file.
        
    Returns:
        Dict containing AOI definition.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"AOI configuration file not found: {config_path}")
        
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        
    if "aois" not in config:
        raise ValueError(f"Invalid config schema: missing 'aois' key in {config_path}")
        
    aoi_data = None
    for aoi in config["aois"]:
        if aoi.get("id") == aoi_id:
            aoi_data = aoi
            break
            
    if not aoi_data:
        raise ValueError(f"AOI ID '{aoi_id}' not found in {config_path}")
        
    # Schema validation
    required_fields = ["id", "name", "state", "category", "bbox", "search_date_range", "max_cloud_cover"]
    for field in required_fields:
        if field not in aoi_data:
            raise ValueError(f"AOI '{aoi_id}' missing required field: {field}")
            
    valid_categories = {"urban", "peri_urban", "flood", "coastal", "hill"}
    if aoi_data["category"] not in valid_categories:
        raise ValueError(f"AOI '{aoi_id}' has invalid category: {aoi_data['category']}")
        
    bbox = aoi_data["bbox"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"AOI '{aoi_id}' bbox must contain exactly 4 values")
        
    for val in bbox:
        if not isinstance(val, (int, float)):
            raise ValueError(f"AOI '{aoi_id}' bbox contains non-numeric value: {val}")
            
    if not (bbox[0] < bbox[2]):
        raise ValueError(f"AOI '{aoi_id}' bbox invalid: min_lon >= max_lon")
    if not (bbox[1] < bbox[3]):
        raise ValueError(f"AOI '{aoi_id}' bbox invalid: min_lat >= max_lat")
        
    if not (-180.0 <= bbox[0] <= 180.0 and -180.0 <= bbox[2] <= 180.0):
        raise ValueError(f"AOI '{aoi_id}' bbox longitudes must be between -180 and 180")
    if not (-90.0 <= bbox[1] <= 90.0 and -90.0 <= bbox[3] <= 90.0):
        raise ValueError(f"AOI '{aoi_id}' bbox latitudes must be between -90 and 90")
        
    max_cc = aoi_data["max_cloud_cover"]
    if not isinstance(max_cc, (int, float)):
        raise ValueError(f"AOI '{aoi_id}' max_cloud_cover must be numeric")
    if not (0 <= max_cc <= 100):
        raise ValueError(f"AOI '{aoi_id}' max_cloud_cover must be between 0 and 100")
        
    date_range = aoi_data["search_date_range"]
    if not isinstance(date_range, str) or "/" not in date_range:
        raise ValueError(f"AOI '{aoi_id}' search_date_range must be in format YYYY-MM-DD/YYYY-MM-DD")
        
    start_date, end_date = date_range.split("/", 1)
    if len(start_date) != 10 or len(end_date) != 10 or start_date > end_date:
        raise ValueError(f"AOI '{aoi_id}' search_date_range invalid format or start > end")
        
    return aoi_data

def select_best_scene(
    aoi_id: str,
    config_path: str = "configs/aois.yaml",
    provider_url: str = EARTH_SEARCH_URL,
    collection: str = "sentinel-2-l2a"
) -> Dict[str, Any]:
    """
    Deterministically select the best Sentinel-2 L2A scene for an AOI.
    
    The selection rule is:
    1. Cloud cover (ascending)
    2. Acquisition datetime (ascending) - deterministic convention, not quality claim
    3. Item ID (ascending lexicographically)
    
    Args:
        aoi_id: The ID of the AOI.
        config_path: Path to the AOI configuration file.
        provider_url: STAC API URL.
        collection: The target collection.
        
    Returns:
        Dict containing reproducible selection metadata.
    """
    aoi = load_aoi_config(aoi_id, config_path)
    
    bbox = aoi["bbox"]
    date_range = aoi["search_date_range"]
    max_cloud_cover = aoi["max_cloud_cover"]
    
    # Query STAC API
    # Apply initial cloud cover filter via API if supported
    query = {"eo:cloud_cover": {"lt": max_cloud_cover + 0.1}}
    
    items = discover_items(
        bbox=bbox,
        datetime=date_range,
        provider_url=provider_url,
        collection=collection,
        max_items=100,
        query=query
    )
    
    if not items:
        raise ValueError(f"No scenes found for AOI '{aoi_id}' within constraints.")
        
    start_date_str, end_date_str = date_range.split("/", 1)
    
    # Client-side validation and extraction
    candidates = []
    for item in items:
        cloud_cover = item.properties.get("eo:cloud_cover")
        if cloud_cover is None:
            continue
            
        # Datetime might be in properties or directly on item
        dt = item.datetime.isoformat() if getattr(item, "datetime", None) else item.properties.get("datetime")
        if not dt:
            continue
            
        dt_str = str(dt)
        date_only = dt_str[:10]
        if not (start_date_str <= date_only <= end_date_str):
            continue
            
        if cloud_cover <= max_cloud_cover:
            candidates.append({
                "item": item,
                "cloud_cover": float(cloud_cover),
                "datetime": dt_str,
                "id": str(item.id)
            })
            
    if not candidates:
        raise ValueError(f"No scenes met the max cloud cover threshold ({max_cloud_cover}) for AOI '{aoi_id}'.")
        
    # Deterministic multi-key sort
    candidates.sort(key=lambda x: (x["cloud_cover"], x["datetime"], x["id"]))
    
    best_candidate = candidates[0]
    best_item = best_candidate["item"]
    
    # Construct Reproducibility Metadata
    metadata = {
        "aoi_id": aoi_id,
        "aoi_name": aoi["name"],
        "bbox": bbox,
        "search_date_range": date_range,
        "max_cloud_cover_threshold": max_cloud_cover,
        "stac_collection": collection,
        "provider_url": provider_url,
        "selected_item_id": best_item.id,
        "acquisition_datetime": best_candidate["datetime"],
        "cloud_cover": best_candidate["cloud_cover"],
        "selection_rule": "cloud_cover (asc) -> datetime (asc) -> item_id (asc)",
        "selection_version": "v1.0"
    }
    
    return metadata
