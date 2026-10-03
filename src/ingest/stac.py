"""
STAC Discovery for Sentinel-2 L2A.

Project: Sentinel-2 Super-Resolution + Trust Harness (SIH26142, NTRO)
"""

from typing import Dict, Any, List, Optional
import pystac_client

# Intended provider order
# 1. Earth Search / AWS primary
# 2. Planetary Computer fallback
EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
PC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Required assets for the pipeline
REQUIRED_BANDS = ["B02", "B03", "B04", "B08", "SCL"]

def get_asset_mapping(provider_url: str) -> Dict[str, str]:
    """
    Return the provider-specific STAC asset keys for the canonical bands.
    
    Args:
        provider_url: The URL of the STAC endpoint.
        
    Returns:
        Dict mapping canonical band names to provider asset keys.
    """
    if "earth-search" in provider_url:
        return {
            "B02": "blue",
            "B03": "green",
            "B04": "red",
            "B08": "nir",
            "SCL": "scl"
        }
    elif "planetarycomputer" in provider_url:
        return {
            "B02": "B02",
            "B03": "B03",
            "B04": "B04",
            "B08": "B08",
            "SCL": "SCL"
        }
    else:
        raise ValueError(f"Unsupported STAC provider: {provider_url}")

def discover_items(
    bbox: List[float],
    datetime: str,
    provider_url: str = EARTH_SEARCH_URL,
    collection: str = "sentinel-2-l2a",
    max_items: int = 10,
    query: Optional[Dict[str, Any]] = None
) -> List[Any]:
    """
    Discover Sentinel-2 L2A STAC items.
    
    Args:
        bbox: Bounding box [minx, miny, maxx, maxy].
        datetime: Date range string, e.g., '2023-01-01/2023-01-31'.
        provider_url: STAC API URL.
        collection: The target collection.
        max_items: Maximum number of items to return.
        query: Additional STAC query parameters (e.g. for cloud cover).
        
    Returns:
        List of pystac.Item instances.
    """
    client = pystac_client.Client.open(provider_url)
    
    search_params = {
        "collections": [collection],
        "bbox": bbox,
        "datetime": datetime,
        "max_items": max_items
    }
    
    if query:
        search_params["query"] = query
        
    search = client.search(**search_params)
    items = list(search.items())
    
    return items

def verify_required_assets(item: Any, provider_url: str) -> None:
    """
    Verify that a STAC item contains all required assets for the pipeline.
    
    Args:
        item: A pystac.Item instance.
        provider_url: The URL of the STAC endpoint used to fetch the item.
        
    Raises:
        ValueError if a required asset is missing.
    """
    mapping = get_asset_mapping(provider_url)
    missing = []
    
    for band in REQUIRED_BANDS:
        expected_key = mapping[band]
        if expected_key not in item.assets:
            missing.append(f"{band} (expected key: {expected_key})")
            
    if missing:
        raise ValueError(f"STAC item is missing required assets: {', '.join(missing)}")
