import pytest
import yaml
from pathlib import Path
from unittest.mock import patch

from src.ingest.scene_selection import load_aoi_config, select_best_scene, EARTH_SEARCH_URL

def test_aoi_validation():
    # Load and validate configs/aois.yaml
    path = Path("configs/aois.yaml")
    assert path.exists(), "AOI config must exist"
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        
    aois = config.get("aois", [])
    assert len(aois) >= 5, "Must have at least 5 AOIs"
    
    seen_ids = set()
    valid_categories = {"urban", "peri_urban", "flood", "coastal", "hill"}
    
    for aoi in aois:
        for field in ["id", "name", "state", "category", "bbox", "search_date_range", "max_cloud_cover"]:
            assert field in aoi, f"Missing {field} in {aoi.get('id', 'unknown')}"
            
        assert aoi["id"] not in seen_ids, f"Duplicate AOI ID: {aoi['id']}"
        seen_ids.add(aoi["id"])
        
        assert aoi["category"] in valid_categories, f"Invalid category: {aoi['category']}"
        
        bbox = aoi["bbox"]
        assert len(bbox) == 4, "bbox must have 4 coordinates"
        assert bbox[0] < bbox[2], "min_lon must be less than max_lon"
        assert bbox[1] < bbox[3], "min_lat must be less than max_lat"
        
        assert -180.0 <= bbox[0] <= 180.0
        assert -180.0 <= bbox[2] <= 180.0
        assert -90.0 <= bbox[1] <= 90.0
        assert -90.0 <= bbox[3] <= 90.0
        
        assert 0 <= aoi["max_cloud_cover"] <= 100

@pytest.fixture
def mock_aoi_config(tmp_path):
    config = {
        "aois": [
            {
                "id": "test_aoi",
                "name": "Test AOI",
                "state": "Test",
                "category": "urban",
                "bbox": [77.0, 28.0, 77.1, 28.1],
                "search_date_range": "2023-01-01/2023-01-31",
                "max_cloud_cover": 10
            }
        ]
    }
    config_file = tmp_path / "test_aois.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    return str(config_file)

@patch('src.ingest.scene_selection.discover_items')
def test_scene_filtering_and_deterministic_selection(mock_discover, mock_aoi_config):
    class MockItem:
        def __init__(self, item_id, cc, dt):
            self.id = item_id
            self.properties = {"eo:cloud_cover": cc, "datetime": dt}
            self.datetime = None
            
    item_high_cloud = MockItem("A", 15.0, "2023-01-05T00:00:00Z") # Should be filtered out
    item_tie1 = MockItem("B", 5.0, "2023-01-10T00:00:00Z") # Best cc, older dt -> Should win
    item_tie2 = MockItem("C", 5.0, "2023-01-15T00:00:00Z") # Best cc, newer dt
    item_tie3 = MockItem("D", 8.0, "2023-01-05T00:00:00Z") # Worse cc
    
    mock_discover.return_value = [item_tie3, item_high_cloud, item_tie2, item_tie1]
    
    result = select_best_scene("test_aoi", mock_aoi_config)
    
    assert result["selected_item_id"] == "B"
    assert result["cloud_cover"] == 5.0
    assert result["acquisition_datetime"] == "2023-01-10T00:00:00Z"

@patch('src.ingest.scene_selection.discover_items')
def test_deterministic_item_id_tiebreaker(mock_discover, mock_aoi_config):
    class MockItem:
        def __init__(self, item_id, cc, dt):
            self.id = item_id
            self.properties = {"eo:cloud_cover": cc, "datetime": dt}
            self.datetime = None
            
    item1 = MockItem("Z_item", 2.0, "2023-01-10T00:00:00Z")
    item2 = MockItem("A_item", 2.0, "2023-01-10T00:00:00Z")
    
    mock_discover.return_value = [item1, item2]
    
    result = select_best_scene("test_aoi", mock_aoi_config)
    assert result["selected_item_id"] == "A_item"
    
    mock_discover.return_value = [item2, item1]
    result2 = select_best_scene("test_aoi", mock_aoi_config)
    assert result2["selected_item_id"] == "A_item"

@patch('src.ingest.scene_selection.discover_items')
def test_explicit_date_filtering(mock_discover, mock_aoi_config):
    class MockItem:
        def __init__(self, item_id, cc, dt):
            self.id = item_id
            self.properties = {"eo:cloud_cover": cc, "datetime": dt}
            self.datetime = None
            
    out_of_range1 = MockItem("past", 1.0, "2022-12-31T23:59:59Z")
    out_of_range2 = MockItem("future", 1.0, "2023-02-01T00:00:00Z")
    valid = MockItem("valid", 9.0, "2023-01-15T10:00:00Z")
    
    mock_discover.return_value = [out_of_range1, out_of_range2, valid]
    
    result = select_best_scene("test_aoi", mock_aoi_config)
    assert result["selected_item_id"] == "valid"

@patch('src.ingest.scene_selection.discover_items')
def test_missing_metadata(mock_discover, mock_aoi_config):
    class MockItem:
        def __init__(self, item_id, cc, dt):
            self.id = item_id
            self.properties = {}
            if cc is not None:
                self.properties["eo:cloud_cover"] = cc
            if dt is not None:
                self.properties["datetime"] = dt
            self.datetime = None
            
    missing_cc = MockItem("no_cc", None, "2023-01-10T00:00:00Z")
    missing_dt = MockItem("no_dt", 2.0, None)
    valid = MockItem("valid", 8.0, "2023-01-15T00:00:00Z")
    
    mock_discover.return_value = [missing_cc, missing_dt, valid]
    
    result = select_best_scene("test_aoi", mock_aoi_config)
    assert result["selected_item_id"] == "valid"
    
@patch('src.ingest.scene_selection.discover_items')
def test_no_result_behavior(mock_discover, mock_aoi_config):
    mock_discover.return_value = []
    with pytest.raises(ValueError, match="No scenes found"):
        select_best_scene("test_aoi", mock_aoi_config)

@patch('src.ingest.scene_selection.discover_items')
def test_selection_metadata_content(mock_discover, mock_aoi_config):
    class MockItem:
        def __init__(self):
            self.id = "valid_item"
            self.properties = {"eo:cloud_cover": 3.5, "datetime": "2023-01-15T00:00:00Z"}
            self.datetime = None
            
    mock_discover.return_value = [MockItem()]
    
    result = select_best_scene("test_aoi", mock_aoi_config)
    
    expected_keys = [
        "aoi_id", "aoi_name", "bbox", "search_date_range", 
        "max_cloud_cover_threshold", "stac_collection", "provider_url", 
        "selected_item_id", "acquisition_datetime", "cloud_cover", 
        "selection_rule", "selection_version"
    ]
    
    for key in expected_keys:
        assert key in result
        
    assert result["aoi_id"] == "test_aoi"
    assert result["selected_item_id"] == "valid_item"
    assert result["cloud_cover"] == 3.5
    assert result["acquisition_datetime"] == "2023-01-15T00:00:00Z"
