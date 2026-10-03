import pytest
from unittest.mock import patch, MagicMock
from src.ingest.stac import get_asset_mapping, discover_items, verify_required_assets, EARTH_SEARCH_URL, PC_URL

def test_asset_mapping():
    es_mapping = get_asset_mapping(EARTH_SEARCH_URL)
    assert es_mapping["B02"] == "blue"
    assert es_mapping["B08"] == "nir"
    assert es_mapping["SCL"] == "scl"
    
    pc_mapping = get_asset_mapping(PC_URL)
    assert pc_mapping["B02"] == "B02"
    assert pc_mapping["SCL"] == "SCL"
    
    with pytest.raises(ValueError, match="Unsupported STAC provider"):
        get_asset_mapping("https://unknown-provider.com")

@patch('src.ingest.stac.pystac_client.Client.open')
def test_discover_items(mock_open):
    mock_client = MagicMock()
    mock_open.return_value = mock_client
    
    mock_search = MagicMock()
    mock_client.search.return_value = mock_search
    
    mock_search.items.return_value = ["item1", "item2"]
    
    bbox = [10, 20, 30, 40]
    datetime = "2023-01-01/2023-01-02"
    query = {"eo:cloud_cover": {"lt": 10}}
    
    items = discover_items(bbox, datetime, query=query)
    
    assert items == ["item1", "item2"]
    mock_open.assert_called_once_with(EARTH_SEARCH_URL)
    mock_client.search.assert_called_once_with(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=datetime,
        max_items=10,
        query=query
    )

def test_verify_required_assets_success():
    class MockItem:
        def __init__(self, assets):
            self.assets = assets
            
    # Earth Search format
    es_item = MockItem({"blue": {}, "green": {}, "red": {}, "nir": {}, "scl": {}, "extra": {}})
    verify_required_assets(es_item, EARTH_SEARCH_URL) # Should not raise
    
    # Planetary Computer format
    pc_item = MockItem({"B02": {}, "B03": {}, "B04": {}, "B08": {}, "SCL": {}})
    verify_required_assets(pc_item, PC_URL) # Should not raise

def test_verify_required_assets_failure():
    class MockItem:
        def __init__(self, assets):
            self.assets = assets
            
    # Missing SCL and red
    es_item = MockItem({"blue": {}, "green": {}, "nir": {}})
    with pytest.raises(ValueError, match="STAC item is missing required assets: B04 \\(expected key: red\\), SCL \\(expected key: scl\\)"):
        verify_required_assets(es_item, EARTH_SEARCH_URL)
