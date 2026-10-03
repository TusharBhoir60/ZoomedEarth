import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from urllib.error import URLError

from src.ingest.download import validate_cached_file, download_file, resolve_and_download_assets
from src.ingest.stac import REQUIRED_BANDS

def test_validate_cached_file(tmp_path):
    valid_file = tmp_path / "valid.tif"
    valid_file.write_text("dummy data")
    assert validate_cached_file(valid_file) is True
    
    empty_file = tmp_path / "empty.tif"
    empty_file.write_text("")
    assert validate_cached_file(empty_file) is False
    
    missing_file = tmp_path / "missing.tif"
    assert validate_cached_file(missing_file) is False

@patch('urllib.request.urlretrieve')
def test_download_file_success(mock_urlretrieve, tmp_path):
    dest = tmp_path / "test.tif"
    
    def mock_retrieve(url, path):
        with open(path, "w") as f:
            f.write("content")
    
    mock_urlretrieve.side_effect = mock_retrieve
    
    download_file("http://example.com/test.tif", dest)
    assert validate_cached_file(dest)
    assert not dest.with_suffix(dest.suffix + ".tmp").exists()

@patch('urllib.request.urlretrieve')
def test_download_file_empty_failure(mock_urlretrieve, tmp_path):
    dest = tmp_path / "test.tif"
    
    def mock_retrieve(url, path):
        with open(path, "w") as f:
            pass
            
    mock_urlretrieve.side_effect = mock_retrieve
    
    with pytest.raises(RuntimeError, match="Failed to download .* empty"):
        download_file("http://example.com/test.tif", dest)
        
    assert not dest.exists()
    assert not dest.with_suffix(dest.suffix + ".tmp").exists()

@patch('urllib.request.urlretrieve')
def test_download_file_network_failure(mock_urlretrieve, tmp_path):
    dest = tmp_path / "test.tif"
    mock_urlretrieve.side_effect = URLError("Network error")
    
    with pytest.raises(RuntimeError, match="Failed to download"):
        download_file("http://example.com/test.tif", dest)
        
    assert not dest.exists()

@pytest.fixture
def mock_selection_metadata():
    return {
        "aoi_id": "test_aoi",
        "aoi_name": "Test",
        "bbox": [0, 0, 1, 1],
        "search_date_range": "2023-01-01/2023-01-31",
        "max_cloud_cover_threshold": 10,
        "stac_collection": "sentinel-2-l2a",
        "provider_url": "https://earth-search.aws.element84.com/v1",
        "selected_item_id": "test_item_123",
        "acquisition_datetime": "2023-01-15T00:00:00Z",
        "cloud_cover": 5.0,
        "selection_rule": "cloud_cover (asc) -> datetime (asc) -> item_id (asc)",
        "selection_version": "v1.0"
    }

@patch('urllib.request.urlretrieve')
@patch('pystac_client.Client.open')
def test_resolve_and_download_assets_success(mock_client_open, mock_urlretrieve, tmp_path, mock_selection_metadata):
    mock_item = MagicMock()
    mock_item.id = "test_item_123"
    
    class MockAsset:
        def __init__(self, href):
            self.href = href
            
    mock_item.assets = {
        "blue": MockAsset("http://example.com/B02.tif"),
        "green": MockAsset("http://example.com/B03.tif"),
        "red": MockAsset("http://example.com/B04.tif"),
        "nir": MockAsset("http://example.com/B08.tif"),
        "scl": MockAsset("http://example.com/SCL.tif")
    }
    
    mock_client = MagicMock()
    mock_search = MagicMock()
    mock_search.items.return_value = [mock_item]
    mock_client.search.return_value = mock_search
    mock_client_open.return_value = mock_client
    
    def mock_retrieve(url, path):
        with open(path, "w") as f:
            f.write("mocked_data")
            
    mock_urlretrieve.side_effect = mock_retrieve
    
    cache_dir = tmp_path / "cache"
    
    metadata = resolve_and_download_assets(mock_selection_metadata, cache_dir=str(cache_dir))
    
    item_cache = cache_dir / "test_item_123"
    assert item_cache.is_dir()
    
    for band in REQUIRED_BANDS:
        assert (item_cache / f"{band}.tif").exists()
        
    assert metadata["item_id"] == "test_item_123"
    assert metadata["collection"] == "sentinel-2-l2a"
    assert metadata["provider_url"] == "https://earth-search.aws.element84.com/v1"
    assert metadata["acquisition_datetime"] == "2023-01-15T00:00:00Z"
    assert "download_timestamp" in metadata
    assert metadata["download_version"] == "v1.0"
    
    assert len(metadata["assets"]) == 5
    for asset in metadata["assets"]:
        assert "band" in asset
        assert "local_path" in asset
        assert "source_href" in asset
        assert "file_size_bytes" in asset
        assert asset["file_size_bytes"] > 0
        
    # Assert skips existing cache
    mock_urlretrieve.reset_mock()
    resolve_and_download_assets(mock_selection_metadata, cache_dir=str(cache_dir))
    mock_urlretrieve.assert_not_called()

@patch('urllib.request.urlretrieve')
@patch('pystac_client.Client.open')
def test_resolve_and_download_missing_asset(mock_client_open, mock_urlretrieve, tmp_path, mock_selection_metadata):
    mock_item = MagicMock()
    mock_item.id = "test_item_123"
    class MockAsset:
        def __init__(self, href):
            self.href = href
            
    mock_item.assets = {
        "blue": MockAsset("http://example.com/B02.tif"),
        "green": MockAsset("http://example.com/B03.tif"),
        "red": MockAsset("http://example.com/B04.tif"),
        "nir": MockAsset("http://example.com/B08.tif")
    }
    
    mock_client = MagicMock()
    mock_search = MagicMock()
    mock_search.items.return_value = [mock_item]
    mock_client.search.return_value = mock_search
    mock_client_open.return_value = mock_client
    
    with pytest.raises(ValueError, match="missing required assets: SCL"):
        resolve_and_download_assets(mock_selection_metadata, cache_dir=str(tmp_path))

@patch('pystac_client.Client.open')
def test_resolve_and_download_item_not_found(mock_client_open, tmp_path, mock_selection_metadata):
    mock_client = MagicMock()
    mock_search = MagicMock()
    mock_search.items.return_value = []
    mock_client.search.return_value = mock_search
    mock_client_open.return_value = mock_client
    
    with pytest.raises(ValueError, match="not found at"):
        resolve_and_download_assets(mock_selection_metadata, cache_dir=str(tmp_path))
