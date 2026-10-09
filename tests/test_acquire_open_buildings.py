import os
import pytest
import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd

import sys
sys.path.append('scripts')
from acquire_open_buildings import validate_parquet, process_aoi

def test_validate_parquet_success(tmp_path):
    gdf = gpd.GeoDataFrame({
        'confidence': [0.8, 0.9]
    }, geometry=[Polygon([(0,0), (1,0), (1,1), (0,1), (0,0)]), Polygon([(2,2), (3,2), (3,3), (2,3), (2,2)])], crs="EPSG:4326")
    
    path = tmp_path / "valid.parquet"
    gdf.to_parquet(path)
    validate_parquet(str(path), expected_len=2, threshold=0.7, min_lon=-1, min_lat=-1, max_lon=4, max_lat=4)

def test_validate_parquet_fails_threshold(tmp_path):
    gdf = gpd.GeoDataFrame({'confidence': [0.6, 0.9]}, geometry=[Polygon([(0,0), (1,0), (1,1), (0,1), (0,0)]), Polygon([(2,2), (3,2), (3,3), (2,3), (2,2)])], crs="EPSG:4326")
    path = tmp_path / "invalid_conf.parquet"
    gdf.to_parquet(path)
    with pytest.raises(ValueError, match="Confidence threshold violated"):
        validate_parquet(str(path), expected_len=2, threshold=0.7, min_lon=-1, min_lat=-1, max_lon=4, max_lat=4)

def test_validate_parquet_fails_bounds(tmp_path):
    gdf = gpd.GeoDataFrame({'confidence': [0.8]}, geometry=[Polygon([(10,10), (11,10), (11,11), (10,11), (10,10)])], crs="EPSG:4326")
    path = tmp_path / "invalid_bounds.parquet"
    gdf.to_parquet(path)
    with pytest.raises(ValueError, match="Bounds violation"):
        validate_parquet(str(path), expected_len=1, threshold=0.7, min_lon=0, min_lat=0, max_lon=5, max_lat=5)

def test_validate_parquet_fails_length(tmp_path):
    gdf = gpd.GeoDataFrame({'confidence': [0.8]}, geometry=[Polygon([(0,0), (1,0), (1,1), (0,1), (0,0)])], crs="EPSG:4326")
    path = tmp_path / "invalid_len.parquet"
    gdf.to_parquet(path)
    with pytest.raises(ValueError, match="Row count mismatch"):
        validate_parquet(str(path), expected_len=2, threshold=0.7, min_lon=-1, min_lat=-1, max_lon=4, max_lat=4)

def test_process_aoi_reuse_existing(tmp_path, monkeypatch):
    import acquire_open_buildings
    download_called = False
    def mock_download(url, path):
        nonlocal download_called
        download_called = True
        return 100
    
    read_csv_called = False
    def mock_read_csv(*args, **kwargs):
        nonlocal read_csv_called
        read_csv_called = True
        return iter([])
        
    monkeypatch.setattr(acquire_open_buildings, 'download_file', mock_download)
    monkeypatch.setattr(pd, 'read_csv', mock_read_csv)
    monkeypatch.setattr(acquire_open_buildings, 'get_s2_cells', lambda *args, **kwargs: ['14'])
    
    gdf = gpd.GeoDataFrame({'confidence': [0.8]}, geometry=[Polygon([(0,0), (1,0), (1,1), (0,1), (0,0)])], crs="EPSG:4326")
    os.makedirs(tmp_path / "processed", exist_ok=True)
    out_file = tmp_path / "processed" / "test_aoi.parquet"
    gdf.to_parquet(out_file)
    
    aoi_info = {'id': 'test_aoi', 'bbox': [0, 0, 5, 5]}
    existing_manifest = {
        'runs': [{
            'aoi_id': 'test_aoi',
            'bbox': [0, 0, 5, 5],
            'threshold': 0.70,
            's2_cells': ['14'],
            'features_after_clip': 1,
            'output_file': str(out_file)
        }]
    }
    
    result = process_aoi(aoi_info, str(tmp_path / "raw"), str(tmp_path / "processed"), threshold=0.70, existing_manifest=existing_manifest)
    
    assert result['features_after_clip'] == 1
    assert not download_called
    assert not read_csv_called

def test_process_aoi_invalid_cache_regenerates(tmp_path, monkeypatch):
    import acquire_open_buildings
    download_called = False
    def mock_download(url, path):
        nonlocal download_called
        download_called = True
        return 100
    
    read_csv_called = False
    def mock_read_csv(*args, **kwargs):
        nonlocal read_csv_called
        read_csv_called = True
        return iter([pd.DataFrame(columns=['latitude', 'longitude', 'confidence', 'geometry'])])
        
    monkeypatch.setattr(acquire_open_buildings, 'download_file', mock_download)
    monkeypatch.setattr(pd, 'read_csv', mock_read_csv)
    monkeypatch.setattr(acquire_open_buildings, 'get_s2_cells', lambda *args, **kwargs: ['14'])
    
    os.makedirs(tmp_path / "processed", exist_ok=True)
    out_file = tmp_path / "processed" / "test_aoi.parquet"
    
    # WRONG LENGTH in manifest to trigger invalidation
    gdf = gpd.GeoDataFrame({'confidence': [0.8]}, geometry=[Polygon([(0,0), (1,0), (1,1), (0,1), (0,0)])], crs="EPSG:4326")
    gdf.to_parquet(out_file)
    
    aoi_info = {'id': 'test_aoi', 'bbox': [0, 0, 5, 5]}
    existing_manifest = {
        'runs': [{
            'aoi_id': 'test_aoi',
            'bbox': [0, 0, 5, 5],
            'threshold': 0.70,
            's2_cells': ['14'],
            'features_after_clip': 999, # mismatch
            'output_file': str(out_file)
        }]
    }
    
    process_aoi(aoi_info, str(tmp_path / "raw"), str(tmp_path / "processed"), threshold=0.70, existing_manifest=existing_manifest)
    
    assert read_csv_called

