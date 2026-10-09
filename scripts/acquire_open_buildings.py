import os
import sys
import yaml
import json
import s2sphere
import urllib.request
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import box
import time

def get_s2_cells(min_lon, min_lat, max_lon, max_lat, level=4):
    p1 = s2sphere.LatLng.from_degrees(min_lat, min_lon)
    p2 = s2sphere.LatLng.from_degrees(max_lat, max_lon)
    rect = s2sphere.LatLngRect.from_point_pair(p1, p2)
    region_coverer = s2sphere.RegionCoverer()
    region_coverer.min_level = level
    region_coverer.max_level = level
    return [cell.to_token() for cell in region_coverer.get_covering(rect)]

def download_file(url, output_path):
    if os.path.exists(output_path):
        try:
            req = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(req) as response:
                expected_size = int(response.headers['Content-Length'])
            if os.path.getsize(output_path) == expected_size:
                print(f"File {output_path} already exists and size matches ({expected_size} bytes). Skipping download.")
                return expected_size
            else:
                print(f"File {output_path} exists but size mismatch. Re-downloading...")
        except Exception as e:
            print(f"Error checking HEAD for {url}: {e}")
            
    print(f"Downloading {url} to {output_path}...")
    with urllib.request.urlopen(url) as response, open(output_path, 'wb') as out_file:
        expected_size = int(response.headers.get('Content-Length', 0))
        downloaded = 0
        chunk_size = 8192 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            print(f"\rDownloaded {downloaded / 1024 / 1024:.2f} MB / {expected_size / 1024 / 1024:.2f} MB", end="")
    print("\nDownload complete.")
    return expected_size

def validate_parquet(filepath, expected_len, threshold, min_lon, min_lat, max_lon, max_lat):
    try:
        test_gdf = gpd.read_parquet(filepath)
    except Exception as e:
        raise ValueError(f"Failed to read parquet: {e}")
        
    if test_gdf.crs is None or test_gdf.crs.to_string() != "EPSG:4326":
        raise ValueError(f"Invalid CRS: {test_gdf.crs}")
    if len(test_gdf) != expected_len:
        raise ValueError(f"Row count mismatch: {len(test_gdf)} vs {expected_len}")
    if 'confidence' not in test_gdf.columns or 'geometry' not in test_gdf.columns:
        raise ValueError("Missing required columns")
    if test_gdf['confidence'].min() < threshold:
        raise ValueError("Confidence threshold violated")
    if not test_gdf.geometry.is_valid.all():
        raise ValueError("Invalid geometries found")
    if test_gdf.geometry.isnull().any():
        raise ValueError("Null geometries found")
    if test_gdf.geometry.is_empty.any():
        raise ValueError("Empty geometries found")
        
    bounds = test_gdf.total_bounds
    if bounds[0] < min_lon - 1e-6 or bounds[1] < min_lat - 1e-6 or bounds[2] > max_lon + 1e-6 or bounds[3] > max_lat + 1e-6:
        raise ValueError("Bounds violation")

def process_aoi(aoi_info, raw_dir, processed_dir, threshold=0.70, existing_manifest=None):
    aoi_id = aoi_info['id']
    min_lon, min_lat, max_lon, max_lat = aoi_info['bbox']
    cells = get_s2_cells(min_lon, min_lat, max_lon, max_lat, level=4)
    
    aoi_box = box(min_lon, min_lat, max_lon, max_lat)
    output_file = os.path.join(processed_dir, f"{aoi_id}.parquet")
    
    # Check for valid reuse
    if existing_manifest and 'runs' in existing_manifest:
        for run in existing_manifest['runs']:
            if (run.get('aoi_id') == aoi_id and 
                run.get('bbox') == aoi_info['bbox'] and
                run.get('threshold') == threshold and
                run.get('s2_cells') == cells and
                os.path.exists(output_file)):
                try:
                    print(f"Validating existing output for {aoi_id}...")
                    validate_parquet(output_file, run['features_after_clip'], threshold, min_lon, min_lat, max_lon, max_lat)
                    print(f"Reusing existing valid output {output_file}.")
                    return run
                except Exception as e:
                    print(f"Existing output for {aoi_id} is invalid ({e}), regenerating...")
                    
    manifest_entry = {
        'aoi_id': aoi_id,
        'bbox': aoi_info['bbox'],
        'threshold': threshold,
        's2_cells': cells,
        'sources': [],
        'features_total_raw': 0,
        'features_after_confidence': 0,
        'features_after_bbox': 0,
        'features_after_clip': 0,
        'output_file': output_file
    }
    
    gdfs = []
    
    for cell in cells:
        url = f"https://storage.googleapis.com/open-buildings-data/v3/polygons_s2_level_4_gzip/{cell}_buildings.csv.gz"
        cache_path = os.path.join(raw_dir, f"{cell}_buildings.csv.gz")
        size = download_file(url, cache_path)
        manifest_entry['sources'].append({'url': url, 'size_bytes': size})
        
        print(f"Processing {cache_path} in chunks...")
        chunk_iter = pd.read_csv(
            cache_path, 
            chunksize=500000, 
            usecols=['latitude', 'longitude', 'confidence', 'geometry'],
            engine='c'
        )
        
        for i, chunk in enumerate(chunk_iter):
            manifest_entry['features_total_raw'] += len(chunk)
            
            chunk = chunk[chunk['confidence'] >= threshold]
            manifest_entry['features_after_confidence'] += len(chunk)
            
            if chunk.empty:
                continue
                
            buf = 0.005
            chunk = chunk[
                (chunk['latitude'] >= min_lat - buf) & (chunk['latitude'] <= max_lat + buf) &
                (chunk['longitude'] >= min_lon - buf) & (chunk['longitude'] <= max_lon + buf)
            ]
            manifest_entry['features_after_bbox'] += len(chunk)
            
            if chunk.empty:
                continue
                
            geometries = chunk['geometry'].apply(wkt.loads)
            gdf_chunk = gpd.GeoDataFrame(chunk[['confidence']], geometry=geometries, crs="EPSG:4326")
            
            gdf_chunk.geometry = gdf_chunk.geometry.make_valid()
            clipped = gpd.clip(gdf_chunk, aoi_box)
            clipped = clipped[~clipped.is_empty]
            
            manifest_entry['features_after_clip'] += len(clipped)
            
            if not clipped.empty:
                gdfs.append(clipped)
                
            print(f"\rProcessed chunk {i+1} for cell {cell}. Cumulative clipped: {manifest_entry['features_after_clip']}", end="")
        print()
        
    if gdfs:
        final_gdf = pd.concat(gdfs, ignore_index=True)
        final_gdf.geometry = final_gdf.geometry.make_valid()
        final_gdf = final_gdf[~final_gdf.is_empty]
        
        tmp_file = output_file + ".tmp"
        print(f"Writing to temporary file {tmp_file}...")
        final_gdf.to_parquet(tmp_file)
        
        print("Validating temporary file...")
        try:
            validate_parquet(tmp_file, len(final_gdf), threshold, min_lon, min_lat, max_lon, max_lat)
            os.replace(tmp_file, output_file)
            print(f"Validation successful. Saved {output_file} with {len(final_gdf)} features.")
        except Exception as e:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
            raise RuntimeError(f"Output validation failed: {e}. Cleaned up temp file.")
            
    else:
        print(f"No features found for {aoi_id} within bounds.")
        
    return manifest_entry

def main():
    target_aois = ['aoi_delhi_urban', 'aoi_mumbai_urban', 'aoi_pune_peri_urban']
    
    with open('configs/aois.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    aois_to_process = [aoi for aoi in config['aois'] if aoi['id'] in target_aois]
    
    raw_dir = 'data/raw/open_buildings'
    processed_dir = 'data/processed/open_buildings'
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    manifest_path = os.path.join(processed_dir, 'manifest.json')
    existing_manifest = None
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            try:
                existing_manifest = json.load(f)
            except json.JSONDecodeError:
                pass
    
    manifest = {
        'dataset': 'Google Open Buildings V3',
        'license': 'CC BY-4.0',
        'attribution': 'Google Research',
        'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'runs': []
    }
    
    for aoi in aois_to_process:
        print(f"=== Processing {aoi['id']} ===")
        entry = process_aoi(aoi, raw_dir, processed_dir, threshold=0.70, existing_manifest=existing_manifest)
        manifest['runs'].append(entry)
        
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")

if __name__ == '__main__':
    main()
