import json, rasterio, numpy as np, pystac_client
from shapely.geometry import box, shape
import os

print("--- Q1: STAC Metadata ---")
with open("outputs/t2_2_smoke/selection.json") as f:
    sel = json.load(f)
item_id = sel["selected_item_id"]

with open(f"data/raw/sentinel2/{item_id}/metadata.json") as f:
    raw_meta = json.load(f)
print("boa_add_offset (from raw metadata.json):", raw_meta.get("boa_add_offset"))

client = pystac_client.Client.open(sel["provider_url"])
stac_item = next(client.search(collections=[sel["stac_collection"]], ids=[item_id]).items())
print("s2:processing_baseline:", stac_item.properties.get("s2:processing_baseline"))
print("earthsearch:boa_offset_applied:", stac_item.properties.get("earthsearch:boa_offset_applied"))
for band in ["red", "nir"]:
    asset = stac_item.assets.get(band)
    rb = asset.extra_fields.get("raster:bands", [{}])[0]
    print(f"band {band} scale:", rb.get("scale"), "offset:", rb.get("offset"))

print("\n--- Q2: Raw DN Percentiles ---")
for b in ["B02", "B04", "B08"]:
    with rasterio.open(f"data/raw/sentinel2/{item_id}/{b}.tif") as src:
        data = src.read(1, out_shape=(src.height // 10, src.width // 10))
        p01 = np.percentile(data, 0.1)
        p1 = np.percentile(data, 1)
        p5 = np.percentile(data, 5)
        p50 = np.percentile(data, 50)
        zeros = np.mean(data == 0)
        print(f"{b} (Raw) - p0.1: {p01}, p1: {p1}, p5: {p5}, p50: {p50}, zeroes: {zeros:.4%}")

print("\n--- Q3: Processed <0 fraction ---")
for b in ["B02", "B03", "B04", "B08"]:
    with rasterio.open(f"data/processed/sentinel2/{item_id}/{b}.tif") as src:
        data = src.read(1, out_shape=(src.height // 10, src.width // 10))
        neg = np.nanmean(data < 0)
        print(f"{b} (Processed) - neg fraction: {neg:.4%}")

print("\n--- Q4: AOI Footprint Overlap ---")
bbox = sel["bbox"]
aoi_geom = box(bbox[0], bbox[1], bbox[2], bbox[3])
item_geom = shape(stac_item.geometry)
intersection = aoi_geom.intersection(item_geom)
print("Overlap fraction:", intersection.area / aoi_geom.area)

