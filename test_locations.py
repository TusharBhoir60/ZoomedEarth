from pathlib import Path
import json
import rasterio

item = "S2B_43RFM_20230203_0_L2A"

tile_root = Path(f"data/tiles/sentinel2/{item}")
sr_root = Path(f"data/outputs/sen2sr/{item}")

print("=== SR TILE LOCATIONS ===")

tiles = sorted(sr_root.glob("tile_*/SEN2SR.tif"))

print("SR tiles found:", len(tiles))

for p in tiles:
    tile_id = p.parent.name

    with rasterio.open(p) as src:
        print(f"\n{tile_id}")
        print("  shape:", (src.height, src.width))
        print("  res:", src.res)
        print("  bounds:", src.bounds)
        print("  crs:", src.crs)

    meta_path = tile_root / tile_id / "metadata.json"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print("  LR window:", meta.get("window"))
        print("  row/column:", meta.get("row"), meta.get("column"))
        print("  LR transform:", meta.get("transform"))

print("\n=== STITCHED SCENE ===")

scene = sr_root / "SEN2SR_scene.tif"

with rasterio.open(scene) as src:
    print("shape:", (src.height, src.width))
    print("res:", src.res)
    print("crs:", src.crs)
    print("bounds:", src.bounds)
    print("transform:", src.transform)

print("\n=== SCENE METADATA ===")

meta = sr_root / "scene_metadata.json"

if meta.exists():
    data = json.loads(meta.read_text())

    for key in [
        "item_id",
        "tile_count",
        "tile_size",
        "overlap",
        "lr_overlap",
        "sr_overlap",
        "blend_method",
        "band_order",
        "source_scene",
        "crs",
        "resolution",
        "width",
        "height",
    ]:
        print(f"{key}: {data.get(key)}")
else:
    print("scene_metadata.json NOT FOUND")
