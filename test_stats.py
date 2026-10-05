import rasterio
import numpy as np
from pathlib import Path

p = Path("data/outputs/sen2sr/S2B_43RFM_20230203_0_L2A/SEN2SR_scene.tif")

with rasterio.open(p) as src:
    print("shape:", (src.height, src.width))
    print("count:", src.count)
    print("dtype:", src.dtypes)
    print("crs:", src.crs)
    print("resolution:", src.res)
    print("transform:", src.transform)
    print("bounds:", src.bounds)
    print("nodata:", src.nodata)
    print("is_tiled:", src.is_tiled)
    print("block_shapes:", src.block_shapes)
    print("compress:", src.compression)
    print("BIGTIFF:", src.profile.get("BIGTIFF"))

    print("\nPer-band statistics:")
    for i in range(1, src.count + 1):
        valid_count = 0
        min_val = np.inf
        max_val = -np.inf
        for ji, window in src.block_windows(i):
            block = src.read(i, window=window)
            valid_mask = ~np.isnan(block)
            valid_pixels = block[valid_mask]
            count = valid_pixels.size
            if count > 0:
                valid_count += count
                b_min = valid_pixels.min()
                b_max = valid_pixels.max()
                if b_min < min_val: min_val = b_min
                if b_max > max_val: max_val = b_max
        print(f"band {i}: valid={valid_count}, min={min_val if valid_count > 0 else None}, max={max_val if valid_count > 0 else None}")
