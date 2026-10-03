"""
Comprehensive deterministic tests for T2.5: LR Tile Generation.

Covers: configuration, geometry, overlap, geospatial contract, data preservation,
edge-padding contract, cloud/valid fractions (scene pixels only), complete source
coverage, reassembly invariant, manifest, determinism, source protection.
"""
import csv
import json
import hashlib
import os
import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin, Affine
from pathlib import Path
import yaml

from src.ingest.tile_scene import create_tiles, load_tiling_config, BANDS

# ─── Helpers ────────────────────────────────────────────────────────────────

def make_scene(root: Path, item_id: str, width: int, height: int, transform=None):
    """Build a synthetic T2.4 prepared scene in *root*."""
    item_dir = root / item_id
    item_dir.mkdir(parents=True)

    if transform is None:
        transform = from_origin(500000.0, 3500000.0, 10.0, 10.0)

    f32_profile = dict(driver="GTiff", height=height, width=width, count=1,
                       dtype=rasterio.float32, crs="EPSG:32643", transform=transform)
    u8_profile  = dict(driver="GTiff", height=height, width=width, count=1,
                       dtype=rasterio.uint8,  crs="EPSG:32643", transform=transform)

    # Each band gets a unique constant so band-order errors are detectable
    band_vals = {"B02": 1.0, "B03": 2.0, "B04": 3.0, "B08": 4.0}
    for band, val in band_vals.items():
        data = np.full((height, width), val, dtype=np.float32)
        with rasterio.open(item_dir / f"{band}.tif", "w", **f32_profile) as dst:
            dst.write(data, 1)

    # SCL: all vegetation (4)
    scl = np.full((height, width), 4, dtype=np.uint8)
    with rasterio.open(item_dir / "SCL.tif", "w", **u8_profile) as dst:
        dst.write(scl, 1)

    # cloud_mask: top-left 10×10 = masked (1), rest clear (0)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:10, :10] = 1
    with rasterio.open(item_dir / "cloud_mask.tif", "w", **u8_profile) as dst:
        dst.write(mask, 1)

    return item_dir

def write_config(path: Path, tile_size: int, overlap: int, edge_policy: str = "pad"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump({"tiling": {"tile_size": tile_size, "overlap": overlap,
                               "edge_policy": edge_policy}}, f)

# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def base(tmp_path):
    """Returns (processed_dir_str, tiles_dir_str, config_path_str)."""
    processed_dir = tmp_path / "processed"
    tiles_dir     = tmp_path / "tiles"
    cfg_path      = tmp_path / "configs" / "tiling.yaml"
    write_config(cfg_path, tile_size=64, overlap=8)
    return str(processed_dir), str(tiles_dir), str(cfg_path)

# ─── 1. Configuration tests ──────────────────────────────────────────────────

def test_valid_config_zero_overlap(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    write_config(cfg, tile_size=64, overlap=0)
    c = load_tiling_config(str(cfg))
    assert c["tile_size"] == 64
    assert c["overlap"] == 0

def test_valid_config_overlap_lt_tile_size(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    write_config(cfg, tile_size=128, overlap=16)
    c = load_tiling_config(str(cfg))
    assert c["overlap"] < c["tile_size"]

@pytest.mark.parametrize("ts,ov,match", [
    (0,  0,  "tile_size"),
    (-1, 0,  "tile_size"),
    (64, -1, "overlap"),
    (64, 64, "overlap"),
    (64, 65, "overlap"),
])
def test_invalid_config(tmp_path, ts, ov, match):
    proc = tmp_path / "processed"
    proc.mkdir()
    item_dir = proc / "item0"
    item_dir.mkdir()
    # minimal scene so we get past the directory check
    transform = from_origin(0, 100, 10, 10)
    profile = dict(driver="GTiff", height=10, width=10, count=1,
                   dtype=rasterio.float32, crs="EPSG:32643", transform=transform)
    for b in BANDS + ["SCL", "cloud_mask"]:
        with rasterio.open(item_dir / f"{b}.tif", "w", **profile) as dst:
            dst.write(np.zeros((10, 10), dtype=np.float32), 1)

    cfg = tmp_path / "cfg.yaml"
    write_config(cfg, tile_size=ts, overlap=ov)
    with pytest.raises(ValueError, match=match):
        create_tiles("item0", processed_dir=str(proc),
                     tiles_dir=str(tmp_path / "tiles"), config_path=str(cfg))

# ─── 2. Geometry: exact-divisible raster ─────────────────────────────────────

def test_exact_divisible_tile_count(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    # tile_size=64, overlap=8 -> stride=56
    # 112 / 56 = 2 -> 2×2 = 4 tiles
    item_id = "scene_exact"
    make_scene(Path(proc_str), item_id, width=112, height=112)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    out = Path(tiles_str) / item_id
    tiles = sorted(out.glob("tile_*"))
    assert len(tiles) == 4, f"Expected 4, got {len(tiles)}"

def test_exact_divisible_windows_and_offsets(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    item_id = "scene_exact2"
    make_scene(Path(proc_str), item_id, width=112, height=112)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    out = Path(tiles_str) / item_id
    metas = {}
    for td in out.glob("tile_*"):
        with open(td / "metadata.json") as f:
            m = json.load(f)
        metas[m["tile_id"]] = m

    # tile_r000_c000
    m00 = metas["tile_r000_c000"]
    assert m00["window"] == [0, 0, 64, 64]
    assert m00["window"][0] == 0 and m00["window"][1] == 0

    # tile_r000_c001 x_offset = stride = 56
    m01 = metas["tile_r000_c001"]
    assert m01["window"] == [56, 0, 64, 64]
    assert m01["window"][0] == 56 and m01["window"][1] == 0

    # tile_r001_c000 y_offset = stride = 56
    m10 = metas["tile_r001_c000"]
    assert m10["window"] == [0, 56, 64, 64]
    assert m10["window"][0] == 0 and m10["window"][1] == 56

def test_exact_divisible_tile_dimensions(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    item_id = "scene_exact3"
    make_scene(Path(proc_str), item_id, width=112, height=112)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    out = Path(tiles_str) / item_id
    for td in out.glob("tile_*"):
        with rasterio.open(td / "B02.tif") as src:
            assert src.width == 64
            assert src.height == 64

# ─── 3. Geometry: non-divisible raster ───────────────────────────────────────

def test_non_divisible_tile_count(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    # 112+1=113, stride=56 -> ceil((113-8)/56) = ceil(1.875) = 2 -> 2×2=4 tiles
    item_id = "scene_odd"
    make_scene(Path(proc_str), item_id, width=113, height=113)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    out = Path(tiles_str) / item_id
    tiles = sorted(out.glob("tile_*"))
    assert len(tiles) == 4

def test_non_divisible_no_source_pixel_dropped(tmp_path, base):
    """Every source pixel is covered by at least one tile window."""
    proc_str, tiles_str, cfg_str = base
    W, H = 100, 100
    item_id = "scene_coverage"
    make_scene(Path(proc_str), item_id, width=W, height=H)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    out = Path(tiles_str) / item_id
    coverage = np.zeros((H, W), dtype=int)

    for td in sorted(out.glob("tile_*")):
        with open(td / "metadata.json") as f:
            m = json.load(f)
        x0, y0 = m["window"][0], m["window"][1]
        # Only source pixels (clip to source extent)
        x1 = min(x0 + m["tile_size"], W)
        y1 = min(y0 + m["tile_size"], H)
        if x1 > x0 and y1 > y0:
            coverage[y0:y1, x0:x1] += 1

    assert np.all(coverage >= 1), "Some source pixels have zero coverage"

# ─── 4. Stride and overlap ───────────────────────────────────────────────────

def test_stride_equals_tile_minus_overlap(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    item_id = "scene_stride"
    make_scene(Path(proc_str), item_id, width=112, height=112)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    out = Path(tiles_str) / item_id
    metas = {}
    for td in out.glob("tile_*"):
        with open(td / "metadata.json") as f:
            m = json.load(f)
        metas[m["tile_id"]] = m

    stride = metas["tile_r000_c000"]["tile_size"] - metas["tile_r000_c000"]["overlap"]
    # horizontal adjacent — use window x-component
    assert metas["tile_r000_c001"]["window"][0] - metas["tile_r000_c000"]["window"][0] == stride
    # vertical adjacent — use window y-component
    assert metas["tile_r001_c000"]["window"][1] - metas["tile_r000_c000"]["window"][1] == stride

def test_horizontal_overlap_geometry(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    item_id = "scene_hov"
    make_scene(Path(proc_str), item_id, width=112, height=112)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    out = Path(tiles_str) / item_id
    m00_x = json.load(open(out / "tile_r000_c000" / "metadata.json"))["window"][0]
    m01_x = json.load(open(out / "tile_r000_c001" / "metadata.json"))["window"][0]
    tile_size = 64
    overlap = 8
    # actual pixel overlap between c000 and c001 at row 0
    c000_end   = m00_x + tile_size   # 64
    c001_start = m01_x               # 56
    actual_overlap = c000_end - c001_start
    assert actual_overlap == overlap

def test_vertical_overlap_geometry(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    item_id = "scene_vov"
    make_scene(Path(proc_str), item_id, width=112, height=112)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    out = Path(tiles_str) / item_id
    m00_y = json.load(open(out / "tile_r000_c000" / "metadata.json"))["window"][1]
    m10_y = json.load(open(out / "tile_r001_c000" / "metadata.json"))["window"][1]
    tile_size = 64
    overlap = 8
    r000_end   = m00_y + tile_size
    r001_start = m10_y
    actual_overlap = r000_end - r001_start
    assert actual_overlap == overlap

# ─── 5. Geospatial contract ───────────────────────────────────────────────────

def test_geospatial_contract(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    # Non-trivial transform: non-zero origin, 10m pixels
    transform = from_origin(712000.0, 2800000.0, 10.0, 10.0)
    item_id = "scene_geo"
    make_scene(Path(proc_str), item_id, width=112, height=112, transform=transform)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    out = Path(tiles_str) / item_id
    stride = 56

    with rasterio.open(Path(proc_str) / item_id / "B02.tif") as src:
        src_transform = src.transform

    for td in sorted(out.glob("tile_*")):
        with open(td / "metadata.json") as f:
            m = json.load(f)
        x0, y0 = m["window"][0], m["window"][1]
        expected_tf = rasterio.windows.transform(
            rasterio.windows.Window(x0, y0, 64, 64), src_transform
        )
        stored_tf = Affine(*m["transform"])
        assert stored_tf == expected_tf, f"Transform mismatch for {m['tile_id']}"
        assert m["crs"] == "EPSG:32643"
        assert m["resolution"] == "10m"

        with rasterio.open(td / "B02.tif") as src:
            assert src.transform == expected_tf
            assert src.crs.to_string() == "EPSG:32643"
            assert abs(src.transform.a) == 10.0

# ─── 6. Data preservation / band order ───────────────────────────────────────

def test_band_order_and_values_preserved(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    item_id = "scene_bands"
    make_scene(Path(proc_str), item_id, width=112, height=112)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    out = Path(tiles_str) / item_id
    # Each band has a unique fill: B02=1, B03=2, B04=3, B08=4
    expected = {"B02": 1.0, "B03": 2.0, "B04": 3.0, "B08": 4.0}
    for td in out.glob("tile_*"):
        for band, val in expected.items():
            with rasterio.open(td / f"{band}.tif") as src:
                data = src.read(1)
                # Interior pixels (not padding) must equal the source value
                assert src.width == 64 and src.height == 64
                m = json.load(open(td / "metadata.json"))
                x0, y0 = m["window"][0], m["window"][1]
                scene_cols = min(64, 112 - x0)
                scene_rows = min(64, 112 - y0)
                interior = data[:scene_rows, :scene_cols]
                assert np.allclose(interior, val), f"{band} value wrong in {td.name}"

def test_no_channel_permutation(tmp_path, base):
    """B04 must NOT appear in the B02 tile file, etc."""
    proc_str, tiles_str, cfg_str = base
    item_id = "scene_perm"
    make_scene(Path(proc_str), item_id, width=112, height=112)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    out = Path(tiles_str) / item_id / "tile_r000_c000"
    with open(out / "metadata.json") as f:
        m = json.load(f)
    assert m["band_order"] == ["B02", "B03", "B04", "B08"]

    for band, expected_val in [("B02", 1.0), ("B03", 2.0), ("B04", 3.0), ("B08", 4.0)]:
        with rasterio.open(out / f"{band}.tif") as src:
            data = src.read(1)
            # Non-padded pixels only (tile_r000_c000 is fully interior in 112-wide scene)
            assert np.allclose(data, expected_val), f"Channel permutation detected for {band}"

def test_scl_values_preserved(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    item_id = "scene_scl"
    make_scene(Path(proc_str), item_id, width=112, height=112)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    out = Path(tiles_str) / item_id / "tile_r000_c000"
    with rasterio.open(out / "SCL.tif") as src:
        data = src.read(1)
        assert src.dtypes[0] == rasterio.uint8
        # interior SCL must be 4 (vegetation)
        assert np.all(data == 4)

def test_cloud_mask_values_preserved(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    item_id = "scene_cm"
    make_scene(Path(proc_str), item_id, width=112, height=112)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    out = Path(tiles_str) / item_id / "tile_r000_c000"
    with rasterio.open(out / "cloud_mask.tif") as src:
        data = src.read(1)
        # top-left 10x10 = 1, rest = 0
        assert np.all(data[:10, :10] == 1)
        assert np.all(data[10:, :] == 0)
        assert np.all(data[:, 10:] == 0)

# ─── 7. Edge-padding contract ─────────────────────────────────────────────────

def test_edge_padding_spectral_is_nan(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    # 100-wide scene, stride=56, tile=64 -> tile_r000_c001 starts at x=56, goes to x=120 -> padded cols 100-119
    item_id = "scene_pad"
    make_scene(Path(proc_str), item_id, width=100, height=100)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    out = Path(tiles_str) / item_id
    # Find a tile that has padding
    with open(out / "tile_r000_c001" / "metadata.json") as f:
        m = json.load(f)
    x0 = m["window"][0]  # 56
    pad_start_col = 100 - x0  # col 44 within the tile

    with rasterio.open(out / "tile_r000_c001" / "B02.tif") as src:
        data = src.read(1)
        # Columns >= pad_start_col should be NaN (padding)
        assert np.all(np.isnan(data[:, pad_start_col:])), "Padded spectral pixels must be NaN"
        # Columns < pad_start_col should be real values (1.0)
        assert np.allclose(data[:, :pad_start_col], 1.0)

def test_edge_padding_cloud_mask_is_masked(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    item_id = "scene_pad_cm"
    make_scene(Path(proc_str), item_id, width=100, height=100)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    out = Path(tiles_str) / item_id
    x0 = 56
    pad_start_col = 100 - x0  # 44
    with rasterio.open(out / "tile_r000_c001" / "cloud_mask.tif") as src:
        data = src.read(1)
        # Padding region treated as masked (fill_value=1 for cloud_mask)
        assert np.all(data[:, pad_start_col:] == 1), "Padded cloud_mask pixels must be 1"

def test_edge_padding_scl_is_deterministic(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    item_id = "scene_pad_scl"
    make_scene(Path(proc_str), item_id, width=100, height=100)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    out = Path(tiles_str) / item_id
    x0 = 56
    pad_start_col = 100 - x0
    with rasterio.open(out / "tile_r000_c001" / "SCL.tif") as src:
        data = src.read(1)
        # fill_value for SCL is 0 (no-data class)
        assert np.all(data[:, pad_start_col:] == 0), "Padded SCL pixels must be 0"

def test_edge_padding_transform_correct(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    item_id = "scene_pad_tf"
    transform = from_origin(500000.0, 3500000.0, 10.0, 10.0)
    make_scene(Path(proc_str), item_id, width=100, height=100, transform=transform)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    out = Path(tiles_str) / item_id
    with open(out / "tile_r000_c001" / "metadata.json") as f:
        m = json.load(f)
    x0, y0 = m["window"][0], m["window"][1]  # 56, 0

    expected_tf = rasterio.windows.transform(
        rasterio.windows.Window(x0, y0, 64, 64), transform
    )
    stored_tf = Affine(*m["transform"])
    assert stored_tf == expected_tf, "Edge tile transform must correspond to requested source window"

# ─── 8. Cloud/valid fraction uses scene pixels only ───────────────────────────

def test_cloud_fraction_scene_pixels_only(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    # 100-wide scene, tile_r000_c001 has 44 real cols and 20 pad cols
    # scene has cloud_mask[:10,:10]=1, rest=0
    # For tile_r000_c001 (x0=56), scene cols 0-43 in tile are source cols 56-99 -> all 0 (no cloud)
    item_id = "scene_frac"
    make_scene(Path(proc_str), item_id, width=100, height=100)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    out = Path(tiles_str) / item_id
    with open(out / "tile_r000_c001" / "metadata.json") as f:
        m = json.load(f)
    # scene cols in tile = min(64, 100-56) = 44; scene rows = min(64, 100-0) = 64
    assert m["scene_pixel_count"] == 44 * 64
    # source cols 56-99, rows 0-63 -> no cloud (cloud only in cols 0-9, rows 0-9)
    assert m["cloud_fraction"] == 0.0
    assert m["valid_fraction"] == 1.0

def test_cloud_fraction_mixed_tile(tmp_path, base):
    """Interior tile covering the cloudy top-left corner."""
    proc_str, tiles_str, cfg_str = base
    item_id = "scene_frac2"
    make_scene(Path(proc_str), item_id, width=112, height=112)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    out = Path(tiles_str) / item_id
    with open(out / "tile_r000_c000" / "metadata.json") as f:
        m = json.load(f)
    # scene 112×112; tile 64×64 from (0,0) -> fully inside source
    assert m["scene_pixel_count"] == 64 * 64
    # cloud mask top-left 10×10 = 100 masked pixels
    assert m["cloud_fraction"] == pytest.approx(100 / (64 * 64))
    assert m["valid_fraction"] == pytest.approx(1.0 - 100 / (64 * 64))

# ─── 9. Complete source coverage ─────────────────────────────────────────────

def test_complete_source_coverage(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    W, H = 150, 130
    item_id = "scene_cov"
    make_scene(Path(proc_str), item_id, width=W, height=H)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    out = Path(tiles_str) / item_id
    coverage = np.zeros((H, W), dtype=int)

    for td in out.glob("tile_*"):
        m = json.load(open(td / "metadata.json"))
        x0, y0 = m["window"][0], m["window"][1]
        x1 = min(x0 + m["tile_size"], W)
        y1 = min(y0 + m["tile_size"], H)
        if x1 > x0 and y1 > y0:
            coverage[y0:y1, x0:x1] += 1

    assert np.all(coverage >= 1), f"Min coverage = {coverage.min()}: some source pixels missed"

# ─── 10. Reassembly invariant ────────────────────────────────────────────────

def test_reassembly_exact(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    W, H = 112, 112
    item_id = "scene_reassemble"
    # Use arange as source to verify exact values
    item_dir = Path(proc_str) / item_id
    item_dir.mkdir(parents=True)
    transform = from_origin(500000.0, 3500000.0, 10.0, 10.0)
    profile = dict(driver="GTiff", height=H, width=W, count=1,
                   dtype=rasterio.float32, crs="EPSG:32643", transform=transform)
    u8_profile = profile.copy()
    u8_profile["dtype"] = rasterio.uint8
    original = np.arange(H * W, dtype=np.float32).reshape(H, W)
    for band in BANDS:
        with rasterio.open(item_dir / f"{band}.tif", "w", **profile) as dst:
            dst.write(original, 1)
    for extra in ["SCL", "cloud_mask"]:
        with rasterio.open(item_dir / f"{extra}.tif", "w", **u8_profile) as dst:
            dst.write(np.zeros((H, W), dtype=np.uint8), 1)

    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    out = Path(tiles_str) / item_id
    stride = 56
    reconstructed = np.zeros((H, W), dtype=np.float32)

    for td in sorted(out.glob("tile_*")):
        m = json.load(open(td / "metadata.json"))
        x0, y0 = m["window"][0], m["window"][1]
        with rasterio.open(td / "B02.tif") as src:
            data = src.read(1)
        # Ownership region (non-overlapping stride block)
        x1 = min(x0 + stride, W)
        y1 = min(y0 + stride, H)
        h = y1 - y0
        w = x1 - x0
        reconstructed[y0:y1, x0:x1] = data[:h, :w]

    np.testing.assert_array_equal(reconstructed, original)

# ─── 11. Manifest ────────────────────────────────────────────────────────────

MANIFEST_REQUIRED_COLS = [
    "tile_id", "item_id", "row", "column", "x_offset", "y_offset",
    "width", "height", "minx", "miny", "maxx", "maxy",
    "cloud_fraction", "valid_fraction", "crs", "split"
]

def test_manifest_required_columns(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    item_id = "scene_manifest"
    make_scene(Path(proc_str), item_id, width=112, height=112)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    with open(Path(tiles_str) / item_id / "manifest.csv") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert set(MANIFEST_REQUIRED_COLS).issubset(set(reader.fieldnames))
    for row in rows:
        for col in MANIFEST_REQUIRED_COLS:
            assert col in row and row[col] != "", f"Column {col} missing/empty"

def test_manifest_deterministic_ordering(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    item_id = "scene_mord"
    make_scene(Path(proc_str), item_id, width=112, height=112)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    with open(Path(tiles_str) / item_id / "manifest.csv") as f:
        rows = list(csv.DictReader(f))

    tile_ids = [r["tile_id"] for r in rows]
    assert tile_ids == sorted(tile_ids)

def test_manifest_split_unknown(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    item_id = "scene_split"
    make_scene(Path(proc_str), item_id, width=112, height=112)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    with open(Path(tiles_str) / item_id / "manifest.csv") as f:
        for row in csv.DictReader(f):
            assert row["split"] == "unknown"

def test_manifest_correct_values(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    item_id = "scene_mvals"
    make_scene(Path(proc_str), item_id, width=112, height=112)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    with open(Path(tiles_str) / item_id / "manifest.csv") as f:
        rows = {r["tile_id"]: r for r in csv.DictReader(f)}

    r00 = rows["tile_r000_c000"]
    assert r00["row"] == "0" and r00["column"] == "0"
    assert r00["x_offset"] == "0" and r00["y_offset"] == "0"
    assert r00["width"] == "64" and r00["height"] == "64"
    assert r00["crs"] == "EPSG:32643"
    r01 = rows["tile_r000_c001"]
    assert r01["x_offset"] == "56"  # stride

# ─── 12. Determinism ─────────────────────────────────────────────────────────

def test_determinism_full(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    item_id = "scene_det"
    make_scene(Path(proc_str), item_id, width=112, height=112)

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=str(out1), config_path=cfg_str)
    create_tiles(item_id, processed_dir=proc_str, tiles_dir=str(out2), config_path=cfg_str)

    # Manifests identical
    m1 = (out1 / item_id / "manifest.csv").read_text()
    m2 = (out2 / item_id / "manifest.csv").read_text()
    assert m1 == m2

    # All tile metadata and pixel data identical
    for td in sorted((out1 / item_id).glob("tile_*")):
        j1 = (td / "metadata.json").read_text()
        j2 = (out2 / item_id / td.name / "metadata.json").read_text()
        assert j1 == j2, f"metadata.json differs in {td.name}"

        for fname in ["B02.tif", "B03.tif", "B04.tif", "B08.tif", "SCL.tif", "cloud_mask.tif"]:
            with rasterio.open(td / fname) as s1, rasterio.open(out2 / item_id / td.name / fname) as s2:
                d1 = s1.read(1)
                d2 = s2.read(1)
                np.testing.assert_array_equal(d1, d2, err_msg=f"{fname} differs in {td.name}")

# ─── 13. Source protection ────────────────────────────────────────────────────

def test_source_protection(tmp_path, base):
    proc_str, tiles_str, cfg_str = base
    item_id = "scene_prot"
    make_scene(Path(proc_str), item_id, width=112, height=112)

    src_files = list((Path(proc_str) / item_id).glob("*"))
    # Compute checksums before
    def chksum(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    before = {f: chksum(f) for f in src_files}

    create_tiles(item_id, processed_dir=proc_str, tiles_dir=tiles_str, config_path=cfg_str)

    for f in src_files:
        assert chksum(f) == before[f], f"{f.name} was modified by tile_scene"
