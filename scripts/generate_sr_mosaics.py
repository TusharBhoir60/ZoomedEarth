import argparse
import logging
import pathlib

from src.ingest.tile_scene import create_tiles
from src.infer.infer_tile import infer_tile
from src.infer.stitch import stitch_scene
from src.infer.bicubic import run_bicubic

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("generate_sr_mosaics")

ALLOWED_AOIS = ["mosaic_aoi_delhi_urban", "mosaic_aoi_pune_peri_urban", "mosaic_aoi_mumbai_urban"]

def generate_bicubic(aoi_id: str, overwrite: bool = False):
    processed_dir = pathlib.Path("data/processed/sentinel2")
    input_path = processed_dir / aoi_id
    output_path = pathlib.Path("data/outputs/sen2sr") / aoi_id / "Bicubic_scene.tif"
    
    if output_path.exists() and not overwrite:
        logger.info(f"[{aoi_id}] Bicubic output already exists, skipping.")
        return
        
    logger.info(f"[{aoi_id}] Running bounded-memory Bicubic upsampling...")
    run_bicubic(
        input_path=input_path,
        output_path=output_path,
        band_order=["B02", "B03", "B04", "B08"],
        scale=4.0
    )
    logger.info(f"[{aoi_id}] Bicubic complete: {output_path}")

def generate_sen2sr(aoi_id: str, overwrite: bool = False):
    output_path = pathlib.Path("data/outputs/sen2sr") / aoi_id / "SEN2SR_scene.tif"
    if output_path.exists() and not overwrite:
        logger.info(f"[{aoi_id}] SEN2SR output already exists, skipping.")
        return

    logger.info(f"[{aoi_id}] Extracting 10m tiles for SEN2SR...")
    create_tiles(item_id=aoi_id)
    
    tiles_dir = pathlib.Path("data/tiles/sentinel2") / aoi_id
    tile_dirs = list(tiles_dir.glob("tile_*"))
    if not tile_dirs:
        raise ValueError(f"No tiles generated for {aoi_id}")
        
    logger.info(f"[{aoi_id}] Running SEN2SR inference on {len(tile_dirs)} tiles...")
    for i, t_dir in enumerate(tile_dirs):
        out_tile = pathlib.Path("data/outputs/sen2sr") / aoi_id / t_dir.name / "SEN2SR.tif"
        if out_tile.exists() and not overwrite:
            continue
            
        infer_tile(
            tile_dir=str(t_dir),
            output_dir=None,
            weights_dir="models/SEN2SRLite",
            device=None
        )
        if (i + 1) % 50 == 0:
            logger.info(f"[{aoi_id}] Processed {i+1}/{len(tile_dirs)} tiles...")
            
    logger.info(f"[{aoi_id}] Stitching SEN2SR tiles...")
    stitch_scene(item_id=aoi_id)
    logger.info(f"[{aoi_id}] SEN2SR stitching complete: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Bicubic and SEN2SR mosaics.")
    parser.add_argument("--aoi", nargs="+", help="Target AOI(s). Must be 'mosaic_aoi_delhi_urban' or 'mosaic_aoi_pune_peri_urban'.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing mosaics")
    parser.add_argument("--bicubic-only", action="store_true", help="Only generate Bicubic mosaics")
    parser.add_argument("--sen2sr-only", action="store_true", help="Only generate SEN2SR mosaics")
    
    args = parser.parse_args()
    
    if not args.aoi:
        targets = ALLOWED_AOIS
    else:
        targets = args.aoi
        
    for aoi_id in targets:
        if aoi_id not in ALLOWED_AOIS:
            logger.error(f"AOI '{aoi_id}' is not permitted. Allowed AOIs are: {ALLOWED_AOIS}")
            exit(1)
            
    for aoi_id in targets:
        logger.info(f"=== Processing {aoi_id} ===")
        if not args.sen2sr_only:
            generate_bicubic(aoi_id, overwrite=args.overwrite)
        if not args.bicubic_only:
            generate_sen2sr(aoi_id, overwrite=args.overwrite)
            
    logger.info("=== All generation tasks complete ===")

if __name__ == "__main__":
    main()
