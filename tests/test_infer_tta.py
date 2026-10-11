import pytest
import numpy as np
import rasterio
from pathlib import Path
from unittest.mock import MagicMock
import torch

from src.infer.infer_tile import infer_tile
from tests.test_infer_tile import make_tile

def make_mock_model(output_val: float = 0.5):
    def _forward(tensor):
        b, c, h, w = tensor.shape
        out_h = h * 4
        out_w = w * 4
        data = torch.full((c, out_h, out_w), output_val)
        return data.unsqueeze(0)
    mock = MagicMock()
    mock.side_effect = _forward
    return mock

def test_default_cli_behavior(tmp_path):
    tile_dir = make_tile(tmp_path)
    mock_model = make_mock_model(output_val=0.42)
    meta = infer_tile(tile_dir=tile_dir, output_dir=tmp_path / "out", model=mock_model)
    
    assert "uncertainty.tif" not in [f.name for f in (tmp_path / "out").glob("*.tif")]
    assert meta.get("uncertainty_mode") == "none"

def test_tta_invalid_mode_rejected(tmp_path):
    tile_dir = make_tile(tmp_path)
    mock_model = make_mock_model()
    with pytest.raises(ValueError, match="Unsupported uncertainty mode: invalid"):
        infer_tile(tile_dir=tile_dir, output_dir=tmp_path / "out", model=mock_model, uncertainty="invalid")

def test_tta_execution_and_output(tmp_path):
    tile_dir = make_tile(tmp_path)
    mock_model = make_mock_model(output_val=0.42)
    meta = infer_tile(tile_dir=tile_dir, output_dir=tmp_path / "out", model=mock_model, uncertainty="tta")
    
    # 1. Check primary SEN2SR.tif remains the same
    with rasterio.open(tmp_path / "out" / "SEN2SR.tif") as src:
        sen2sr_arr = src.read()
        assert np.allclose(sen2sr_arr, 0.42)
        assert src.count == 4
    
    # 2. Check uncertainty.tif
    unc_path = tmp_path / "out" / "uncertainty.tif"
    assert unc_path.exists()
    with rasterio.open(unc_path) as src:
        unc_arr = src.read()
        assert np.allclose(unc_arr, 0.0)
        assert src.count == 4
        assert src.dtypes[0] == "float32"
        assert list(src.descriptions) == ["B02", "B03", "B04", "B08"]

    assert meta.get("uncertainty_mode") == "tta"

def test_tta_rectangular_input_fails(tmp_path):
    tile_dir = make_tile(tmp_path, width=200, height=250)
    mock_model = make_mock_model()
    with pytest.raises(ValueError, match="unsupported by upstream"):
        infer_tile(tile_dir=tile_dir, output_dir=tmp_path / "out", model=mock_model, uncertainty="tta")

def test_tta_nan_mask_propagation(tmp_path):
    tile_dir = make_tile(tmp_path)
    def _forward(tensor):
        b, c, h, w = tensor.shape
        data = torch.full((c, h * 4, w * 4), float("nan"))
        return data.unsqueeze(0)
    
    mock_model = MagicMock(side_effect=_forward)
    
    with pytest.raises(ValueError, match="NaN"):
        infer_tile(tile_dir=tile_dir, output_dir=tmp_path / "out", model=mock_model, uncertainty="tta")

def test_tta_existing_output_raises(tmp_path):
    tile_dir = make_tile(tmp_path)
    mock_model = make_mock_model(output_val=0.42)
    
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    unc_path = output_dir / "uncertainty.tif"
    with open(unc_path, "w") as f:
        f.write("mock")
        
    with pytest.raises(FileExistsError, match="Uncertainty raster already exists"):
        infer_tile(tile_dir=tile_dir, output_dir=output_dir, model=mock_model, uncertainty="tta")

def test_tta_failure_leaves_no_partial_files(tmp_path):
    tile_dir = make_tile(tmp_path)
    
    def _failing_forward(tensor):
        # We fail halfway to simulate inference crash during TTA
        # but the primary output shouldn't have failed yet if it's the second pass?
        # Actually, primary output executes first, then TTA passes.
        # If we raise an exception during a TTA pass, it shouldn't leave an `uncertainty.tif.tmp`.
        pass
        
    # We will just patch rasterio.open to fail during writing uncertainty
    mock_model = make_mock_model(output_val=0.42)
    output_dir = tmp_path / "out"
    
    import unittest.mock as mock
    original_open = rasterio.open
    def mock_open(*args, **kwargs):
        if "uncertainty.tif.tmp" in str(args[0]):
            raise RuntimeError("Disk full")
        return original_open(*args, **kwargs)
        
    with mock.patch("rasterio.open", side_effect=mock_open):
        with pytest.raises(RuntimeError, match="Disk full"):
            infer_tile(tile_dir=tile_dir, output_dir=output_dir, model=mock_model, uncertainty="tta")
            
    # Check that tmp file is NOT left behind, or if it is, the main file doesn't exist.
    # Actually, if rasterio.open fails, the file might not be created or might be empty.
    # The requirement is "Failures during inference, aggregation, writing, or validation cannot leave a partial file that looks like a valid product."
    # Since we write to a `.tmp` file and rename, `uncertainty.tif` is definitely not there!
    assert not (output_dir / "uncertainty.tif").exists()
