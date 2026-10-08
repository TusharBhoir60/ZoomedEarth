import os
import tempfile
import torch
import math
from pathlib import Path
import pytest
import numpy as np

from src.train.finetune_gan import (
    downsample_box_4x4_tensor,
    PatchGANDiscriminator,
    load_config,
    set_seed,
    WaldDataset
)
from src.infer.run_sen2sr import permute_bgrn_to_rgbn, permute_rgbn_to_bgrn

def test_band_order_handling():
    # BGRN: [B02, B03, B04, B08]
    # RGBN: [B04, B03, B02, B08] (indices 2, 1, 0, 3)
    bgrn = torch.tensor([1, 2, 3, 4]).view(1, 4, 1, 1).float()
    
    rgbn = permute_bgrn_to_rgbn(bgrn)
    assert rgbn.shape == (1, 4, 1, 1)
    assert rgbn[0, 0, 0, 0].item() == 3 # B04
    assert rgbn[0, 1, 0, 0].item() == 2 # B03
    assert rgbn[0, 2, 0, 0].item() == 1 # B02
    assert rgbn[0, 3, 0, 0].item() == 4 # B08
    
    recovered = permute_rgbn_to_bgrn(rgbn)
    assert torch.all(recovered == bgrn)

def test_consistency_loss_shape_behavior():
    # HR is 256x256, SR is 256x256, LR is 64x64
    sr = torch.ones(2, 4, 256, 256)
    lr = downsample_box_4x4_tensor(sr)
    assert lr.shape == (2, 4, 64, 64)
    assert torch.all(lr == 1.0)

def test_deterministic_seed():
    set_seed(42)
    val1 = torch.rand(1).item()
    set_seed(42)
    val2 = torch.rand(1).item()
    assert math.isclose(val1, val2, rel_tol=1e-5)

def test_config_loading_and_loss_weights(tmp_path):
    cfg_content = """
loss_weights:
  l1: 1.0
  perceptual: 0.01
  adversarial: 0.005
  consistency: 0.1
    """
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(cfg_content)
    
    cfg = load_config(str(cfg_file))
    assert cfg["loss_weights"]["l1"] == 1.0
    assert cfg["loss_weights"]["perceptual"] == 0.01

def test_nan_inf_detection():
    import torch.nn as nn
    l1 = nn.L1Loss()
    a = torch.tensor([float('nan')])
    b = torch.tensor([1.0])
    loss = l1(a, b)
    assert torch.isnan(loss)

def test_checkpoint_save_load(tmp_path):
    from src.infer.run_sen2sr import load_sen2sr_lite_model
    import torch.optim as optim
    
    device = torch.device("cpu")
    # Load model (mocked or real)
    try:
        model = load_sen2sr_lite_model("models/SEN2SRLite", device=device)
    except FileNotFoundError:
        pytest.skip("SEN2SRLite weights not found, skipping checkpoint test.")
        
    opt = optim.Adam(model.sr_model.parameters(), lr=1e-4)
    
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({
        "step": 500,
        "generator": model.sr_model.state_dict(),
        "opt_g": opt.state_dict()
    }, ckpt_path)
    
    loaded = torch.load(ckpt_path)
    assert loaded["step"] == 500
    model.sr_model.load_state_dict(loaded["generator"])
    opt.load_state_dict(loaded["opt_g"])

def test_dataset_rejection_behavior(tmp_path, monkeypatch):
    lr_dir = tmp_path / "lr"
    hr_dir = tmp_path / "hr"
    lr_dir.mkdir()
    hr_dir.mkdir()
    
    import rasterio
    import numpy as np
    
    # Mock rasterio.open to just return a dummy object with shape
    class MockRasterioSrc:
        def __init__(self, shape):
            self.shape = shape
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        def read(self, *args, **kwargs):
            arr = np.zeros((64, 64), dtype=np.float32)
            arr[0, 0] = np.nan # inject NaN
            return arr
            
    def mock_open(*args, **kwargs):
        return MockRasterioSrc((2745, 2745))
        
    monkeypatch.setattr(rasterio, "open", mock_open)
    
    ds = WaldDataset(str(lr_dir), str(hr_dir))
    
    with pytest.raises(RuntimeError, match="Failed to load a valid patch"):
        ds[0]

