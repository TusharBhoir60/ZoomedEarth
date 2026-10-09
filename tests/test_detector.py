import torch
import pytest
import os
import tempfile
import pathlib
import numpy as np
import random

from src.models.detector import BuildingDetectorUNet
from src.train.train_detector import proxy_bce_loss, evaluate_pune_tier_c, train_model

def test_model_shapes_and_parameters():
    model = BuildingDetectorUNet(in_channels=4, out_channels=1)
    x = torch.randn(2, 4, 128, 128)
    out = model(x)
    assert out.shape == (2, 1, 128, 128)

def test_finite_loss_and_valid_pixels():
    model = BuildingDetectorUNet()
    x = torch.randn(2, 4, 128, 128)
    proxy_labels = torch.randint(0, 2, (2, 128, 128)).float()
    valid_mask = torch.zeros(2, 128, 128, dtype=torch.bool)
    valid_mask[:, 10:20, 10:20] = True
    
    logits = model(x)
    loss = proxy_bce_loss(logits, proxy_labels, valid_mask)
    assert torch.isfinite(loss)
    loss.backward()

def test_zero_valid_pixels():
    model = BuildingDetectorUNet()
    x = torch.randn(1, 4, 128, 128)
    proxy_labels = torch.zeros(1, 128, 128)
    valid_mask = torch.zeros(1, 128, 128, dtype=torch.bool)
    
    logits = model(x)
    loss = proxy_bce_loss(logits, proxy_labels, valid_mask)
    assert loss.item() == 0.0

@pytest.fixture
def mock_tier_c_artifacts(tmp_path):
    import rasterio
    import json
    from rasterio.transform import Affine
    
    b_path = tmp_path / "mock_input.tif"
    s_path = tmp_path / "mock_sen2sr.tif"
    l_path = tmp_path / "labels.tif"
    c_path = tmp_path / "cloud.tif"
    budget_path = tmp_path / "budget.json"
    
    transform = Affine(2.5, 0.0, 0.0, 0.0, -2.5, 1000.0)
    c_transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 1000.0)
    
    with rasterio.open(b_path, "w", driver="GTiff", height=256, width=256, count=4, dtype="float32", transform=transform, crs="EPSG:32643") as dst:
        dst.write(np.zeros((4, 256, 256), dtype="float32"))
    with rasterio.open(s_path, "w", driver="GTiff", height=256, width=256, count=4, dtype="float32", transform=transform, crs="EPSG:32643") as dst:
        dst.write(np.zeros((4, 256, 256), dtype="float32"))
        
    labels = np.zeros((1, 256, 256), dtype="uint8")
    labels[0, :64, :64] = 1
    with rasterio.open(l_path, "w", driver="GTiff", height=256, width=256, count=1, dtype="uint8", transform=transform, crs="EPSG:32643") as dst:
        dst.write(labels)
        
    with rasterio.open(c_path, "w", driver="GTiff", height=64, width=64, count=1, dtype="uint8", transform=c_transform, crs="EPSG:32643") as dst:
        dst.write(np.zeros((1, 64, 64), dtype="uint8"))
        
    with open(budget_path, "w") as f:
        json.dump({"arithmetic_mean_budget": 0.1, "ratios": {}}, f)
        
    return b_path, s_path, l_path, c_path, budget_path

def test_zero_pixel_batch_skips_optimizer_step(mock_tier_c_artifacts, tmp_path, monkeypatch):
    b_path, s_path, l_path, c_path, budget_path = mock_tier_c_artifacts
    class Config:
        pass
    config = Config()
    config.train_bicubic = str(b_path)
    config.train_sen2sr = str(s_path)
    config.train_labels = str(l_path)
    config.train_cloud_mask = str(c_path)
    config.val_sen2sr = str(s_path)
    config.val_bicubic = str(b_path)
    config.val_labels = str(l_path)
    config.val_cloud_mask = str(c_path)
    config.budget_path = str(budget_path)
    config.checkpoint_dir = str(tmp_path)
    config.epochs = 1
    config.batch_size = 1
    config.lr = 1e-3
    config.patch_size = 256
    config.seed = 42
    config.num_workers = 0
    config.val_cadence = 1
    config.resume_from = None
    
    step_calls = 0
    
    from torch.optim import AdamW
    original_step = AdamW.step
    def mock_step(self, *args, **kwargs):
        nonlocal step_calls
        step_calls += 1
        original_step(self, *args, **kwargs)
        
    monkeypatch.setattr(AdamW, "step", mock_step)
    original_iter = torch.utils.data.DataLoader.__iter__
    def mock_iter(self):
        it = original_iter(self)
        for batch in it:
            batch['valid_imagery_mask'][:] = False
            yield batch
            
    monkeypatch.setattr(torch.utils.data.DataLoader, "__iter__", mock_iter)
    train_model(config)
    assert step_calls == 0

def test_separated_checkpoints_and_resume(mock_tier_c_artifacts, tmp_path, monkeypatch):
    b_path, s_path, l_path, c_path, budget_path = mock_tier_c_artifacts
    class Config:
        pass
    config = Config()
    config.train_bicubic = str(b_path)
    config.train_sen2sr = str(s_path)
    config.train_labels = str(l_path)
    config.train_cloud_mask = str(c_path)
    config.val_sen2sr = str(s_path)
    config.val_bicubic = str(b_path)
    config.val_labels = str(l_path)
    config.val_cloud_mask = str(c_path)
    config.budget_path = str(budget_path)
    config.checkpoint_dir = str(tmp_path)
    config.epochs = 3
    config.batch_size = 1
    config.lr = 1e-3
    config.patch_size = 256
    config.seed = 42
    config.num_workers = 0
    config.val_cadence = 1
    config.resume_from = None

    # Epoch 0: SEN2SR = 0.5 (improves best, saves best_model.pt)
    # Epoch 1: SEN2SR = 0.4 (does NOT improve best, best_model.pt not updated, last_checkpoint.pt updated)
    # Epoch 2: SEN2SR = invalid (raises Exception, does not overwrite best)
    scores = [
        {"recall": 0.5}, {"recall": 0.3}, # Epoch 0
        {"recall": 0.4}, {"recall": 0.2}, # Epoch 1
        Exception("Invalid score"), {"recall": 0.1} # Epoch 2
    ]
    
    def mock_score_tier_c(pred_path, label_path, cloud_mask_path, budget_path, aoi_id):
        res = scores.pop(0)
        if isinstance(res, Exception):
            raise res
        return res
        
    monkeypatch.setattr("src.train.train_detector.score_tier_c", mock_score_tier_c)
    
    train_model(config)
    
    best_ckpt = tmp_path / "best_model.pt"
    last_ckpt = tmp_path / "last_checkpoint.pt"
    
    assert best_ckpt.exists()
    assert last_ckpt.exists()
    
    best_data = torch.load(best_ckpt)
    assert best_data['epoch'] == 0
    assert best_data['best_val_score'] == 0.5
    
    last_data = torch.load(last_ckpt)
    assert last_data['epoch'] == 2 # 3 epochs complete (0, 1, 2)
    assert last_data['best_val_score'] == 0.5 # Best score persists through non-improving epochs
    
    # Check RNG states
    assert 'python_rng' in last_data
    assert 'optimizer_state' in last_data
    
    # Verify that trying to resume from best_model fails due to lack of optimizer
    config.epochs = 4
    config.resume_from = str(best_ckpt)
    with pytest.raises(ValueError, match="Attempted to resume from a checkpoint lacking optimizer state"):
        train_model(config)
        
    # Verify that resuming from last_checkpoint works and starts at epoch 3
    config.resume_from = str(last_ckpt)
    
    # Reset scores explicitly for resumed training
    scores.clear()
    scores.extend([{"recall": 0.6}, {"recall": 0.5}])
    
    train_model(config)
    
    last_data_resumed = torch.load(last_ckpt)
    assert last_data_resumed['epoch'] == 3
    assert last_data_resumed['best_val_score'] == 0.6
    assert last_data_resumed['best_val_score'] == 0.6
    
    best_data_resumed = torch.load(best_ckpt)
    assert best_data_resumed['epoch'] == 3
    assert best_data_resumed['best_val_score'] == 0.6

from src.train.train_detector import generate_validation_predictions
from rasterio.transform import Affine

def test_generate_validation_predictions_padding(tmp_path):
    import numpy as np
    import rasterio
    import torch
    from src.models.detector import BuildingDetectorUNet
    
    input_path = tmp_path / "odd_input.tif"
    output_path = tmp_path / "odd_output.tif"
    
    data = np.random.randn(4, 20, 25).astype(np.float32)
    # Set one pixel to NaN to test NaN propagation
    data[:, 10, 13] = np.nan
    
    transform = Affine(2.5, 0.0, 100.0, 0.0, -2.5, 200.0)
    crs = "EPSG:32643"
    
    with rasterio.open(
        input_path, 'w',
        driver='GTiff',
        height=20, width=25,
        count=4,
        dtype='float32',
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(data)
        
    model = BuildingDetectorUNet()
    
    # patch_size=16 so we get the edge tiles and satisfy GDAL block rules
    generate_validation_predictions(model, str(input_path), str(output_path), torch.device('cpu'), patch_size=16)
    
    assert output_path.exists()
    
    with rasterio.open(output_path) as src:
        assert src.width == 25
        assert src.height == 20
        assert src.count == 1
        assert src.dtypes[0] == 'float32'
        assert src.crs.to_string() == crs
        assert src.transform == transform
        
        out_data = src.read(1)
        assert out_data.shape == (20, 25)
        
        # Verify NaN is propagated exactly at (10, 13)
        assert np.isnan(out_data[10, 13])
        # Ensure other pixels are valid floats
        out_data[10, 13] = 0.0
        assert np.isfinite(out_data).all()

def test_generate_validation_predictions_coverage(tmp_path):
    import numpy as np
    import rasterio
    import torch
    from src.train.train_detector import generate_validation_predictions
    from rasterio.transform import Affine
    
    input_path = tmp_path / "cov_input.tif"
    output_path = tmp_path / "cov_output.tif"
    
    # Create input raster
    data = np.ones((4, 15, 17), dtype=np.float32)
    transform = Affine(2.5, 0.0, 100.0, 0.0, -2.5, 200.0)
    crs = "EPSG:32643"
    
    with rasterio.open(
        input_path, 'w',
        driver='GTiff',
        height=15, width=17,
        count=4,
        dtype='float32',
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(data)
        
    class MockModel(torch.nn.Module):
        def forward(self, x):
            # Return exactly 1.0 for every pixel
            return torch.ones((1, 1, x.shape[2], x.shape[3]), dtype=torch.float32).to(x.device)
            
    model = MockModel()
    
    generate_validation_predictions(model, str(input_path), str(output_path), torch.device('cpu'), patch_size=16)
    
    with rasterio.open(output_path) as src:
        out_data = src.read(1)
        # Verify every pixel was written exactly once (value is 1.0)
        assert np.allclose(out_data, 1.0)
        assert out_data.shape == (15, 17)
