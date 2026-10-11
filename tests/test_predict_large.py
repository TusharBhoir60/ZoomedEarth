import torch
import pytest
from src.infer.predict_large import predict_large_patched

class MockModel(torch.nn.Module):
    def forward(self, x):
        # Mocks 4x SR by repeating pixels
        return x.repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)

class SpatialMockModel(torch.nn.Module):
    def forward(self, x):
        # Mocks 4x SR and adds spatial coordinates to verify seams
        B, C, H, W = x.shape
        out = x.repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)
        
        # Add a ramp based on coordinate position inside the 128x128 chunk
        # This creates distinct values for every pixel in the chunk
        grid_y, grid_x = torch.meshgrid(torch.arange(H*4), torch.arange(W*4), indexing='ij')
        # scale to small float to not overflow
        out = out + (grid_y * 0.001) + (grid_x * 0.0001)
        return out

@pytest.fixture
def mock_model():
    return MockModel()

@pytest.fixture
def spatial_model():
    return SpatialMockModel()

def test_square_standard(mock_model):
    x = torch.ones(4, 256, 256)
    out = predict_large_patched(x, mock_model, overlap=32)
    assert out.shape == (4, 1024, 1024)
    assert (out == 1.0).all(), "Uninitialized regions found"

def test_square_below_threshold(mock_model):
    x = torch.ones(4, 64, 64)
    out = predict_large_patched(x, mock_model, overlap=32)
    assert out.shape == (4, 256, 256)
    assert (out == 1.0).all()

def test_square_at_threshold(mock_model):
    x = torch.ones(4, 128, 128)
    out = predict_large_patched(x, mock_model, overlap=32)
    assert out.shape == (4, 512, 512)
    assert (out == 1.0).all()

def test_rectangular_h_greater_w(mock_model):
    x = torch.ones(4, 300, 200)
    out = predict_large_patched(x, mock_model, overlap=32)
    assert out.shape == (4, 1200, 800)
    assert (out == 1.0).all()

def test_rectangular_w_greater_h(mock_model):
    x = torch.ones(4, 200, 300)
    out = predict_large_patched(x, mock_model, overlap=32)
    assert out.shape == (4, 800, 1200)
    assert (out == 1.0).all()

def test_non_divisible_dimensions(mock_model):
    # Dimensions that don't nicely divide by step (96) or chunk (128)
    x = torch.ones(4, 177, 233)
    out = predict_large_patched(x, mock_model, overlap=32)
    assert out.shape == (4, 177*4, 233*4)
    assert (out == 1.0).all()

def test_spatial_seams(spatial_model):
    # Verify that there are no zeros (uninitialized regions) even with a spatial model
    # The spatial model adds varying values, so if the crop indices are misaligned,
    # we might get zeros or strange artifacts.
    x = torch.ones(4, 200, 200)
    out = predict_large_patched(x, spatial_model, overlap=32)
    assert out.shape == (4, 800, 800)
    assert (out != 0.0).all(), "Uninitialized zero pixels found in output!"

def test_invalid_input_shape(mock_model):
    x = torch.ones(256, 256) # missing channel dim
    with pytest.raises(IndexError):
        predict_large_patched(x, mock_model, overlap=32)

if __name__ == "__main__":
    import sys
    print("Running tests natively...")
    m = MockModel()
    s = SpatialMockModel()
    try:
        test_square_standard(m)
        test_square_below_threshold(m)
        test_square_at_threshold(m)
        test_rectangular_h_greater_w(m)
        test_rectangular_w_greater_h(m)
        test_non_divisible_dimensions(m)
        test_spatial_seams(s)
        test_invalid_input_shape(m)
        print("ALL TESTS PASSED")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

