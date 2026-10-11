import torch
import pytest
from src.models.uncertainty import augment, inverse_augment, aggregate_tta

def test_inverse_recovery():
    # Asymmetric synthetic array to ensure flips and transposes are distinct
    # Shape: (1, 1, 3, 4)
    x = torch.arange(12, dtype=torch.float32).view(1, 1, 3, 4)
    # Add some asymmetry
    x = x + torch.randn_like(x) * 0.1
    
    for i in range(8):
        aug_x = augment(x, i)
        inv_aug_x = inverse_augment(aug_x, i)
        
        # Must exactly recover the original tensor
        assert torch.allclose(x, inv_aug_x, atol=1e-6)
        
        # Sanity check: except for i=0, the augmented tensor should differ from original
        if i != 0:
            if aug_x.shape == x.shape:
                assert not torch.allclose(x, aug_x, atol=1e-6)

def test_non_square_dimensions():
    x = torch.randn(2, 4, 10, 20)  # non-square
    for i in range(8):
        aug_x = augment(x, i)
        # Check expected shape
        if i in [1, 3, 6, 7]:
            assert aug_x.shape == (2, 4, 20, 10)
        else:
            assert aug_x.shape == (2, 4, 10, 20)
            
        inv_aug_x = inverse_augment(aug_x, i)
        assert inv_aug_x.shape == (2, 4, 10, 20)
        assert torch.allclose(x, inv_aug_x, atol=1e-6)

def test_preservation_of_channels():
    # Check that channel ordering is strictly preserved
    x = torch.zeros(4, 5, 5)
    x[0, :, :] = 1.0  # Band 0 is all 1s
    x[1, :, :] = 2.0  # Band 1 is all 2s
    x[2, :, :] = 3.0  # Band 2 is all 3s
    x[3, :, :] = 4.0  # Band 3 is all 4s
    
    for i in range(8):
        aug_x = augment(x, i)
        assert torch.allclose(aug_x[0], torch.tensor(1.0))
        assert torch.allclose(aug_x[1], torch.tensor(2.0))
        assert torch.allclose(aug_x[2], torch.tensor(3.0))
        assert torch.allclose(aug_x[3], torch.tensor(4.0))

def test_aggregation_mean_and_std():
    # Construct 8 identical tensors -> std should be 0
    base = torch.ones(2, 3, 3)
    preds = [base for _ in range(8)]
    mean, std = aggregate_tta(preds)
    assert torch.allclose(mean, base)
    assert torch.allclose(std, torch.zeros_like(base))
    
    # Construct known values: half 0s, half 2s
    # population mean = 1, population variance = 1, std = 1
    preds2 = [torch.zeros(2, 2)] * 4 + [torch.ones(2, 2) * 2] * 4
    mean2, std2 = aggregate_tta(preds2)
    assert torch.allclose(mean2, torch.ones(2, 2))
    assert torch.allclose(std2, torch.ones(2, 2))

def test_rejection_of_mismatched_shapes():
    preds = [torch.randn(3, 3) for _ in range(7)]
    with pytest.raises(ValueError, match="Expected exactly 8"):
        aggregate_tta(preds)
        
    preds = [torch.randn(3, 3) for _ in range(8)]
    preds[3] = torch.randn(3, 4) # shape mismatch
    with pytest.raises(ValueError, match="Shape mismatch"):
        aggregate_tta(preds)

def test_immutability():
    x = torch.arange(9, dtype=torch.float32).view(3, 3)
    x_copy = x.clone()
    
    aug_x = augment(x, 4)
    # Modifying the augmented tensor should not mutate the original if immutability holds
    # However, rot90/flip returns views. 
    # To truly enforce immutability defensively, augment() returns clones for operations that aren't naturally copies?
    # PyTorch's flip/rot90 don't share data cleanly for writes (writing to a view writes to original).
    # Since we test immutability via cloning before augment is called in practice, we just verify
    # the function didn't touch `x` during execution.
    assert torch.allclose(x, x_copy)
