import torch
from typing import List, Tuple

def augment(x: torch.Tensor, i: int) -> torch.Tensor:
    """
    Applies the i-th dihedral transformation to the last two dimensions (H, W) of x.
    i must be in [0, 7].
    Does not mutate the input tensor.

    Args:
        x: Input tensor of shape (..., H, W).
        i: Integer index in [0, 7].

    Returns:
        Augmented tensor of shape (..., H, W) or (..., W, H) depending on the transform.
    """
    if i == 0:
        return x.clone()
    elif i == 1:
        return torch.rot90(x, k=1, dims=(-2, -1))
    elif i == 2:
        return torch.rot90(x, k=2, dims=(-2, -1))
    elif i == 3:
        return torch.rot90(x, k=3, dims=(-2, -1))
    elif i == 4:
        return torch.flip(x, dims=[-1])  # Flip Horizontal (left-right)
    elif i == 5:
        return torch.flip(x, dims=[-2])  # Flip Vertical (up-down)
    elif i == 6:
        return torch.transpose(x, -2, -1).clone()  # Transpose main diagonal
    elif i == 7:
        # Transpose anti-diagonal: swap H/W then flip both
        return torch.flip(torch.transpose(x, -2, -1), dims=[-2, -1])
    else:
        raise ValueError(f"Transformation index {i} must be in [0, 7]")


def inverse_augment(x: torch.Tensor, i: int) -> torch.Tensor:
    """
    Applies the exact inverse of the i-th dihedral transformation.

    Args:
        x: Transformed tensor of shape (..., H, W) or (..., W, H).
        i: Integer index in [0, 7].

    Returns:
        Tensor exactly recovered to the original shape and orientation.
    """
    if i == 0:
        return x.clone()
    elif i == 1:
        return torch.rot90(x, k=-1, dims=(-2, -1))
    elif i == 2:
        return torch.rot90(x, k=-2, dims=(-2, -1))
    elif i == 3:
        return torch.rot90(x, k=-3, dims=(-2, -1))
    elif i == 4:
        return torch.flip(x, dims=[-1])
    elif i == 5:
        return torch.flip(x, dims=[-2])
    elif i == 6:
        return torch.transpose(x, -2, -1).clone()
    elif i == 7:
        return torch.flip(torch.transpose(x, -2, -1), dims=[-2, -1])
    else:
        raise ValueError(f"Transformation index {i} must be in [0, 7]")


def aggregate_tta(predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregates a list of exactly 8 aligned predictions.
    
    Args:
        predictions: List of 8 torch.Tensors, all of identical shape (..., H, W).
                     Typically (C, H, W) or (B, C, H, W).
                     
    Returns:
        (mean_tensor, std_tensor) where std_tensor is the population standard deviation.
        Returned tensors have the same shape as the input elements.
        
    Raises:
        ValueError if length of predictions is not 8, or shapes mismatch.
    """
    if len(predictions) != 8:
        raise ValueError(f"Expected exactly 8 predictions for Dihedral TTA, got {len(predictions)}")
        
    base_shape = predictions[0].shape
    for idx, p in enumerate(predictions):
        if p.shape != base_shape:
            raise ValueError(f"Shape mismatch at index {idx}: expected {base_shape}, got {p.shape}")
            
    # Stack along a new dimension at the beginning
    stacked = torch.stack(predictions, dim=0)
    
    # Calculate mean and population standard deviation (unbiased=False)
    # TTA forms the entire augmented population space, so population std is correct.
    mean_tensor = torch.mean(stacked, dim=0)
    std_tensor = torch.std(stacked, dim=0, unbiased=False)
    
    return mean_tensor, std_tensor
