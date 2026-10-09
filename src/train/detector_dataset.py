import hashlib
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset


class DetectorDataset(Dataset):
    """
    Dataset for D017 Proxy-BCE and D018 Mixed-Domain training experiments.
    
    This strictly implements:
    - Bounded-memory out-of-core reads.
    - 50/50 deterministic sampling of SEN2SR and Bicubic patches.
    - Proxy-BCE label transformation (255 -> 0) IN MEMORY ONLY.
    - Preservation of canonical unknown and invalid-imagery masks.
    """
    
    def __init__(self, 
                 bicubic_path: str, 
                 sen2sr_path: str, 
                 label_path: str, 
                 cloud_mask_path: str, 
                 patch_size: int = 256, 
                 seed: int = 42):
        self.bicubic_path = bicubic_path
        self.sen2sr_path = sen2sr_path
        self.label_path = label_path
        self.cloud_mask_path = cloud_mask_path
        self.patch_size = patch_size
        self.seed = seed
        
        if self.patch_size % 4 != 0:
            raise ValueError("patch_size must be a multiple of 4 to perfectly align with the 10m cloud mask.")
            
        self._verify_contracts()
        
        self.n_rows = self.height // self.patch_size
        self.n_cols = self.width // self.patch_size
        self.length = self.n_rows * self.n_cols

    def _verify_contracts(self):
        # Explicitly reject validation/test AOIs if they slip in by path name
        if "pune" in str(self.bicubic_path).lower() or "mumbai" in str(self.bicubic_path).lower():
            raise ValueError("Pune and Mumbai are prohibited from training dataset instantiation.")
            
        with rasterio.open(self.bicubic_path) as b, \
             rasterio.open(self.sen2sr_path) as s, \
             rasterio.open(self.label_path) as l, \
             rasterio.open(self.cloud_mask_path) as c:
                 
            # 1. Image and label geometries must perfectly match
            if b.shape != s.shape or b.shape != l.shape:
                raise ValueError(f"Geometry mismatch: Bicubic {b.shape}, SEN2SR {s.shape}, Labels {l.shape}")
            if b.transform != s.transform or b.transform != l.transform:
                raise ValueError("Transform mismatch between imagery and labels.")
            if b.crs != s.crs or b.crs != l.crs:
                raise ValueError("CRS mismatch between imagery and labels.")
                
            # 2. Cloud mask must be exactly 10m (4x 2.5m)
            if c.transform.a != b.transform.a * 4 or c.transform.e != b.transform.e * 4:
                raise ValueError("Cloud mask does not have the expected 10m affine transform relationship to the 2.5m grid.")
            if c.shape[0] * 4 < b.shape[0] or c.shape[1] * 4 < b.shape[1]:
                raise ValueError("Cloud mask dimensions are too small for the 2.5m grid.")
                
            # 3. Channel counts and dtypes
            if b.count != 4 or s.count != 4:
                raise ValueError("Imagery must have exactly 4 channels.")
            if b.dtypes[0] != 'float32' or s.dtypes[0] != 'float32':
                raise ValueError("Imagery must be float32.")
            if l.dtypes[0] != 'uint8':
                raise ValueError("Labels must be uint8.")
                
            # 4. Canonical Band Order (if descriptions exist, enforce B02 first)
            if b.descriptions and b.descriptions[0] and b.descriptions[0] != 'B02':
                raise ValueError(f"Expected B02 as first band, got {b.descriptions[0]}")
                
            self.height, self.width = b.shape

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError("Index out of bounds")
            
        # Deterministic 50/50 sampling based on index and seed
        hash_input = f"{self.seed}_{idx}".encode('utf-8')
        hex_hash = hashlib.md5(hash_input).hexdigest()
        use_sen2sr = (int(hex_hash, 16) % 2) == 1
        
        img_path = self.sen2sr_path if use_sen2sr else self.bicubic_path
        modality = "sen2sr" if use_sen2sr else "bicubic"
        
        row_idx = idx // self.n_cols
        col_idx = idx % self.n_cols
        
        row_off = row_idx * self.patch_size
        col_off = col_idx * self.patch_size
        
        window = Window(col_off, row_off, self.patch_size, self.patch_size)
        c_window = Window(col_off // 4, row_off // 4, self.patch_size // 4, self.patch_size // 4)
        
        with rasterio.open(img_path) as src:
            image = src.read(window=window)  # (4, H, W)
            
        with rasterio.open(self.label_path) as l_src:
            labels_raw = l_src.read(1, window=window)  # (H, W)
            
        with rasterio.open(self.cloud_mask_path) as c_src:
            cmask_10m = c_src.read(1, window=c_window)
            
        # Nearest-neighbor upsample of 10m cloud mask to 2.5m
        cmask = np.kron(cmask_10m, np.ones((4, 4), dtype=cmask_10m.dtype))
        
        # Valid imagery mask (0 = valid, 1 = cloud/invalid in original L2A contract)
        # NaN in imagery is also invalid
        img_nan = np.isnan(image).any(axis=0)
        valid_imagery_mask = (cmask == 0) & (~img_nan)
        
        # D017 Proxy-BCE Transformation
        # Canonical: 1 = positive, 0 = verified negative, 255 = unknown.
        # Proxy: 255 mapped to 0 IN MEMORY ONLY.
        unknown_mask = (labels_raw == 255)
        
        proxy_labels = np.copy(labels_raw)
        proxy_labels[unknown_mask] = 0
        proxy_labels = proxy_labels.astype(np.float32)
        
        # Replace NaNs in image with 0.0 purely to prevent PyTorch NaN propagation.
        # The valid_imagery_mask must be used by the loss function to ignore these pixels.
        image = np.nan_to_num(image, nan=0.0).astype(np.float32)
        
        return {
            "image": torch.from_numpy(image),
            "proxy_labels": torch.from_numpy(proxy_labels),
            "unknown_mask": torch.from_numpy(unknown_mask),
            "valid_imagery_mask": torch.from_numpy(valid_imagery_mask),
            "modality": modality,
            "window_coords": (col_off, row_off)
        }
