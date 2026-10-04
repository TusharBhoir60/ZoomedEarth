"""
Canonical Sentinel-2 Preprocessing Contract.

Project: Sentinel-2 Super-Resolution + Trust Harness (SIH26142, NTRO)
"""

import numpy as np

# Canonical repository band order for Sentinel-2
CANONICAL_BANDS = ["B02", "B03", "B04", "B08"]

def resolve_boa_offset(boa_offset_applied, processing_baseline):
    """
    Compute effective BOA offset DN from STAC metadata rules.
    """
    if boa_offset_applied is None:
        raise ValueError("earthsearch:boa_offset_applied is missing.")
    if boa_offset_applied is True:
        return 0.0, "applied_true"
        
    parts = str(processing_baseline).split('.')
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        raise ValueError(f"Invalid processing_baseline format: {processing_baseline}")
        
    if major > 4 or (major == 4 and minor >= 0):
        return -1000.0, "applied_false_pb_ge_0400"
    return 0.0, "applied_false_pb_lt_0400"

def dn_to_reflectance(
    data: np.ndarray,
    boa_add_offset: float,
    quantification_value: float,
    nodata_value: float = 0.0
) -> np.ndarray:
    """Core function to convert DN array (any shape) to reflectance."""
    if boa_add_offset is None or quantification_value is None:
        raise ValueError("boa_add_offset and quantification_value must be explicitly provided.")
        
    if quantification_value <= 0:
        raise ValueError("quantification_value must be strictly positive.")
        
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("Input data must be numeric.")

    if np.any(np.isinf(data)):
        raise ValueError("Input data contains Inf values.")

    data_float = data.astype(np.float32)
    
    if np.isnan(nodata_value):
        nodata_mask = np.isnan(data_float)
    else:
        nodata_mask = (data == nodata_value)
    
    reflectance = (data_float + boa_add_offset) / quantification_value
    reflectance[nodata_mask] = np.nan
    return reflectance

def to_reflectance(
    data: np.ndarray,
    boa_add_offset: float,
    quantification_value: float,
    nodata_value: float = 0.0
) -> np.ndarray:
    """
    Convert Sentinel-2 L2A Digital Numbers (DN) to BOA reflectance.
    """
    if data.ndim != 3 or data.shape[0] != 4:
        raise ValueError(f"Expected data shape (4, H, W), got {data.shape}")
    return dn_to_reflectance(data, boa_add_offset, quantification_value, nodata_value)
