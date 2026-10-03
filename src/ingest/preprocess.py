"""
Canonical Sentinel-2 Preprocessing Contract.

Project: Sentinel-2 Super-Resolution + Trust Harness (SIH26142, NTRO)
"""

import numpy as np

# Canonical repository band order for Sentinel-2
CANONICAL_BANDS = ["B02", "B03", "B04", "B08"]

def to_reflectance(
    data: np.ndarray,
    boa_add_offset: float,
    quantification_value: float,
    nodata_value: float = 0.0
) -> np.ndarray:
    """
    Convert Sentinel-2 L2A Digital Numbers (DN) to BOA reflectance.
    
    The conversion is defined as:
        reflectance = (DN + boa_add_offset) / quantification_value
        
    Pixels where DN == nodata_value are mapped to np.nan.
    
    Args:
        data: Sentinel-2 L2A data array (typically uint16) containing 4 bands [B02, B03, B04, B08].
              Expected shape is (4, H, W).
        boa_add_offset: The BOA_ADD_OFFSET metadata value (e.g. 0 for PB < 04.00, -1000 for PB >= 04.00).
        quantification_value: The QUANTIFICATION_VALUE metadata value (typically 10000).
        nodata_value: The digital number representing NoData (default 0).
        
    Returns:
        np.ndarray: A float32 array of reflectances with NoData pixels as np.nan.
    """
    if boa_add_offset is None or quantification_value is None:
        raise ValueError("boa_add_offset and quantification_value must be explicitly provided.")
        
    if quantification_value <= 0:
        raise ValueError("quantification_value must be strictly positive.")
        
    if data.ndim != 3 or data.shape[0] != 4:
        raise ValueError(f"Expected data shape (4, H, W), got {data.shape}")
        
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("Input data must be numeric.")

    if np.any(np.isinf(data)):
        raise ValueError("Input data contains Inf values.")

    # Convert to float32
    data_float = data.astype(np.float32)
    
    # Identify NoData pixels
    if np.isnan(nodata_value):
        nodata_mask = np.isnan(data_float)
    else:
        nodata_mask = (data == nodata_value)
    
    # Apply reflectance conversion
    reflectance = (data_float + boa_add_offset) / quantification_value
    
    # Apply NoData handling
    reflectance[nodata_mask] = np.nan
    
    return reflectance
