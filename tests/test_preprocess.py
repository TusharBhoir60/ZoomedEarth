import numpy as np
import pytest
from src.ingest.preprocess import to_reflectance, CANONICAL_BANDS

def test_canonical_bands_order():
    assert CANONICAL_BANDS == ["B02", "B03", "B04", "B08"]

def test_zero_offset():
    """PB < 04.00, offset=0, quant=10000"""
    data = np.array([[[1000, 0], [5000, 10000]]], dtype=np.uint16)
    data = np.repeat(data, 4, axis=0) # shape (4, 2, 2)
    
    res = to_reflectance(data, boa_add_offset=0, quantification_value=10000, nodata_value=0)
    
    assert res.shape == (4, 2, 2)
    assert res.dtype == np.float32
    
    # Check valid values
    np.testing.assert_allclose(res[0, 0, 0], 0.1)
    np.testing.assert_allclose(res[0, 1, 0], 0.5)
    np.testing.assert_allclose(res[0, 1, 1], 1.0)
    
    # Check nodata (DN=0)
    assert np.isnan(res[0, 0, 1])

def test_negative_1000_offset():
    """PB >= 04.00, offset=-1000, quant=10000"""
    data = np.array([[[2000, 0], [6000, 11000]]], dtype=np.uint16)
    data = np.repeat(data, 4, axis=0) # shape (4, 2, 2)
    
    res = to_reflectance(data, boa_add_offset=-1000, quantification_value=10000, nodata_value=0)
    
    # 2000 -> 0.1
    np.testing.assert_allclose(res[0, 0, 0], 0.1)
    # 6000 -> 0.5
    np.testing.assert_allclose(res[0, 1, 0], 0.5)
    # 11000 -> 1.0
    np.testing.assert_allclose(res[0, 1, 1], 1.0)
    
    # Check nodata (DN=0) mapped to nan, rather than (0-1000)/10000 = -0.1
    assert np.isnan(res[0, 0, 1])

def test_metadata_specified_quantification_value():
    """Varying quantification value"""
    data = np.array([[[1000, 2000]]], dtype=np.uint16)
    data = np.repeat(data, 4, axis=0)
    
    res = to_reflectance(data, boa_add_offset=0, quantification_value=2000, nodata_value=0)
    np.testing.assert_allclose(res[0, 0, 0], 0.5)
    np.testing.assert_allclose(res[0, 0, 1], 1.0)

def test_missing_metadata_failures():
    data = np.ones((4, 2, 2), dtype=np.uint16)
    
    with pytest.raises(ValueError, match="must be explicitly provided"):
        to_reflectance(data, boa_add_offset=None, quantification_value=10000)
        
    with pytest.raises(ValueError, match="must be explicitly provided"):
        to_reflectance(data, boa_add_offset=0, quantification_value=None)
        
    with pytest.raises(ValueError, match="must be strictly positive"):
        to_reflectance(data, boa_add_offset=0, quantification_value=0)

def test_invalid_shape():
    data_3bands = np.ones((3, 2, 2), dtype=np.uint16)
    with pytest.raises(ValueError, match="Expected data shape"):
        to_reflectance(data_3bands, boa_add_offset=0, quantification_value=10000)
        
    data_2d = np.ones((10, 10), dtype=np.uint16)
    with pytest.raises(ValueError, match="Expected data shape"):
        to_reflectance(data_2d, boa_add_offset=0, quantification_value=10000)

def test_invalid_nan_inf():
    data = np.ones((4, 2, 2), dtype=np.float32)
    data[0, 0, 0] = np.inf
    
    with pytest.raises(ValueError, match="contains Inf values"):
        to_reflectance(data, boa_add_offset=0, quantification_value=10000)
