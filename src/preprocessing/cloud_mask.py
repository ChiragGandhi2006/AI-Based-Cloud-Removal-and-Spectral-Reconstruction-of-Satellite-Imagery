import numpy as np


def create_cloud_mask(bands, threshold=0.5):
    """
    Create a cloud mask from Sentinel-2 bands.
    Uses simple thresholding on NIR and SWIR bands.
    
    Args:
        bands: numpy array (N, H, W) - Sentinel-2 bands
        threshold: cloud detection threshold
        
    Returns:
        binary mask (1, H, W) where 1 = cloudy
    """
    # Band 10 is typically the cirrus band, B8 is NIR
    nir = bands[7]  # B8 - NIR
    swir = bands[10]  # B11 - SWIR
    
    # Simple cloud detection: high NIR + low SWIR often indicates cloud
    cloud_score = nir / (swir + 1e-8)
    mask = (cloud_score > threshold).astype(np.float32)
    
    return mask[np.newaxis, ...]


def create_shadow_mask(cloud_mask, sun_zenith=30.0):
    """
    Create a cloud shadow mask from a cloud mask.
    Uses simple geometry based on sun position.
    
    Args:
        cloud_mask: binary cloud mask (1, H, W)
        sun_zenith: sun zenith angle in degrees
        
    Returns:
        binary shadow mask (1, H, W) where 1 = shadow
    """
    # Simple shadow extension from cloud boundaries
    mask = cloud_mask.squeeze()
    
    # Dilate cloud mask to estimate shadow area
    # In practice, use Sun angle and DEM for accurate shadow prediction
    from scipy import ndimage
    shadow = ndimage.binary_dilation(mask, structure=np.ones((5, 5))).astype(np.float32)
    
    return shadow[np.newaxis, ...]