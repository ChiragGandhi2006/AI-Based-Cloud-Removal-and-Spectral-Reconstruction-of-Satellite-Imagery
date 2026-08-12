import numpy as np
import rasterio
from rasterio.enums import Resampling


def resample_bands(bands, target_shape, reference_image=None):
    """
    Resample bands to target shape.
    
    Args:
        bands: numpy array of shape (bands, H, W)
        target_shape: tuple (H, W) target dimensions
        reference_image: optional reference raster for proper geotransform
        
    Returns:
        resampled bands array
    """
    H, W = target_shape
    n_bands = bands.shape[0]
    
    resampled = np.zeros((n_bands, H, W))
    
    for i in range(n_bands):
        # Use skimage resizing or simple interpolation
        from skimage.transform import resize
        resampled[i] = resize(bands[i], (H, W), preserve_range=True, anti_aliasing=True)
    
    return resampled