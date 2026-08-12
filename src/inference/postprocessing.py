import numpy as np


def clip_reflectance(prediction, min_val=0.0, max_val=1.0):
    """Clip prediction to valid reflectance range."""
    return np.clip(prediction, min_val, max_val)


def spectral_smoothing(prediction, window_size=3):
    """Apply simple spectral smoothing across bands."""
    from scipy.ndimage import uniform_filter1d
    
    n_bands = prediction.shape[0]
    smoothed = np.zeros_like(prediction)
    
    for i in range(n_bands):
        smoothed[i] = uniform_filter1d(prediction[i], size=window_size)
    
    return smoothed


def adjust_sharpness(prediction, alpha=1.0):
    """Adjust spatial sharpness of prediction."""
    # Simple unsharp masking
    from scipy.ndimage import gaussian_filter
    
    blurred = gaussian_filter(prediction, sigma=1.0 / alpha)
    sharp = prediction + alpha * (prediction - blurred)
    
    return clip_reflectance(sharp)


def generate_color_composite(prediction, bands=[3, 2, 1]):
    """Generate RGB color composite from multispectral prediction."""
    # Sentinel-2 band indices (0-based): B3=RGB, B2=RGB, B1=RGB
    # Typical true color: B4(RED), B3(GREEN), B2(BLUE)
    # But user specified bands, so use those
    
    result = np.zeros((3, prediction.shape[1], prediction.shape[2]))
    
    for i, band_idx in enumerate(bands):
        if band_idx < prediction.shape[0]:
            result[i] = prediction[band_idx]
    
    # Normalize each channel
    for i in range(3):
        result[i] = (result[i] - result[i].min()) / (result[i].max() - result[i].min() + 1e-8)
    
    return result