import numpy as np


def normalize_reflectance(bands, mean=None, std=None):
    """Normalize reflectance bands."""
    if mean is not None and std is not None:
        return (bands - mean) / std
    else:
        # Simple normalization to [0, 1]
        min_val = bands.min()
        max_val = bands.max()
        return (bands - min_val) / (max_val - min_val + 1e-8)


def denormalize_reflectance(bands, mean, std):
    """Denormalize reflectance bands."""
    return bands * std + mean


def add_batch_dimension(tensor):
    """Ensure tensor has batch dimension."""
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    return tensor


def remove_batch_dimension(tensor):
    """Remove batch dimension if present."""
    if tensor.ndim == 4:
        return tensor.squeeze(0)
    return tensor