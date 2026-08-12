import numpy as np


def normalize_bands(bands, mean, std):
    """Normalize Sentinel-2 bands using provided statistics."""
    return (bands - mean) / std


def denormalize_bands(bands, mean, std):
    """Denormalize Sentinel-2 bands using provided statistics."""
    return bands * std + mean


class NormalizationStats:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def normalize(self, bands):
        return normalize_bands(bands, self.mean, self.std)

    def denormalize(self, bands):
        return denormalize_bands(bands, self.mean, self.std)