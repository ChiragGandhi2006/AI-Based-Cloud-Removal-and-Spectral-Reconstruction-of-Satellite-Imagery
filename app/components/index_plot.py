"""Spectral index maps (NDVI / NDWI) for the Streamlit app."""
import numpy as np
import matplotlib.pyplot as plt


def index_map(bands, kind="ndvi"):
    """Compute a spectral index map from (C, H, W) bands."""
    if kind.lower() == "ndvi":
        nir, red = bands[7], bands[3]
        return (nir - red) / (nir + red + 1e-8)
    green, swir = bands[2], bands[10]
    return (green - swir) / (green + swir + 1e-8)


def plot_index_maps(cloudy, prediction, kind="ndvi"):
    """Compare an index map for cloudy input vs prediction."""
    idx_c = index_map(cloudy, kind)
    idx_p = index_map(prediction, kind)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, arr, label in [
        (axes[0], idx_c, f"{kind.upper()} - cloudy input"),
        (axes[1], idx_p, f"{kind.upper()} - prediction"),
    ]:
        im = ax.imshow(arr, cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_title(label)
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.85, orientation="horizontal", pad=0.08)
    plt.tight_layout()
    return fig
