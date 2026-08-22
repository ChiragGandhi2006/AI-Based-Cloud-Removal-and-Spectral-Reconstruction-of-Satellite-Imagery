"""Spectral profile plotting for the Streamlit app."""
import numpy as np
import matplotlib.pyplot as plt

from ..shared import BAND_NAMES


def plot_spectral_profiles(cloudy, prediction=None, target=None, title="Mean spectral reflectance"):
    """Plot mean per-band reflectance for input, prediction and target."""
    bands = np.arange(cloudy.shape[0])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(bands, cloudy.mean(axis=(1, 2)), "o-", label="Cloudy input")
    if prediction is not None:
        ax.plot(bands, prediction.mean(axis=(1, 2)), "^-", label="Prediction")
    if target is not None:
        ax.plot(bands, target.mean(axis=(1, 2)), "s--", label="Ground truth")

    ax.set_xticks(bands)
    ax.set_xticklabels(BAND_NAMES, rotation=45)
    ax.set_xlabel("Band")
    ax.set_ylabel("Mean reflectance")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig
