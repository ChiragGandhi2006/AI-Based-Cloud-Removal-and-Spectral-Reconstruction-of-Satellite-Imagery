"""Side-by-side comparison figures for the Streamlit app."""
import matplotlib.pyplot as plt

from .image_viewer import render_rgb


def side_by_side(cloudy, prediction, bands=(3, 2, 1), title="Cloud removal result"):
    """Cloudy input vs cloud-free prediction."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    render_rgb(cloudy, bands, "Cloudy Input", axes[0])
    render_rgb(prediction, bands, "Cloud-free Prediction", axes[1])
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig


def triple(cloudy, prediction, target, bands=(3, 2, 1)):
    """Cloudy input, prediction, and ground truth (if available)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    render_rgb(cloudy, bands, "Cloudy Input", axes[0])
    render_rgb(prediction, bands, "Prediction", axes[1])
    render_rgb(target, bands, "Ground Truth", axes[2])
    plt.tight_layout()
    return fig
