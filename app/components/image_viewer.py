"""Image visualization helpers for the Streamlit app."""
import numpy as np
import matplotlib.pyplot as plt


def to_rgb(image, bands=(3, 2, 1)):
    """Convert a (C, H, W) multispectral array to a normalized RGB array."""
    img = image
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 2:
        rgb = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[0] in (3, 13):
        channels = [img[b] for b in bands if b < img.shape[0]]
        rgb = np.stack(channels, axis=-1)
        while rgb.shape[-1] < 3:
            rgb = np.concatenate([rgb, rgb[..., :1]], axis=-1)
    else:
        rgb = img[..., :3]

    rgb = rgb.astype(np.float32)
    out = np.zeros_like(rgb)
    for c in range(3):
        ch = rgb[..., c]
        out[..., c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
    return np.clip(out, 0, 1)


def render_rgb(image, bands=(3, 2, 1), title="", ax=None):
    """Plot an RGB composite of a multispectral array."""
    rgb = to_rgb(image, bands)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")
    return ax


def plot_band_montage(image, n_bands=None, title="Spectral bands"):
    """Plot individual bands as a montage."""
    n = min(n_bands or image.shape[0], 13)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(rows * cols):
        if i < n:
            band = image[i]
            axes[i].imshow(band, cmap="terrain",
                           vmin=band.min(), vmax=band.max())
            axes[i].set_title(f"Band {i + 1}")
        axes[i].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    return fig
