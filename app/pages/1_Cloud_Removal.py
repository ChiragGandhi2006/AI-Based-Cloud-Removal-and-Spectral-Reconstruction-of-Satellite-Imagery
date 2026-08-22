"""Interactive cloud removal demo."""
import os
import sys

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(APP_DIR)
for p in (ROOT, APP_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from shared import ensure_sample, find_checkpoint, run_inference
from components.comparison import side_by_side, triple
from components.image_viewer import plot_band_montage, render_rgb
from components.metrics_panel import metrics_dataframe

st.set_page_config(page_title="Cloud Removal", layout="wide", page_icon="☁️")
st.title("☁️ Cloud Removal")

sample = ensure_sample()
ckpt = find_checkpoint()

st.sidebar.header("Settings")
show_bands = st.sidebar.multiselect(
    "Bands for RGB composite",
    ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
    default=["B4", "B3", "B2"],
)
band_indices = {"B2": 1, "B3": 2, "B4": 3, "B5": 4, "B6": 5, "B7": 6, "B8": 7, "B8A": 8, "B11": 10, "B12": 11}
rgb_bands = tuple(band_indices[b] for b in show_bands[:3]) or (3, 2, 1)

st.markdown(f"**Sample:** `{sample['name']}`")

col1, col2 = st.columns(2)
with col1:
    fig = plt.figure(figsize=(5, 5))
    render_rgb(sample["cloudy"], rgb_bands, "Cloudy Input", fig.gca())
    st.pyplot(fig)
    plt.close(fig)

if ckpt is None:
    st.error(
        "No trained checkpoint found. Run `python scripts/train.py` first, or place "
        "a model at `checkpoints/best.pth`."
    )
    st.stop()

with st.spinner("Running cloud removal..."):
    pred = run_inference(sample, ckpt)

with col2:
    fig = plt.figure(figsize=(5, 5))
    render_rgb(pred, rgb_bands, "Cloud-free Prediction", fig.gca())
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Side-by-side comparison")
st.pyplot(side_by_side(sample["cloudy"], pred, rgb_bands))
plt.close("all")

if "target" in sample:
    st.subheader("Metrics vs ground truth")
    st.dataframe(metrics_dataframe(pred, sample["target"]))

    st.subheader("Triple view (input / prediction / truth)")
    st.pyplot(triple(sample["cloudy"], pred, sample["target"], rgb_bands))
    plt.close("all")

with st.expander("Inspect individual bands"):
    choice = st.selectbox("Show bands for", ["Cloudy input", "Prediction"])
    arr = sample["cloudy"] if choice == "Cloudy input" else pred
    st.pyplot(plot_band_montage(arr))
    plt.close("all")

col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button(
        "Download prediction (.npy)",
        pred.astype(np.float32).tobytes(),
        file_name="cloud_free_prediction.npy",
        mime="application/octet-stream",
    )
with col_dl2:
    st.download_button(
        "Download input (.npy)",
        sample["cloudy"].astype(np.float32).tobytes(),
        file_name="cloudy_input.npy",
        mime="application/octet-stream",
    )
