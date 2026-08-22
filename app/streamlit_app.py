"""AI-Based Cloud Removal and Spectral Reconstruction - Streamlit app."""
import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
for p in (ROOT, APP_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import matplotlib.pyplot as plt

from shared import ensure_sample, find_checkpoint, run_inference
from components.comparison import side_by_side
from components.metrics_panel import metrics_dataframe

st.set_page_config(
    page_title="AI Cloud Removal & Spectral Reconstruction",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ AI-Based Cloud Removal & Spectral Reconstruction")
st.caption("Sentinel-2 optical + Sentinel-1 SAR fusion with Attention U-Net")

st.markdown(
    """
This app demonstrates removing clouds from satellite imagery and reconstructing the
underlying spectral information using a deep learning pipeline trained on the
SEN12MS-CR dataset. Use the sidebar pages for interactive demos:

- **1. Cloud Removal** - run the model on a sample patch and inspect the result
- **2. Spectral Analysis** - compare spectral profiles and vegetation/water indices
- **3. Model Performance** - aggregate metrics over the test split
- **4. About Project** - background and documentation
"""
)

st.divider()

sample = ensure_sample()
ckpt = find_checkpoint()

with st.spinner("Preparing demo..."):
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Cloudy Input")
        fig = plt.figure(figsize=(4, 4))
        from components.image_viewer import render_rgb
        render_rgb(sample["cloudy"], title=sample["name"], ax=fig.gca())
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        st.subheader("Cloud-free Prediction")
        if ckpt is not None:
            pred = run_inference(sample, ckpt)
            fig = plt.figure(figsize=(4, 4))
            render_rgb(pred, title="Reconstructed", ax=fig.gca())
            st.pyplot(fig)
            plt.close(fig)
            if "target" in sample:
                st.dataframe(metrics_dataframe(pred, sample["target"]))
        else:
            st.info(
                "No trained checkpoint found in `checkpoints/`. Train a model with "
                "`python scripts/train.py` to enable predictions here."
            )

if ckpt is not None:
    st.success(f"Loaded checkpoint: `{os.path.relpath(ckpt, ROOT)}`")
else:
    st.warning("No checkpoint found — demo is running in synthetic mode.")
