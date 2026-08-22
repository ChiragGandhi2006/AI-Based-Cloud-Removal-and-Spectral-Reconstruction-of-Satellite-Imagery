"""Spectral analysis: profiles and vegetation/water indices."""
import os
import sys

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(APP_DIR)
for p in (ROOT, APP_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import matplotlib.pyplot as plt

from shared import ensure_sample, find_checkpoint, run_inference
from components.spectral_plot import plot_spectral_profiles
from components.index_plot import plot_index_maps

st.set_page_config(page_title="Spectral Analysis", layout="wide", page_icon="📊")
st.title("📊 Spectral Analysis")

sample = ensure_sample()
ckpt = find_checkpoint()

if ckpt is None:
    st.error(
        "No trained checkpoint found. Run `python scripts/train.py` first, or place "
        "a model at `checkpoints/best.pth`."
    )
    st.stop()

with st.spinner("Running cloud removal..."):
    pred = run_inference(sample, ckpt)

target = sample.get("target")

st.subheader("Mean spectral reflectance profile")
st.pyplot(plot_spectral_profiles(sample["cloudy"], pred, target))
plt.close("all")

index_kind = st.radio("Spectral index", ["NDVI", "NDWI"], horizontal=True)
st.subheader(f"{index_kind} comparison")
st.pyplot(plot_index_maps(sample["cloudy"], pred, kind=index_kind.lower()))
plt.close("all")
