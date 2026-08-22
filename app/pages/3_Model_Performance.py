"""Aggregate model performance over the test split."""
import os
import sys

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(APP_DIR)
for p in (ROOT, APP_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from shared import find_checkpoint
from src.data import SEN12MSCRDataset
from src.inference.predict import load_model
from src.evaluation import PSNR, SSIM, RMSE, SAM

st.set_page_config(page_title="Model Performance", layout="wide", page_icon="📈")
st.title("📈 Model Performance")

ckpt = find_checkpoint()
if ckpt is None:
    st.error("No trained checkpoint found. Run `python scripts/train.py` first.")
    st.stop()

split = st.sidebar.selectbox("Split", ["test", "val", "train"], index=0)

dataset = SEN12MSCRDataset(ROOT, split=split)
if len(dataset) == 0:
    st.warning(f"No processed patches found under `data/processed/{split}`.")
    st.stop()

model, _ = load_model(ckpt, device="cpu")
device = torch.device("cpu")

rows = []
with st.spinner(f"Evaluating {len(dataset)} patches..."):
    for i in range(len(dataset)):
        sample = dataset[i]
        cloudy = torch.as_tensor(sample["cloudy"]).unsqueeze(0).to(device)
        target = torch.as_tensor(sample["target"]).unsqueeze(0).to(device)
        sar = torch.as_tensor(sample["sar"]).unsqueeze(0).to(device)
        mask = torch.as_tensor(sample["mask"]).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(cloudy, sar, mask)

        rows.append({
            "patch": sample["scene"] + "/" + sample["patch_id"],
            "PSNR (dB)": PSNR(pred, target),
            "SSIM": SSIM(pred, target),
            "RMSE": RMSE(pred, target),
            "SAM (rad)": SAM(pred, target),
        })

df = pd.DataFrame(rows)

st.subheader(f"Aggregate metrics on {split} split (n={len(df)})")
summary = df[["PSNR (dB)", "SSIM", "RMSE", "SAM (rad)"]].agg(["mean", "std", "min", "max"])
st.dataframe(summary.round(4))

st.subheader("Per-patch metrics")
st.dataframe(df.round(4))

metric = st.selectbox("Metric to visualize", ["PSNR (dB)", "SSIM", "RMSE", "SAM (rad)"])
fig = px.bar(df, x="patch", y=metric, title=f"{metric} per patch")
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)
