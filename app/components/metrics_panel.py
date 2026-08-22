"""Metric computation helpers for the Streamlit app."""
import numpy as np
import pandas as pd
import torch

from src.evaluation import PSNR, SSIM, RMSE, SAM


def compute_metrics(prediction, target):
    """Compute PSNR/SSIM/RMSE/SAM between prediction and target arrays."""
    p = torch.as_tensor(prediction, dtype=torch.float32).unsqueeze(0)
    t = torch.as_tensor(target, dtype=torch.float32).unsqueeze(0)
    return {
        "PSNR (dB)": round(PSNR(p, t), 3),
        "SSIM": round(SSIM(p, t), 4),
        "RMSE": round(RMSE(p, t), 4),
        "SAM (rad)": round(SAM(p, t), 4),
    }


def metrics_dataframe(prediction, target):
    """Return a one-row DataFrame of metrics for st.dataframe."""
    return pd.DataFrame([compute_metrics(prediction, target)])
