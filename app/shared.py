"""Shared helpers for the Streamlit app (path bootstrap, model/data loading)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, APP_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.inference.predict import predict_from_checkpoint, postprocess_output

BAND_NAMES = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7",
    "B8", "B8A", "B9", "B10", "B11", "B12",
]
RGB_BANDS = (3, 2, 1)  # B4(Red), B3(Green), B2(Blue)

_CANDIDATE_CHECKPOINTS = [
    "checkpoints/best.pth",
    "checkpoints/attention_unet/best.pth",
    "checkpoints/baseline/best.pth",
    "checkpoints/e2e_test/best.pth",
]


def find_checkpoint():
    """Return the first existing checkpoint path, else None."""
    for rel in _CANDIDATE_CHECKPOINTS:
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            return path
    return None


def load_sample():
    """Load the first available npz patch from data/processed or None."""
    for split in ["test", "val", "train"]:
        base = os.path.join(ROOT, "data", "processed", split)
        if not os.path.isdir(base):
            continue
        for scene in sorted(os.listdir(base)):
            scene_dir = os.path.join(base, scene)
            if not os.path.isdir(scene_dir):
                continue
            for fname in sorted(os.listdir(scene_dir)):
                if fname.endswith(".npz"):
                    with np.load(os.path.join(scene_dir, fname)) as data:
                        sample = {k: np.asarray(v, dtype=np.float32) for k, v in data.items()}
                    sample["name"] = f"{split}/{scene}/{fname}"
                    return sample
    return None


def synthetic_sample():
    """Generate a synthetic cloudy patch for demo purposes."""
    rng = np.random.RandomState(0)
    cloudy = np.clip(rng.rand(13, 256, 256) * 1.2, 0, 1).astype(np.float32)
    cloudy[:, 70:190, 70:190] = np.clip(cloudy[:, 70:190, 70:190] + 0.9, 0, 1)
    target = np.clip(cloudy - 0.35, 0, 1).astype(np.float32)
    target[:, 70:190, 70:190] = np.clip(rng.rand(13, 120, 120) * 1.0, 0, 1).astype(np.float32)
    sample = {
        "cloudy": cloudy,
        "target": target,
        "sar": rng.rand(2, 256, 256).astype(np.float32),
        "mask": np.zeros((1, 256, 256), dtype=np.float32),
        "name": "synthetic-demo (no dataset found)",
    }
    return sample


def ensure_sample():
    sample = load_sample()
    if sample is None:
        sample = synthetic_sample()
    return sample


def run_inference(sample, checkpoint):
    """Run cloud removal on a sample using a checkpoint. Returns prediction array."""
    cloudy = sample["cloudy"]
    sar = sample.get("sar", np.zeros((2, *cloudy.shape[1:]), dtype=np.float32))
    mask = sample.get("mask", np.ones((1, *cloudy.shape[1:]), dtype=np.float32))
    pred = predict_from_checkpoint(cloudy, sar, mask, checkpoint, device="cpu")
    pred = postprocess_output(pred, {})
    return pred
