"""Run cloud-removal inference on a single patch and save the result."""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.predict import predict_from_checkpoint, postprocess_output
from src.evaluation import PSNR, SSIM, RMSE, SAM
from src.utils import get_device


def save_rgb(path, image, title, bands=(3, 2, 1)):
    """Save a normalized RGB composite of a (C, H, W) array."""
    rgb = np.stack([image[b] for b in bands], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    plt.figure(figsize=(4, 4))
    plt.imshow(np.clip(rgb, 0, 1))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Predict cloud removal for a patch.")
    parser.add_argument("--input", required=True, help="Path to .npz patch (cloudy/sar/mask).")
    parser.add_argument("--checkpoint", default="checkpoints/best.pth", help="Path to .pth checkpoint.")
    parser.add_argument("--output-dir", default="outputs/predictions", help="Output directory.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--base-filters", type=int, default=None, help="Override model base_filters (optional).")
    parser.add_argument("--use-sar", dest="use_sar", action="store_true", default=None)
    parser.add_argument("--no-use-sar", dest="use_sar", action="store_false")
    parser.add_argument("--use-mask", dest="use_mask", action="store_true", default=None)
    parser.add_argument("--no-use-mask", dest="use_mask", action="store_false")
    parser.set_defaults(use_sar=None, use_mask=None)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    device = get_device(args.device)

    data = np.load(args.input)
    cloudy = data["cloudy"].astype(np.float32)
    sar = data["sar"].astype(np.float32) if "sar" in data else np.zeros((2, *cloudy.shape[1:]))
    mask = data["mask"].astype(np.float32) if "mask" in data else np.ones((1, *cloudy.shape[1:]))

    overrides = {}
    if args.base_filters is not None:
        overrides["base_filters"] = args.base_filters
    if args.use_sar is not None:
        overrides["use_sar"] = args.use_sar
    if args.use_mask is not None:
        overrides["use_mask"] = args.use_mask

    pred = predict_from_checkpoint(
        cloudy, sar, mask, os.path.join(root, args.checkpoint),
        device=device, **overrides,
    )
    pred = postprocess_output(pred, {})

    os.makedirs(os.path.join(root, args.output_dir), exist_ok=True)
    out_npy = os.path.join(root, args.output_dir, "prediction.npy")
    np.save(out_npy, pred)
    print(f"Saved prediction: {out_npy}  shape={pred.shape}")

    save_rgb(os.path.join(root, args.output_dir, "input_rgb.png"), cloudy, "Cloudy input")
    save_rgb(os.path.join(root, args.output_dir, "prediction_rgb.png"), pred, "Cloud-free prediction")

    if "target" in data:
        target = data["target"].astype(np.float32)
        pred_t = torch.as_tensor(pred).unsqueeze(0)
        tgt_t = torch.as_tensor(target).unsqueeze(0)
        print("\nMetrics vs ground truth:")
        print(f"  PSNR: {PSNR(pred_t, tgt_t):.3f}")
        print(f"  SSIM: {SSIM(pred_t, tgt_t):.4f}")
        print(f"  RMSE: {RMSE(pred_t, tgt_t):.4f}")
        print(f"  SAM:  {SAM(pred_t, tgt_t):.4f}")


if __name__ == "__main__":
    main()