"""Evaluate a trained checkpoint on the test split and print aggregate metrics."""
import argparse
import os
import sys

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import build_model
from src.data import SEN12MSCRDataset
from src.evaluation import PSNR, SSIM, RMSE, SAM, NDVI, NDWI
from src.utils import get_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate cloud-removal checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pth", help="Path to .pth checkpoint.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--model", default="attention_unet", help="Model architecture name.")
    parser.add_argument("--base-filters", type=int, default=None, help="Override base filters (optional).")
    parser.add_argument("--use-sar", dest="use_sar", action="store_true", default=None)
    parser.add_argument("--no-use-sar", dest="use_sar", action="store_false")
    parser.add_argument("--use-mask", dest="use_mask", action="store_true", default=None)
    parser.add_argument("--no-use-mask", dest="use_mask", action="store_false")
    parser.set_defaults(use_sar=None, use_mask=None)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    device = get_device(args.device)

    ckpt_path = os.path.join(root, args.checkpoint)
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = dict(ckpt.get("model_config") or {})
    overrides = {}
    if args.base_filters is not None:
        overrides["base_filters"] = args.base_filters
    if args.use_sar is not None:
        overrides["use_sar"] = args.use_sar
    if args.use_mask is not None:
        overrides["use_mask"] = args.use_mask
    model_config.update(overrides)

    model = build_model(
        model_config.pop("name", args.model),
        **model_config,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    dataset = SEN12MSCRDataset(root, split=args.split)
    if len(dataset) == 0:
        print(f"No samples in data/processed/{args.split}.")
        sys.exit(1)

    print(f"Evaluating {len(dataset)} patches on {device}...")

    agg = {"psnr": [], "ssim": [], "rmse": [], "sam": []}
    for i in tqdm(range(len(dataset)), desc=f"Evaluating [{args.split}]"):
        sample = dataset[i]
        cloudy = torch.as_tensor(sample["cloudy"]).unsqueeze(0).to(device)
        target = torch.as_tensor(sample["target"]).unsqueeze(0).to(device)
        sar = torch.as_tensor(sample["sar"]).unsqueeze(0).to(device)
        mask = torch.as_tensor(sample["mask"]).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(cloudy, sar, mask)

        agg["psnr"].append(PSNR(pred, target))
        agg["ssim"].append(SSIM(pred, target))
        agg["rmse"].append(RMSE(pred, target))
        agg["sam"].append(SAM(pred, target))

    print("\n=== Evaluation results ===")
    for metric, values in agg.items():
        print(f"  {metric.upper():>5}: {float(np.mean(values)):.4f}  (n={len(values)})")


if __name__ == "__main__":
    main()