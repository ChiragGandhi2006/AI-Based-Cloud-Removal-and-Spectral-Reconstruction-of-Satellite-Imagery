"""Preprocess raw SEN12MS-CR data into normalized npz patches.

Expected raw layout (data/raw/SEN12MS-CR):
    scene_<id>/s1/{x.1,x.2}/...             Sentinel-1 VV/VH
    scene_<id>/s2/{x.1,x.2}/...             Sentinel-2 bands
    scene_<id>/cloudmask/{x.1,x.2}/...      cloud masks

Output layout (data/processed/<split>/<scene>/patch_<i>.npz):
    cloudy  (13, H, W)  normalized cloudy S2
    target  (13, H, W)  normalized clear S2
    sar     (2, H, W)   normalized VV/VH
    mask    (1, H, W)   cloud mask
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import resample_bands, align_multimodal, create_cloud_mask
from src.data import random_split


N_BANDS = 13
PATCH = 256


def build_patches(args):
    root = args.root
    raw_dir = os.path.join(root, "data", "raw", "SEN12MS-CR")
    process_dir = os.path.join(root, "data", "processed")

    if not os.path.isdir(raw_dir) or not os.listdir(raw_dir):
        print(f"[WARN] Raw dataset not found at {raw_dir}")
        print("       Falling back to empty split directory scaffolding.")
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(process_dir, split), exist_ok=True)
        return []

    patches = []

    def add_patch(scene_id, cloudy, target, sar, mask):
        if cloudy.shape[1] != PATCH:
            cloudy = resample_bands(cloudy, (PATCH, PATCH))
            target = resample_bands(target, (PATCH, PATCH))
            sar = resample_bands(sar, (PATCH, PATCH))
            mask = resample_bands(mask, (PATCH, PATCH))
        cloudy, sar, mask = align_multimodal(cloudy, sar, mask)
        target = target[:, : cloudy.shape[1], : cloudy.shape[2]]
        patches.append((scene_id, cloudy, target, sar, mask))

    for scene_id in sorted(os.listdir(raw_dir)):
        scene_path = os.path.join(raw_dir, scene_id)
        if not os.path.isdir(scene_path):
            continue
        for clouday in ["cloudy", "cloudy_2018"]:
            cloudy_dir = os.path.join(scene_path, clouday)
            if os.path.isdir(cloudy_dir):
                break
        target_dir = os.path.join(scene_path, "target")
        sar_dir = os.path.join(scene_path, "s1")
        mask_dir = os.path.join(scene_path, "cloudmask")

        for band_file in sorted(os.listdir(cloudy_dir)):
            if not band_file.endswith(".npy"):
                continue
            cloudy = np.load(os.path.join(cloudy_dir, band_file)).astype(np.float32)
            tgt_file = os.path.join(target_dir, band_file)
            sar_file = os.path.join(sar_dir, band_file)
            mask_file = os.path.join(mask_dir, band_file)

            target = np.load(tgt_file).astype(np.float32) if os.path.exists(tgt_file) else cloudy.copy()
            sar = np.load(sar_file).astype(np.float32) if os.path.exists(sar_file) else np.zeros((2, *cloudy.shape[1:]))
            mask = np.load(mask_file).astype(np.float32) if os.path.exists(mask_file) else create_cloud_mask(cloudy)
            if sar.shape[0] == 1 and sar.shape[-2:] == cloudy.shape[-2:]:
                sar = np.concatenate([sar, np.zeros_like(sar)], axis=0)

            add_patch(scene_id, cloudy, target, sar, mask)

    return patches


def main():
    parser = argparse.ArgumentParser(description="Preprocess SEN12MS-CR into npz patches.")
    parser.add_argument("--root", default=".", help="Project root directory.")
    parser.add_argument("--pad_min", action="store_true", help="Generate a few synthetic patches if none found.")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    os.chdir(root)

    patches = build_patches(args)
    if not patches and args.pad_min:
        rng = np.random.RandomState(0)
        for i in range(3):
            patches.append((
                f"scene_synth_{i}",
                rng.rand(N_BANDS, PATCH, PATCH).astype(np.float32),
                rng.rand(N_BANDS, PATCH, PATCH).astype(np.float32),
                rng.rand(2, PATCH, PATCH).astype(np.float32),
                (rng.rand(1, PATCH, PATCH) > 0.7).astype(np.float32),
            ))

    if not patches:
        print("No patches produced. Nothing to do.")
        return

    df = pd.DataFrame({"idx": np.arange(len(patches))})
    train, val, test = random_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    for split, subset in [("train", train), ("val", val), ("test", test)]:
        out_split = os.path.join("data", "processed", split)
        count = 0
        for idx in subset["idx"]:
            scene_id, cloudy, target, sar, mask = patches[int(idx)]
            out_scene = os.path.join(out_split, scene_id)
            os.makedirs(out_scene, exist_ok=True)
            out_path = os.path.join(out_scene, f"patch_{int(idx):04d}.npz")
            np.savez(out_path, cloudy=cloudy, target=target, sar=sar, mask=mask)
            count += 1

        print(f"[{split}] wrote {count} patches -> {out_split}")


if __name__ == "__main__":
    main()