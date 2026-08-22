import os

import torch
from torch.utils.data import Dataset
import numpy as np


class SEN12MSCRDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        processed_dir = os.path.join(self.root_dir, "data", "processed", self.split)
        if not os.path.isdir(processed_dir):
            processed_dir = os.path.join(self.root_dir, "processed", self.split)
        samples = []
        for scene_dir in sorted(os.listdir(processed_dir)):
            scene_path = os.path.join(processed_dir, scene_dir)
            if os.path.isdir(scene_path):
                for patch_file in sorted(os.listdir(scene_path)):
                    if patch_file.endswith(".npz"):
                        samples.append(os.path.join(scene_path, patch_file))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        data = np.load(path)
        cloudy = data["cloudy"]
        target = data["target"]
        mask = data.get("mask", np.ones_like(cloudy[:, :1]))
        sar = data.get("sar", np.zeros((2, *cloudy.shape[1:])))

        sample = {
            "cloudy": cloudy,
            "target": target,
            "mask": mask,
            "sar": sar,
            "scene": os.path.basename(os.path.dirname(path)),
            "patch_id": os.path.basename(path).replace(".npz", ""),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample