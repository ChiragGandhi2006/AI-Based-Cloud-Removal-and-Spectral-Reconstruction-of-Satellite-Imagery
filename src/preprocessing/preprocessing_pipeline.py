import numpy as np
from .resampling import resample_bands
from .alignment import align_multimodal


class PreprocessingPipeline:
    def __init__(self, config):
        self.config = config
        self.means = None
        self.stds = None
        
    def fit(self, dataset):
        """Compute normalization statistics from training data."""
        # Compute mean/std across all training patches
        all_bands = []
        for i in range(len(dataset)):
            sample = dataset[i]
            all_bands.append(sample["cloudy"])
        
        all_bands = np.concatenate(all_bands, axis=0)
        self.means = all_bands.mean(axis=(0, 2, 3))
        self.stds = all_bands.std(axis=(0, 2, 3))
        
    def __call__(self, sample):
        """Apply full preprocessing pipeline to a sample."""
        cloudy = sample["cloudy"]
        target = sample["target"]
        sar = sample["sar"]
        mask = sample["mask"]
        
        # Resample if needed
        if cloudy.shape[1] != self.config.get("patch_size", 256):
            cloudy = resample_bands(cloudy, (self.config.get("patch_size", 256),)*2)
            target = resample_bands(target, (self.config.get("patch_size", 256),)*2)
            sar = resample_bands(sar, (self.config.get("patch_size", 256),)*2)
            mask = resample_bands(mask, (self.config.get("patch_size", 256),)*2)
        
        # Normalize
        if self.means is not None:
            cloudy = (cloudy - self.means) / (self.stds + 1e-8)
            target = (target - self.means) / (self.stds + 1e-8)
        
        # Align multimodal inputs
        cloudy, sar, mask = align_multimodal(cloudy, sar, mask)
        
        return {
            "cloudy": cloudy,
            "target": target,
            "sar": sar,
            "mask": mask,
        }