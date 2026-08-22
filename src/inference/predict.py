import torch
import numpy as np
from pathlib import Path


def load_model(checkpoint_path, device, model_name="attention_unet", **model_kwargs):
    """Load model from checkpoint, rebuilding architecture from saved config."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    from src.models import build_model

    config = dict(checkpoint.get("model_config") or {})
    config.setdefault("name", model_name)
    for key, value in model_kwargs.items():
        config[key] = value
    name = config.pop("name")

    model = build_model(name, **config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def predict_from_checkpoint(cloudy_s2, sar, mask, checkpoint_path, device=None, 
                            mean=None, std=None, **model_kwargs):
    """
    Run inference on cloudy Sentinel-2 data.
    
    Args:
        cloudy_s2: cloudy Sentinel-2 bands (1, 13, H, W) or (13, H, W)
        sar: Sentinel-1 SAR (1, 2, H, W) or (2, H, W) - VV, VH
        mask: cloud/shadow mask (1, 1, H, W) or (1, H, W)
        checkpoint_path: path to model checkpoint
        device: torch device
        mean: normalization mean
        std: normalization std
        
    Returns:
        reconstructed S2 bands
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, checkpoint = load_model(checkpoint_path, device, **model_kwargs)
    
    # Preprocess input
    if mean is not None and std is not None:
        cloudy_s2 = (cloudy_s2 - mean) / std

    # Ensure proper shapes and convert to tensors
    cloudy_s2 = torch.as_tensor(cloudy_s2, dtype=torch.float32)
    sar = torch.as_tensor(sar, dtype=torch.float32) if sar is not None else None
    mask = torch.as_tensor(mask, dtype=torch.float32) if mask is not None else None

    if cloudy_s2.ndim == 3:
        cloudy_s2 = cloudy_s2.unsqueeze(0)  # Add batch dimension
    if sar is not None and sar.ndim == 3:
        sar = sar.unsqueeze(0)
    if mask is not None and mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif mask is not None and mask.ndim == 3:
        mask = mask.unsqueeze(0)

    # Move to device
    cloudy_s2 = cloudy_s2.to(device)
    sar = sar.to(device) if sar is not None else None
    mask = mask.to(device) if mask is not None else None
    
    # Inference
    with torch.no_grad():
        prediction = model(cloudy_s2, sar, mask)
    
    # Denormalize if mean/std provided
    if mean is not None and std is not None:
        prediction = prediction * std + mean
    
    return prediction.cpu().numpy()


def preprocess_input(cloudy_s2, sar, mask, config):
    """Preprocess input data for model."""
    # Normalize
    if "normalization" in config and config["normalization"] == "train_stats":
        # Placeholder - would use actual training stats
        pass
    
    # Resample if needed
    target_size = config.get("patch_size", 256)
    from src.preprocessing.resampling import resample_bands
    
    if cloudy_s2.shape[1] != target_size:
        cloudy_s2 = resample_bands(cloudy_s2, (target_size, target_size))
        if sar is not None:
            sar = resample_bands(sar, (target_size, target_size))
        if mask is not None:
            mask = resample_bands(mask, (target_size, target_size))
    
    # Align
    from src.preprocessing.alignment import align_multimodal
    cloudy_s2, sar, mask = align_multimodal(cloudy_s2, sar, mask)
    
    return cloudy_s2, sar, mask


def postprocess_output(prediction, config):
    """Postprocess model output."""
    # Clip to valid reflectance range [0, 1]
    prediction = np.clip(prediction, 0, 1)
    
    # Ensure correct shape (C, H, W)
    if prediction.ndim == 4:
        prediction = prediction.squeeze(0)
    
    return prediction