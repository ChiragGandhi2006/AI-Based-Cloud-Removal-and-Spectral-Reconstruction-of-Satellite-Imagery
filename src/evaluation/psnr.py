import torch
import torch.nn.functional as F


def PSNR(prediction, target, max_val=1.0):
    """Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(prediction, target, reduction='none').mean(dim=(1, 2, 3))
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.mean().item()


def SSIM(prediction, target, window_size=11):
    """Structural Similarity Index."""
    # Simplified SSIM calculation
    mu_pred = F.avg_pool2d(prediction, window_size, stride=1, padding=window_size // 2)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)
    
    sigma_pred2 = F.avg_pool2d(prediction ** 2, window_size, stride=1, padding=window_size // 2) - mu_pred ** 2
    sigma_target2 = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size // 2) - mu_target ** 2
    sigma_pred_target = F.avg_pool2d(prediction * target, window_size, stride=1, padding=window_size // 2) - mu_pred * mu_target
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim_map = ((2 * mu_pred * mu_target + c1) * (2 * sigma_pred_target + c2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + c1) * (sigma_pred2 + sigma_target2 + c2))
    
    return ssim_map.mean().item()


def RMSE(prediction, target):
    """Root Mean Squared Error."""
    return torch.sqrt(F.mse_loss(prediction, target)).item()


def SAM(prediction, target):
    """Spectral Angle Mapper."""
    pred_norm = F.normalize(prediction, p=2, dim=1)
    tgt_norm = F.normalize(target, p=2, dim=1)
    cos_sim = (pred_norm * tgt_norm).sum(dim=1).mean()
    return torch.acos(cos_sim.clamp(-1, 1)).item()


def NDVI(prediction, target):
    """Normalized Difference Vegetation Index.
    Uses B8 (NIR) and B4 (Red) from Sentinel-2."""
    nir_pred = prediction[:, 7, :, :]  # B8
    red_pred = prediction[:, 3, :, :]  # B4
    nir_tgt = target[:, 7, :, :]
    red_tgt = target[:, 3, :, :]
    
    ndvi_pred = (nir_pred - red_pred) / (nir_pred + red_pred + 1e-8)
    ndvi_target = (nir_tgt - red_tgt) / (nir_tgt + red_tgt + 1e-8)
    
    return ndvi_pred, ndvi_target


def NDWI(prediction, target):
    """Normalized Difference Water Index.
    Uses B3 (Green) and B11 (SWIR1) from Sentinel-2."""
    green_pred = prediction[:, 2, :, :]  # B3
    swir_pred = prediction[:, 10, :, :]  # B11
    green_tgt = target[:, 2, :, :]
    swir_tgt = target[:, 10, :, :]
    
    ndwi_pred = (green_pred - swir_pred) / (green_pred + swir_pred + 1e-8)
    ndwi_target = (green_tgt - swir_tgt) / (green_tgt + swir_tgt + 1e-8)
    
    return ndwi_pred, ndwi_target


def error_maps(prediction, target):
    """Compute per-pixel error maps."""
    mse_map = (prediction - target) ** 2
    mae_map = torch.abs(prediction - target)
    return mse_map, mae_map