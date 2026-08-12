import torch
import torch.nn.functional as F


def SSIM(prediction, target, window_size=11):
    """Structural Similarity Index."""
    # Gaussian window
    window = torch.ones(1, 1, window_size, window_size) / (window_size ** 2)
    window = window.to(prediction.device)
    
    mu1 = F.conv2d(prediction, window, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=1)
    
    sigma12 = F.conv2d(prediction * target, window, padding=window_size // 2, groups=1) - mu1 * mu2
    sigma1_1 = F.conv2d(prediction ** 2, window, padding=window_size // 2, groups=1) - mu1 ** 2
    sigma2_2 = F.conv2d(target ** 2, window, padding=window_size // 2, groups=1) - mu2 ** 2
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_1 + sigma2_2 + c2))
    
    return ssim.mean().item()