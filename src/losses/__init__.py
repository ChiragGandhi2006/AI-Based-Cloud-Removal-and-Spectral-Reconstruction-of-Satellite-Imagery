import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    """L1 (Mean Absolute Error) loss."""
    
    def __init__(self, reduction='mean'):
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.l1 = nn.L1Loss(reduction=reduction)
    
    def forward(self, prediction, target):
        return self.l1(prediction, target)


class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss (per-band)."""
    
    def __init__(self, window_size=11, reduction='mean'):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.reduction = reduction
        self.register_buffer('window', torch.ones(1, 1, window_size, window_size) / (window_size ** 2))
    
    def forward(self, prediction, target):
        # Compute SSIM per band
        prediction = prediction.float()
        target = target.float()
        
        B, C, H, W = prediction.shape
        pred = prediction.reshape(B * C, 1, H, W)
        tgt = target.reshape(B * C, 1, H, W)
        pad = self.window_size // 2
        
        # Mean calculation
        mean_pred = F.conv2d(pred, self.window, padding=pad)
        mean_target = F.conv2d(tgt, self.window, padding=pad)
        
        # Variance and covariance
        var_pred = F.conv2d(pred ** 2, self.window, padding=pad) - mean_pred ** 2
        var_target = F.conv2d(tgt ** 2, self.window, padding=pad) - mean_target ** 2
        cov_pred_target = F.conv2d(pred * tgt, self.window, padding=pad) - mean_pred * mean_target
        
        # SSIM formula
        k1, k2 = 0.01, 0.03
        l = 255
        C1 = (k1 * l) ** 2
        C2 = (k2 * l) ** 2
        
        ssim_map = ((2 * mean_pred * mean_target + C1) * (2 * cov_pred_target + C2)) / \
                   ((mean_pred ** 2 + mean_target ** 2 + C1) * (var_pred + var_target + C2))
        
        if self.reduction == 'mean':
            return 1 - ssim_map.mean()
        return (1 - ssim_map).reshape(B, C, H, W)


class SpectralAngleLoss(nn.Module):
    """Spectral Angle Mapper (SAM) loss."""
    
    def forward(self, prediction, target):
        """
        Compute spectral angle between prediction and target.
        Args:
            prediction: (N, C, H, W)
            target: (N, C, H, W)
        Returns:
            mean spectral angle across batch
        """
        pred = prediction.float()
        tgt = target.float()
        
        # Normalize along channel dimension
        pred_norm = F.normalize(pred, p=2, dim=1)
        tgt_norm = F.normalize(tgt, p=2, dim=1)
        
        # Compute cosine similarity
        cos_sim = (pred_norm * tgt_norm).sum(dim=1)
        cos_sim = cos_sim.clamp(-1, 1)
        
        # Angle in radians
        angle = torch.acos(cos_sim)
        
        return angle.mean()


class CombinedLoss(nn.Module):
    """Combined loss: L1 + SSIM + SAM."""
    
    def __init__(self, lambda_l1=1.0, lambda_ssim=0.5, lambda_sam=0.3, mask_weight=2.0):
        super(CombinedLoss, self).__init__()
        self.l1 = L1Loss(reduction='none')
        self.ssim = SSIMLoss()
        self.sam = SpectralAngleLoss()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_sam = lambda_sam
        self.mask_weight = mask_weight
    
    def forward(self, prediction, target, mask=None):
        # L1 loss (optionally mask-aware: up-weight cloudy regions)
        l1_loss = self.l1(prediction, target)
        if mask is not None:
            m = mask.squeeze(1) if mask.dim() == 4 else mask  # (B, H, W)
            if m.dim() == 3 and m.shape[0] == prediction.shape[0]:
                per_pixel = l1_loss.mean(dim=1)  # (B, H, W)
                weight = 1.0 + (self.mask_weight - 1.0) * m
                l1_loss = (per_pixel * weight).mean()
            else:
                l1_loss = l1_loss.mean()
        else:
            l1_loss = l1_loss.mean()
        
        # SSIM loss
        ssim_loss = self.ssim(prediction, target)
        
        # SAM loss
        sam_loss = self.sam(prediction, target)
        
        total = (self.lambda_l1 * l1_loss + 
                 self.lambda_ssim * ssim_loss + 
                 self.lambda_sam * sam_loss)
        
        return total, {"l1": l1_loss, "ssim": ssim_loss, "sam": sam_loss}