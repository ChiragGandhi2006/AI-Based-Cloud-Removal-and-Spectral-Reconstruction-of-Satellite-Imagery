import torch
from torch.utils.data import Dataset


class Validation:
    """Validation metrics and evaluation."""
    
    @staticmethod
    def compute_psnr(prediction, target, max_val=1.0):
        """Compute PSNR between prediction and target."""
        mse = torch.mean((prediction - target) ** 2)
        if mse == 0:
            return float("inf")
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
        return psnr.item()
    
    @staticmethod
    def compute_ssim(prediction, target):
        """Compute SSIM between prediction and target."""
        # Placeholder - use SSIMLoss from losses module
        from src.losses.ssim_loss import SSIMLoss
        ssim_loss = SSIMLoss()
        return 1 - ssim_loss(prediction, target).item()
    
    @staticmethod
    def compute_rmse(prediction, target):
        """Compute RMSE between prediction and target."""
        mse = torch.mean((prediction - target) ** 2)
        rmse = torch.sqrt(mse)
        return rmse.item()
    
    @staticmethod
    def compute_sam(prediction, target):
        """Compute Spectral Angle Mapper."""
        from src.losses.spectral_loss import SpectralAngleLoss
        sam_loss = SpectralAngleLoss()
        return sam_loss(prediction, target).item()
    
    @staticmethod
    def compute_ndvi(prediction, target):
        """Compute NDVI for comparison."""
        # B8 = NIR, B4 = Red (Sentinel-2 indices 7 and 3, 0-indexed)
        nir = prediction[:, 7, :, :]
        red = prediction[:, 3, :, :]
        nir_target = target[:, 7, :, :]
        red_target = target[:, 3, :, :]
        
        ndvi_pred = (nir - red) / (nir + red + 1e-8)
        ndvi_target = (nir_target - red_target) / (nir_target + red_target + 1e-8)
        
        return ndvi_pred, ndvi_target