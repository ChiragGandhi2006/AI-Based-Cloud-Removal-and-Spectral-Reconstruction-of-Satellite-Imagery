import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from src.training.train import train_one_epoch, validate


class Trainer:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        
        self.model = self.model.to(self.device)
        
        from src.data.dataloader import create_dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=config.get("batch_size", 8),
            num_workers=config.get("num_workers", 4)
        )
        
        from src.losses.combined_loss import CombinedLoss

        loss_cfg = config.get("loss", {})
        if not isinstance(loss_cfg, dict):
            loss_cfg = {}

        self.loss_fn = CombinedLoss(
            lambda_l1=loss_cfg.get("lambda_l1", 1.0),
            lambda_ssim=loss_cfg.get("lambda_ssim", 0.5),
            lambda_sam=loss_cfg.get("lambda_sam", 0.3),
            mask_weight=loss_cfg.get("mask_weight", 2.0)
        )
        
    def fit(self, epochs=None):
        """Train the model."""
        epochs = epochs or self.config.get("epochs", 100)
        from torch.cuda.amp import GradScaler
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        
        optimizer = AdamW(self.model.parameters(), 
                         lr=self.config.get("learning_rate", 0.001),
                         weight_decay=self.config.get("weight_decay", 1e-4))
        self.optimizer = optimizer
        scheduler = ReduceLROnPlateau(optimizer, 
                                     patience=self.config.get("patience", 10),
                                     min_lr=self.config.get("min_lr", 1e-6))
        scaler = GradScaler() if (self.config.get("mixed_precision", True) and torch.cuda.is_available()) else None
        
        best_val_loss = float("inf")
        
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                self.model, self.train_loader, optimizer, self.loss_fn, 
                self.device, scaler
            )
            val_loss, val_metrics = validate(
                self.model, self.val_loader, self.loss_fn, self.device, scaler
            )
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, val_metrics)
            
            print(f"Epoch {epoch}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.4f}")
        
        return self
    
    def _save_checkpoint(self, epoch, val_loss, metrics):
        """Save model checkpoint."""
        ckpt_path = self.config.get("checkpoint_path", "checkpoints/best.pth")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "metrics": metrics,
            "model_config": getattr(self.model, "model_config", None),
        }, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")