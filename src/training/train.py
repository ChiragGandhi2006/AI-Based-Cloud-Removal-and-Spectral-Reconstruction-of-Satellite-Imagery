import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import yaml


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        cloudy = batch["cloudy"].to(device)
        target = batch["target"].to(device)
        sar = batch.get("sar", torch.zeros_like(cloudy[:, :2]).to(device))
        mask = batch.get("mask", torch.ones_like(cloudy[:, :1]).to(device))
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                prediction = model(cloudy, sar, mask)
                loss, loss_components = loss_fn(prediction, target, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            prediction = model(cloudy, sar, mask)
            loss, loss_components = loss_fn(prediction, target, mask)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({"loss": total_loss / num_batches})
    
    return total_loss / num_batches


def validate(model, dataloader, loss_fn, device, scaler=None):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_metrics = {}
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            cloudy = batch["cloudy"].to(device)
            target = batch["target"].to(device)
            sar = batch.get("sar", torch.zeros_like(cloudy[:, :2]).to(device))
            mask = batch.get("mask", torch.ones_like(cloudy[:, :1]).to(device))
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    prediction = model(cloudy, sar, mask)
                    loss, loss_components = loss_fn(prediction, target, mask)
            else:
                prediction = model(cloudy, sar, mask)
                loss, loss_components = loss_fn(prediction, target, mask)
            
            total_loss += loss.item()
            
            # Accumulate metrics
            if num_batches == 0:
                for key in loss_components:
                    total_metrics[key] = 0.0
            
            for key in total_metrics:
                if key in loss_components:
                    total_metrics[key] += loss_components[key]
            
            num_batches += 1
            
            pbar.set_postfix({"val_loss": total_loss / num_batches})
    
    # Average metrics
    for key in total_metrics:
        total_metrics[key] /= num_batches
    
    return total_loss / num_batches, total_metrics


def run_training(model, train_loader, val_loader, config, device):
    """Run complete training pipeline."""
    from torch.cuda.amp import GradScaler
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    
    epochs = config.get("epochs", 100)
    lr = config.get("learning_rate", 0.001)
    batch_size = config.get("batch_size", 8)
    weight_decay = config.get("weight_decay", 1e-4)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, min_lr=1e-6)
    scaler = GradScaler() if config.get("mixed_precision", True) else None
    loss_fn = CombinedLoss(
        lambda_l1=config.get("lambda_l1", 1.0),
        lambda_ssim=config.get("lambda_ssim", 0.5),
        lambda_sam=config.get("lambda_sam", 0.3),
        mask_weight=config.get("mask_weight", 2.0)
    )
    
    best_val_loss = float("inf")
    checkpoint_dir = config.get("save_dir", "checkpoints")
    
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        val_loss, val_metrics = validate(model, val_loader, loss_fn, device, scaler)
        
        scheduler.step(val_loss)
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, f"best_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_loss,
                "metrics": val_metrics,
            }, ckpt_path)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        for key, val in val_metrics.items():
            print(f"  Val {key}: {val:.4f}")
    
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return model