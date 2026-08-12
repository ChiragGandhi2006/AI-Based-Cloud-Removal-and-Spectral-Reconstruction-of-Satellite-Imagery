import torch
import numpy as np
import os
import setproctitle


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(prefer="auto"):
    """Get computation device."""
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif prefer == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        return torch.device("cpu")


def setup_logger(name="AICloudRemoval", log_file="training.log"):
    """Set up logger."""
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_checkpoint(state, checkpoint_dir, filename="checkpoint.pth"):
    """Save training checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    return path


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load training checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model):
    """Print model parameter count."""
    n_params = count_parameters(model)
    print(f"Model: {type(model).__name__}")
    print(f"Trainable parameters: {n_params:,}")


def visualize_tensor(tensor, title="Tensor", save_path=None):
    """Visualize a tensor as image."""
    import matplotlib.pyplot as plt
    
    # Remove batch dim if present
    if tensor.ndim == 4:
        tensor = tensor[0]
    
    # Clip to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    plt.figure(figsize=(8, 8))
    if tensor.ndim == 3:
        # (C, H, W) - show first few bands or create montage
        n_bands = min(tensor.shape[0], 12)
        n_rows = (n_bands + 3) // 4
        n_cols = min(4, n_bands)
        
        for i in range(n_bands):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(tensor[i].cpu().numpy(), cmap='terrain')
            plt.axis('off')
            plt.title(f"Band {i+1}")
    else:
        plt.imshow(tensor.cpu().numpy(), cmap='terrain')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()