import torch


def get_device():
    """Get available computation device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')