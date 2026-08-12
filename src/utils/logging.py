import torch
import numpy as np


def to_device(data, device):
    """Move tensor(s) to device."""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    
    return data.to(device, non_blocking=True)


def move_to_device(batch, device):
    """Move entire batch to device."""
    return {
        key: to_device(value, device)
        for key, value in batch.items()
    }


def clip_gradients(optimizer, max_norm=1.0):
    """Clip gradients to max norm."""
    total_norm = 0.0
    for p in optimizer.param_groups[0]['params']:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    clip_coeff = max_norm / (total_norm + 1e-6)
    clip_coeff = min(clip_coeff, 1.0)
    
    for p in optimizer.param_groups[0]['params']:
        if p.grad is not None:
            p.grad.data.mul_(clip_coeff)
    
    return total_norm