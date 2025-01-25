# utils/data_loader.py

import torch

def collapse_weights(model, sparsity=0.5):
    """
    Prune weights below the given sparsity threshold.
    """
    with torch.no_grad():
        for param in model.parameters():
            threshold = torch.quantile(torch.abs(param), sparsity)
            param[param.abs() < threshold] = 0

def reexpand_weights(model, recovery_rate=0.1):
    """
    Re-expand pruned weights by injecting random noise scaled by recovery_rate.
    """
    with torch.no_grad():
        for param in model.parameters():
            mask = param == 0
            if mask.sum() > 0:
                param[mask] = torch.randn(mask.sum(), device=param.device) * recovery_rate
