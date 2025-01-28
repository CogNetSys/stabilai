# utils/data_loader.py

import torch
from torch.utils.data import DataLoader, TensorDataset

def create_synthetic_data_loader(num_samples, batch_size, input_size, output_size):
    """
    Function to create a synthetic dataset for binary classification.
    Generates random input data and binary output labels based on a simple rule.
    """
    # Generate random input data
    inputs = torch.randn(num_samples, input_size)
    
    # Binary target generation: sum of inputs > 0
    targets = (inputs.sum(dim=1) > 0).long()
    
    # Create a DataLoader
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
