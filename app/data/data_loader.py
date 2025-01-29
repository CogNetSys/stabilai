import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def create_synthetic_binary_classification_dataset(num_samples=4000, input_dim=128, threshold=0.0):
    """
    Generate a linearly separable dataset for binary classification.
    """
    X = torch.randn(num_samples, input_dim)
    y = (X.sum(dim=1) > threshold).long()
    return X, y


def create_ood_binary_classification_dataset(num_samples=1000, input_dim=128, threshold=1.0):
    """
    Generate an out-of-distribution dataset for binary classification.
    """
    X = torch.randn(num_samples, input_dim) + 1.0  # Shifted mean for OOD
    y = (X.sum(dim=1) > threshold).long()
    return X, y


def get_data_loaders(train_size=4000, val_size=1000, ood_size=1000, input_dim=128, batch_size=64):
    """
    Create DataLoaders for training, validation, and OOD datasets.
    """
    # Training data
    X_train, y_train = create_synthetic_binary_classification_dataset(num_samples=train_size, input_dim=input_dim)
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Validation data
    X_val, y_val = create_synthetic_binary_classification_dataset(num_samples=val_size, input_dim=input_dim)
    val_ds = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # OOD data
    X_ood, y_ood = create_ood_binary_classification_dataset(num_samples=ood_size, input_dim=input_dim)
    ood_ds = TensorDataset(X_ood, y_ood)
    ood_loader = DataLoader(ood_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, ood_loader


def add_gaussian_noise(inputs, noise_level=0.1):
    """
    Add Gaussian noise to input features.
    """
    noise = torch.randn_like(inputs) * noise_level
    noisy_inputs = inputs + noise
    return noisy_inputs
