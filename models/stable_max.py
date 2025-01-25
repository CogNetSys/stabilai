# models/stable_max.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class StableMax(nn.Module):
    """
    A numerically stable alternative to softmax.
    """
    def __init__(self, dim=-1):
        super(StableMax, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Subtract max for numerical stability
        x_max, _ = torch.max(x, dim=self.dim, keepdim=True)
        x = x - x_max
        return F.softmax(x, dim=self.dim)
