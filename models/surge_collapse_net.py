# models/surge_collapse_net.py

import torch
import torch.nn as nn
from .stable_max import StableMax

class SurgeCollapseNet(nn.Module):
    """
    Neural Network incorporating StableMax activation.
    """
    def __init__(self, input_size=128, hidden_size=256, output_size=128):
        super(SurgeCollapseNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.stablemax = StableMax(dim=1)  # Replaces typical softmax
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.stablemax(x)
        x = self.fc3(x)
        return x
