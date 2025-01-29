# app/models/fastgrokkingrush.py

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class FastGrokkingRushNetWithGAT(nn.Module):
    def __init__(self, input_size=128, gat_hidden_dim=64, gat_out_dim=64, gat_heads=4, hidden_size=256, output_size=2, use_gat=True, debug=False):
        super(FastGrokkingRushNetWithGAT, self).__init__()
        self.use_gat = use_gat
        self.debug = debug

        if self.use_gat:
            self.gat_extractor = GATConv(input_size, gat_hidden_dim, heads=gat_heads, concat=True)
            self.fc1 = nn.Linear(gat_hidden_dim * gat_heads, hidden_size)
        else:
            self.fc1 = nn.Linear(input_size, hidden_size)

        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index=None):
        if self.use_gat:
            x = self.gat_extractor(x, edge_index)
            if self.debug:
                print(f"GAT Output: {x.shape}")
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        logits = self.fc3(x)
        return logits
