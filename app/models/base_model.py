# app/models/base_model.py

import torch
import torch.nn as nn

# base_model.py

class BaseModel(nn.Module):
    """
    Base model for binary classification.
    """
    def __init__(self, input_size=128, hidden_size=256, output_size=2, use_gat=False, **kwargs):
        super(BaseModel, self).__init__()
        self.use_gat = use_gat
        if self.use_gat:
            from app.models.surge_collapse_net import GATFeatureExtractor
            self.gat_extractor = GATFeatureExtractor(input_dim=input_size)
            self.fc1 = nn.Linear(64, hidden_size)  # Adjust based on GAT output
        else:
            self.fc1 = nn.Linear(input_size, hidden_size)
        
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        if self.use_gat:
            x = self.gat_extractor(x)
        x = self.relu(self.fc1(x))
        x = self.relu2(self.fc2(x))
        logits = self.fc3(x)
        return logits
