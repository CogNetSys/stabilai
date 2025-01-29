import torch
import torch.nn as nn
from app.models.base_model import BaseModel

# noise_model.py

class NoiseModel(BaseModel):
    """
    Noise-injected model for binary classification.
    """
    def __init__(self, input_size=128, hidden_size=256, output_size=2, use_gat=False, noise_level=0.0, **kwargs):
        super(NoiseModel, self).__init__(input_size, hidden_size, output_size, use_gat, **kwargs)
        self.noise_level = noise_level
    
    def forward(self, x):
        logits, activations = super().forward(x)
        # Optionally add noise to logits or activations
        if self.training and self.noise_level > 0.0:
            noise = torch.randn_like(logits) * self.noise_level
            logits = logits + noise
            activations['fc3_noisy'] = logits  # Optionally store the noisy logits
        return logits, activations
