# app/models/fastgrok_model.py

import torch
import torch.nn as nn
from app.models.base_model import BaseModel

# fastgrok_model.py

class FastGrokModel(BaseModel):
    """
    FastGrok enhanced model for binary classification.
    """
    def __init__(self, input_size=128, hidden_size=256, output_size=2, use_gat=False, **kwargs):
        super(FastGrokModel, self).__init__(input_size, hidden_size, output_size, use_gat, **kwargs)
        # Additional layers or modifications can be added here if needed
