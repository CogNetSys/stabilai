# app/utils/losses.py

import torch
import torch.nn as nn

# losses.py

class StableMaxCrossEntropy(nn.Module):
    """
    Cross-Entropy loss with a stable max-based function to prevent numerical issues.
    """
    def __init__(self):
        super(StableMaxCrossEntropy, self).__init__()
    
    def forward(self, logits, targets):
        """
        logits: [batch_size, n_classes]
        targets: [batch_size] with class indices
        """
        # Define a "StableMax" function:
        #   s(x) = x + 1 for x >= 0
        #        = 1 / (1 - x) for x < 0
        stable_logits = torch.where(
            logits >= 0,
            logits + 1,
            1.0 / (1.0 - logits)
        )
        # Sum across classes to get denominator
        sum_logits = torch.sum(stable_logits, dim=1, keepdim=True) + 1e-8
        probs = stable_logits / sum_logits
    
        # Negative log likelihood
        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        correct_probs = probs[batch_indices, targets]
        loss = -torch.log(correct_probs + 1e-8).mean()
    
        return loss
