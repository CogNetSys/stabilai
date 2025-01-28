# app/utils/optimizer.py

import torch
import torch.optim as optim

# optimizer.py

class OrthogonalGradientOptimizer:
    """
    Optimizer wrapper that removes the gradient component aligned with the parameter vector.
    Prevents naive loss minimization scaling and helps the model generalize.
    """
    def __init__(self, base_optimizer):
        """
        Initialize with a base optimizer (e.g., Adam).
        """
        self.base_optimizer = base_optimizer
    
    def zero_grad(self):
        """
        Zero the gradients of the base optimizer.
        """
        self.base_optimizer.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step:
        1. Project out the naive scaling direction from each parameter's gradient.
        2. Let the base optimizer perform the update.
        """
        # Gradient projection
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                w = p.data
                denom = torch.dot(w.view(-1), w.view(-1)) + 1e-12
                num = torch.dot(w.view(-1), grad.view(-1))
                projection = (num / denom)
                # Remove the projection component
                grad_orth = grad - projection * w
                p.grad.copy_(grad_orth)
        
        # Base optimizer step
        self.base_optimizer.step(closure=closure)
    
    def add_param_group(self, param_group):
        """
        Add a parameter group to the base optimizer.
        """
        self.base_optimizer.add_param_group(param_group)
