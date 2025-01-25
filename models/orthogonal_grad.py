# models/orthogonal_grad.py

import torch
from torch.optim import Optimizer

class OrthogonalGrad(Optimizer):
    """
    Orthogonal Gradient Optimizer to prevent naive logit scaling.
    Projects gradients orthogonal to the weight vectors.
    """
    def __init__(self, params, lr=1e-3, weight_decay=1e-5):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(OrthogonalGrad, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if weight_decay != 0:
                    grad = grad.add(param.data, alpha=weight_decay)

                # Project out the component parallel to the parameter vector
                param_norm = param.data.norm()
                if param_norm > 0:
                    parallel_component = (torch.dot(param.data.view(-1), grad.view(-1)) / (param_norm ** 2)) * param.data
                    grad = grad - parallel_component

                # Update parameters
                param.data -= lr * grad

        return loss
