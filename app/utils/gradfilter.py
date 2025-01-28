# app/utils/gradfilter.py

import torch
from collections import deque

# gradfilter.py

def gradfilter_ma(model, grads=None, window_size=100, lamb=5.0, filter_type='mean', warmup=True):
    """
    Moving Average (MA) gradient filter for Grokfast-MA.

    Parameters:
    - model: the neural network model
    - grads: dictionary of deques storing past gradients
    - window_size: size of the moving window
    - lamb: amplification factor
    - filter_type: 'mean' or 'sum'
    - warmup: if True, wait until enough gradients are collected
    """
    if grads is None:
        grads = {n: deque(maxlen=window_size) for n, p in model.named_parameters() if p.requires_grad}

    for n, p in model.named_parameters():
        if p.requires_grad:
            grads[n].append(p.grad.data.detach().clone())

            if not warmup or len(grads[n]) == window_size:
                if filter_type == "mean":
                    avg = torch.mean(torch.stack(list(grads[n])), dim=0)
                elif filter_type == "sum":
                    avg = torch.sum(torch.stack(list(grads[n])), dim=0)
                else:
                    raise ValueError(f"Unrecognized filter_type {filter_type}")
                p.grad.data = p.grad.data + avg * lamb

    return grads

def gradfilter_ema(model, grads=None, alpha=0.98, lamb=2.0):
    """
    Exponential Moving Average (EMA) gradient filter for Grokfast-EMA.

    Parameters:
    - model: the neural network model
    - grads: dictionary of EMA values for each parameter
    - alpha: momentum parameter for EMA
    - lamb: amplification factor
    """
    if grads is None:
        grads = {n: torch.zeros_like(p.grad.data) for n, p in model.named_parameters() if p.requires_grad}

    for n, p in model.named_parameters():
        if p.requires_grad:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data = p.grad.data + grads[n] * lamb

    return grads
