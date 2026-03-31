"""
Robustness analyzer — utilities to simulate production noise and drift.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Union


class NoiseInjectionWrapper(nn.Module):
    """
    Wraps a model or a specific layer to inject noise during training,
    simulating production-level data instability.
    
    Args:
        module: The layer or model to wrap.
        noise_level: Standard deviation of Gaussian noise.
        label_noise: Probability of flipping binary labels (0 to 1).
        feature_drift: Scaling factor to simulate feature attenuation.
    """
    def __init__(
        self, 
        module: nn.Module, 
        noise_level: float = 0.05, 
        label_noise: float = 0.0,
        feature_drift: float = 1.0
    ):
        super().__init__()
        self.module = module
        self.noise_level = noise_level
        self.label_noise = label_noise
        self.feature_drift = feature_drift

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.training:
            # Feature noise
            if self.noise_level > 0:
                noise = torch.randn_like(x) * self.noise_level
                x = x + noise
            
            # Feature drift (attenuation)
            if self.feature_drift != 1.0:
                x = x * self.feature_drift
                
        return self.module(x, *args, **kwargs)


def apply_label_noise(labels: torch.Tensor, p: float = 0.05) -> torch.Tensor:
    """
    Randomly flips classification labels with probability p.
    """
    if p <= 0:
        return labels
    mask = torch.rand_like(labels.float()) < p
    # For binary labels (0/1): flip using XOR or 1-val
    if labels.dtype in (torch.int64, torch.int32):
        # Assume binary for simplicity, or handle multi-class
        num_classes = labels.max().item() + 1
        if num_classes == 2:
            return torch.where(mask, 1 - labels, labels)
        else:
            # Random class shift for multi-class
            random_labels = torch.randint_all_like(labels, 0, int(num_classes))
            return torch.where(mask, random_labels, labels)
    return labels
