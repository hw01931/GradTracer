"""
Pruning module — Combines pruning recommendation (Saliency) and physical application.

Provides `PruningAdvisor` to determine which layers are safe to prune based on
training dynamics, and utility methods to actually apply unstructured or structured pruning.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from gradtracer.snapshot import SnapshotStore


class PruningAdvisor:
    """
    Analyze layer-level saliency using training dynamics from FlowTracker.
    Recommends which layers should be pruned and by how much.
    """

    def __init__(self, tracker):
        self.tracker = tracker
        self.store = tracker.store

    def fisher_saliency(self) -> Dict[str, float]:
        """
        Fisher Information proxy: Σ (grad_k^2).
        Measures how sensitive the loss is to small changes in weights.
        """
        fisher = {}
        for name in self.store.layer_names:
            series = self.store.get_layer_series(name, "grad_norm")
            if not series:
                fisher[name] = 0.0
                continue
            # Use moving average of squared grad norms as a proxy for the diagonal of the Fisher Matrix
            recent_grads = np.array(series[-10:], dtype=float)
            fisher[name] = float(np.mean(recent_grads**2))
        
        # Relative Importance
        total = sum(fisher.values()) + 1e-12
        return {name: f / total for name, f in fisher.items()}

    def generate_pruning_plan(self, target_sparsity: float = 0.3) -> Dict[str, float]:
        """
        Generate a heterogeneous pruning plan that EXACTLY meets the global target_sparsity.
        Uses a binary search over sparsity multipliers to follow the inverse-Fisher distribution.
        """
        fisher = self.fisher_saliency()
        
        # 1. Map layers to weights and parameter counts
        model_p = self.tracker.model
        param_counts = {}
        for name, module in model_p.named_modules():
            if name in self.store.layer_names or f"{name}.weight" in self.store.layer_names:
                if hasattr(module, "weight"):
                    param_counts[name] = module.weight.numel()
        
        if not param_counts:
            return {}
            
        total_params = sum(param_counts.values())
        budget_to_prune = total_params * target_sparsity
        
        # 2. Importance-based Prunability
        # Low Fisher -> High Prunability. We protect layers with high gradient energy.
        importance = []
        for name in param_counts.keys():
            val = fisher.get(name, 0.0) or fisher.get(f"{name}.weight", 0.0)
            importance.append(val)
        
        importance = np.array(importance)
        # Power-scaling to protect sensitive layers more aggressively (e.g., 1/x^2)
        prunability = 1.0 / (importance + 1e-9)**0.5
        prunability /= prunability.sum()
        
        # 3. Solver: Sum(param_counts[i] * sparsity[i]) == budget_to_prune
        # where sparsity[i] = clip(lambda * prunability[i], 0.0, 0.95)
        low, high = 0.0, 1.0 / (prunability.min() + 1e-12)
        best_lambda = low
        for _ in range(30):
            mid = (low + high) / 2
            current_pruned = sum(count * np.clip(mid * prunability[i], 0.0, 0.95) 
                                for i, (name, count) in enumerate(param_counts.items()))
            if current_pruned < budget_to_prune:
                low = mid
            else:
                high = mid
            best_lambda = mid
            
        plan = {name: float(np.clip(best_lambda * prunability[i], 0.0, 0.95)) 
                for i, name in enumerate(param_counts.keys())}
        
        return plan

def apply_global_pruning(model: nn.Module, sparsity: float) -> nn.Module:
    """Apply standard global unstructured pruning uniformly."""
    if sparsity <= 0:
        return model
        
    model_p = model  # Assumes operated in place or already cloned upstream
    params_to_prune = []
    for name, module in model_p.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if hasattr(module, "weight"):
                params_to_prune.append((module, "weight"))

    if params_to_prune:
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity,
        )
        for module, param_name in params_to_prune:
            try:
                prune.remove(module, param_name)
            except ValueError:
                pass
    return model_p

def apply_heterogeneous_pruning(model: nn.Module, plan: Dict[str, float]) -> nn.Module:
    """Apply varied sparsity to specific layers based on a pruning plan."""
    model_p = model
    
    for name, module in model_p.named_modules():
        # Match module names with plan (plan keys might be '.weight' appended)
        sparsity = None
        if name in plan:
            sparsity = plan[name]
        elif f"{name}.weight" in plan:
            sparsity = plan[f"{name}.weight"]
            
        if sparsity and sparsity > 0 and isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if hasattr(module, "weight"):
                prune.l1_unstructured(module, name="weight", amount=sparsity)
                prune.remove(module, "weight")
                
    return model_p
