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

    def velocity_saliency(self) -> Dict[str, float]:
        window = min(10, self.store.num_steps)
        if window == 0:
            return {}
        velocities = {}
        for name in self.store.layer_names:
            series = self.store.get_layer_series(name, "velocity")
            velocities[name] = float(np.mean(series[-window:])) if series else 0.0
        max_v = max(velocities.values()) if velocities else 1e-10
        max_v = max(max_v, 1e-12)
        return {name: v / max_v for name, v in velocities.items()}

    def gradient_momentum(self) -> Dict[str, float]:
        results = {}
        for name in self.store.layer_names:
            series = self.store.get_layer_series(name, "grad_norm")
            if len(series) < 4:
                results[name] = 0.0
                continue
            recent = series[len(series) // 2:]
            if np.std(recent) < 1e-12:
                results[name] = 0.0
                continue
            x = np.arange(len(recent), dtype=float)
            y = np.array(recent, dtype=float)
            results[name] = float(np.polyfit(x, y, 1)[0])
        return results

    def dead_neuron_candidates(self) -> List[str]:
        dead_layers = []
        for name in self.store.layer_names:
            history = self.store.get_layer_history(name)
            if history and history[-1].dead_ratio > 0.5:
                dead_layers.append(name)
        return dead_layers

    def generate_pruning_plan(self, target_sparsity: float = 0.3) -> Dict[str, float]:
        """
        Generate a heterogeneous pruning plan allocating sparsity based on saliency.
        Highly salient layers retain full density (sparsity 0).
        Dead/low saliency layers receive heavy sparsity.
        """
        vel_sal = self.velocity_saliency()
        grad_mom = self.gradient_momentum()
        dead_set = set(self.dead_neuron_candidates())

        priorities = {}
        for name in self.store.layer_names:
            v_score = 1.0 - vel_sal.get(name, 0.5)
            g_score = 0.3 if grad_mom.get(name, 0) < 0 else 0.0
            
            history = self.store.get_layer_history(name)
            dr = history[-1].dead_ratio if history else 0.0
            d_score = dr * 0.3

            # Priority 0: extremely important, do not prune
            # Priority > 0.6: dead/useless, prune heavily
            priorities[name] = v_score * 0.4 + g_score + d_score

        # Distribute target_sparsity smoothly
        # Layers with priority > 0.4 get pruned, others get 0
        plan = {}
        for name, p in priorities.items():
            if p > 0.6:
                plan[name] = min(target_sparsity * 2.0, 0.9) # Heavy prune
            elif p > 0.4:
                plan[name] = target_sparsity # Normal prune
            else:
                plan[name] = 0.0 # Protect
                
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
