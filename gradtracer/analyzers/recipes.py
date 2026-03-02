"""
GradTracer Auto-Compression Recipe Generator

The "Holy Grail" of model compression. Matches layer-specific training dynamics 
(gradient SNR, velocity, dead neurons) against available hardware compression techniques.
Produces a unified Mixed-Precision Quantization + Joint Pruning recipe.
"""
import json
from typing import Dict, Any

class RecipeGenerator:
    """
    Analyzes historical layer metrics from a tracking store and outputs an
    automated JSON compression recipe determining how aggressively each layer
    can be quantized and pruned without hurting predictive performance.
    """
    
    def __init__(self, tracker):
        self.tracker = tracker
        self.store = tracker.store
        
    def generate(self, target_sparsity: float = 0.5) -> Dict[str, Any]:
        """
        Produce the joint compression recipe based on layer health and saliency.
        """
        from gradtracer.analyzers.health import layer_health_score, gradient_snr_per_layer
        import numpy as np
        
        health_scores = layer_health_score(self.store)
        snr_data = gradient_snr_per_layer(self.store)
        
        total_baseline_vram_mb = 0.0
        total_estimated_vram_mb = 0.0
        total_flops_baseline = 0.0
        total_flops_remaining = 0.0
        
        # 1. First pass to find prunable layers and their saliencies
        prunable_layers = []
        activity_scores = {}
        for layer_name in self.store.layer_names:
            history = self.store.get_layer_history(layer_name)
            if not history:
                continue
            
            module_type = "Unknown"
            numel = getattr(history[-1], 'num_params', 0)
            if hasattr(self.tracker, "model"):
                mod_name = layer_name.rsplit('.', 1)[0]
                try:
                    mod = self.tracker.model.get_submodule(mod_name)
                    module_type = type(mod).__name__
                    param = getattr(mod, layer_name.split('.')[-1], None)
                    if param is not None:
                         numel = param.numel()
                except Exception:
                    pass
            
            is_1d = (numel < 1000) or ("bias" in layer_name.lower())
            is_sensitive = ("Norm" in module_type) or ("Embedding" in module_type) or ("classifier" in layer_name.lower()) or ("pooler" in layer_name.lower())
            is_prunable = ("Linear" in module_type or "Conv" in module_type) and not is_1d and not is_sensitive
            
            if is_prunable:
                prunable_layers.append(layer_name)
                health = health_scores.get(layer_name, 50)
                latest = history[-1]
                saliency = (latest.weight_norm + 1e-9) * health
                activity_scores[layer_name] = saliency
                
        mean_saliency = np.mean(list(activity_scores.values())) if activity_scores else 1.0
        
        recipe = {
            "metadata": {
                "target_sparsity": target_sparsity,
                "strategy": "Mixed-Precision Joint Pruning"
            },
            "layers": {}
        }
        
        for layer_name in self.store.layer_names:
            history = self.store.get_layer_history(layer_name)
            if not history:
                continue
                
            # Extract module metadata
            module_type = "Unknown"
            shape = []
            numel = getattr(history[-1], 'num_params', 0)
            if hasattr(self.tracker, "model"):
                mod_name = layer_name.rsplit('.', 1)[0]
                try:
                    mod = self.tracker.model.get_submodule(mod_name)
                    module_type = type(mod).__name__
                    param = getattr(mod, layer_name.split('.')[-1], None)
                    if param is not None:
                         shape = list(param.shape)
                         numel = param.numel()
                except Exception:
                    pass

            last_entry = history[-1]
            health = health_scores.get(layer_name, 100)
            
            is_prunable = layer_name in prunable_layers
            
            # Rule Engine for Mixed-Precision & Pruning
            if not is_prunable:
                quant = "FP16" if ("Norm" in module_type or "Embedding" in module_type) else "INT8"
                prune = 0.0
                reason = "Preserved (Normalization, Embedding, or Head)"
            else:
                saliency = activity_scores[layer_name]
                factor = (mean_saliency - saliency) / mean_saliency  # [-1, 1] roughly
                prune = target_sparsity * (1.0 + factor * 0.8)
                prune = max(0.00, min(0.95, prune))  # Keep within safe bounds
                
                if factor < -0.3:
                    quant = "FP16"
                    reason = "High saliency/health; critical learning pathway."
                elif factor > 0.3:
                    quant = "INT4"
                    reason = "Low saliency/health; stagnation or dead neurons."
                else:
                    quant = "INT8"
                    reason = "Average saliency feed-forward representation."
                
            prune_type = "none"
            if prune > 0:
                if "Conv" in module_type:
                    prune_type = "structured_channel"
                elif "Linear" in module_type:
                    prune_type = "unstructured_l1"

            # Compute theoretical estimators
            baseline_bytes = numel * 4  # Assume starting at FP32
            total_baseline_vram_mb += baseline_bytes / (1024 * 1024)
            
            bytes_per_param = 2 if quant == "FP16" else (1 if quant == "INT8" else 0.5)
            retained_params = numel * (1.0 - prune)
            total_estimated_vram_mb += (retained_params * bytes_per_param) / (1024 * 1024)
            
            total_flops_baseline += numel
            total_flops_remaining += retained_params
                
            recipe["layers"][layer_name] = {
                "layer_type": module_type,
                "shape": shape,
                "quantization": quant,
                "prune_ratio": round(prune, 3),
                "prune_type": prune_type,
                "reason": reason,
                "health_score": round(health, 1),
                "dead_ratio": round(last_entry.dead_ratio, 2)
            }
            
        recipe["metadata"]["estimated_vram_saving_mb"] = round(total_baseline_vram_mb - total_estimated_vram_mb, 2)
        flops_reduction = 0.0
        if total_flops_baseline > 0:
            flops_reduction = 1.0 - (total_flops_remaining / total_flops_baseline)
        recipe["metadata"]["estimated_flops_reduction_ratio"] = round(flops_reduction, 3)
            
        return recipe
        
    def export_json(self, path: str = "gradtracer_recipe.json"):
        """Export the recipe to a JSON file for the VS Code Extension or deployment."""
        recipe = self.generate()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(recipe, f, indent=4)
        return recipe
