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
    
    def __init__(self, store):
        self.store = store
        
    def generate(self, target_sparsity: float = 0.5) -> Dict[str, Any]:
        """
        Produce the joint compression recipe based on layer health and saliency.
        """
        from gradtracer.analyzers.health import layer_health_score, gradient_snr_per_layer
        
        health_scores = layer_health_score(self.store)
        snr_data = gradient_snr_per_layer(self.store)
        
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
                
            last_entry = history[-1]
            health = health_scores.get(layer_name, 100)
            snr = snr_data.get(layer_name, [0.0])[-1] if snr_data.get(layer_name) else 0.0
            
            # Rule Engine for Mixed-Precision & Pruning
            
            # 1. Critical Information Layers (High variance, active learning)
            if snr > 1.0 or health > 90:
                quant = "FP16"  # Preserve precision
                prune = 0.0     # Don't prune active parameters
                reason = "High gradient SNR; critical learning pathway."
                
            # 2. Dying or Dead Layers (Zero activations, stagnation)
            elif last_entry.dead_ratio > 0.5 or health < 30:
                quant = "INT4"  # Highest quantization
                prune = 0.8     # Aggressive structural pruning
                reason = "Severe stagnation or dead neurons detected."
                
            # 3. Dense but Low-Variance Layers (Feed-forward blocks, highly stable)
            else:
                quant = "INT8"  # Standard quantization
                prune = target_sparsity
                reason = "Stable, low-variance feed-forward representation."
                
            recipe["layers"][layer_name] = {
                "quantization": quant,
                "prune_ratio": round(prune, 2),
                "reason": reason,
                "health_score": round(health, 1),
                "dead_ratio": round(last_entry.dead_ratio, 2)
            }
            
        return recipe
        
    def export_json(self, path: str = "gradtracer_recipe.json"):
        """Export the recipe to a JSON file for the VS Code Extension or deployment."""
        recipe = self.generate()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(recipe, f, indent=4)
        return recipe
