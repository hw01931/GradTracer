"""
InterpretationAdvisor — Mechanistic Interpretability & Training Dynamics XAI.

This module implements 2025 SOTA XAI concepts based on training dynamics:
1. Grokking Progress (Memorization vs Circuit Formation)
2. Shortcut Circuit Detection (Gradient Starvation)
3. Epistemic Uncertainty Estimation (via Gradient Variance Trajectories)
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Any
from copy import deepcopy

from gradtracer.snapshot import SnapshotStore


class InterpretationAdvisor:
    """
    Translates raw gradient trajectories into Mechanistic Interpretability insights.
    
    Instead of post-hoc attribution (like SHAP), this advisor analyzes the 
    'Training Dynamics' to explain *why* and *how* the model learned its representations.
    """

    def __init__(self, tracker: Any):
        """
        Args:
            tracker: A FlowTracker instance containing the SnapshotStore.
        """
        self.tracker = tracker
        self.store: SnapshotStore = tracker.store

    def grokking_progress(self) -> Dict[str, Dict[str, Any]]:
        """
        Evaluates the 'Grokking' phase of each layer.
        
        Theory (Nanda et al., 2023): Models go through Memorization -> Circuit Formation.
        - Memorization Phase: High gradient variance, high parameter updates (noisy fitting).
        - Circuit Formation (Grokking): Gradient variance drops significantly, weights stabilize into robust circuits.
        
        Returns:
            Dict mapping layer name to its Grokking profile:
            {
                "phase": "memorization" | "circuit_formation" | "collapse",
                "progress_score": float (0.0 to 1.0, 1.0 = fully grokked),
                "superposition_risk": bool
            }
        """
        results = {}
        for name in self.store.layer_names:
            series_vel = self.store.get_layer_series(name, "velocity")
            series_norm = self.store.get_layer_series(name, "grad_norm")
            
            if len(series_vel) < 5:
                results[name] = {"phase": "insufficient_data", "progress_score": 0.0, "superposition_risk": False}
                continue
                
            # Analyze the latter half of the training trajectory
            recent_window = min(len(series_vel) // 2, 20)
            recent_vel = series_vel[-recent_window:]
            recent_norm = series_norm[-recent_window:]
            
            vel_variance = np.var(recent_vel)
            norm_trend = np.polyfit(np.arange(len(recent_norm)), recent_norm, 1)[0]
            
            # High variance indicates ongoing shuffling (Superposition collapse / Memorization)
            if vel_variance > 0.5:
                phase = "memorization"
                progress = max(0.0, 1.0 - vel_variance)
                sup_risk = True
            elif norm_trend < -1e-4 and vel_variance < 0.1:
                phase = "circuit_formation"
                progress = min(1.0, 0.5 + (0.1 - vel_variance)*5.0)
                sup_risk = False
            else:
                phase = "transitional"
                progress = 0.5
                sup_risk = False
                
            history = self.store.get_layer_history(name)
            if history and history[-1].dead_ratio > 0.4:
                phase = "collapse"
                progress = 0.0
                sup_risk = True

            results[name] = {
                "phase": phase,
                "progress_score": round(float(progress), 4),
                "superposition_risk": sup_risk
            }
            
        return results

    def detect_shortcut_learning(self) -> Dict[str, Any]:
        """
        Detects 'Gradient Starvation' (Pezeshki et al., 2020), a strong indicator of Shortcut Learning.
        
        Theory: If a shallow/spurious feature easily minimizes the loss, the gradients flowing to 
        deeper/complex semantic layers will "starve" (their relative proportion of the gradient 
        energy will drop to near zero), while the shortcut layer monopolizes the flow.
        """
        starved_layers = []
        dominant_layers = []
        
        # Calculate total gradient energy over time to normalize
        num_steps = len(self.store.get_layer_series(self.store.layer_names[0], "grad_norm"))
        if num_steps < 10:
            return {"shortcut_detected": False, "starved_circuits": [], "dominant_circuits": []}
            
        total_flow = np.zeros(num_steps)
        layer_flows = {}
        for name in self.store.layer_names:
            series = np.array(self.store.get_layer_series(name, "grad_norm"))
            layer_flows[name] = series
            total_flow += series
            
        total_flow = total_flow + 1e-12 # prevent div by zero
        
        for name, series in layer_flows.items():
            if "classifier" in name or "out" in name:
                continue # Skip the final layer which naturally dominates 
                
            relative_flow = series / total_flow
            q_len = max(num_steps // 4, 1)
            early_share = np.mean(relative_flow[:q_len])
            late_share = np.mean(relative_flow[-q_len:])
            
            # Identify heavily starved deeper circuits
            if early_share > 0.02 and late_share < (early_share * 0.5):
                starved_layers.append(name)
            
            # Identify dominant shortcut circuits
            elif early_share > 0.01 and late_share > (early_share * 1.2) and late_share > 0.05:
                dominant_layers.append(name)
                
        is_shortcut = False
        if len(starved_layers) > 0 and len(dominant_layers) > 0:
            is_shortcut = True
            
        return {
            "shortcut_detected": is_shortcut,
            "starved_circuits": starved_layers,
            "dominant_circuits": dominant_layers,
            "explanation": "Shortcut Circuit formed" if is_shortcut else "Gradient flow is balanced."
        }
        
    def epistemic_uncertainty_profile(self) -> Dict[str, float]:
        """
        Estimates Epistemic (Model) Uncertainty using gradient trajectory variance.
        
        Theory (Kendall & Gal, 2017): Areas of the model that experienced highly
        oscillating gradients (high variance) during training represent boundaries
        the model is uncertain about, even if inference confidence (Softmax) is high.
        
        Returns:
            Dict mapping layer name to Epistemic Uncertainty Score (0.0 to 1.0).
        """
        uncertainties = {}
        for name in self.store.layer_names:
            history = self.store.get_layer_history(name)
            if not history:
                uncertainties[name] = 0.0
                continue
                
            # We use the trajectory of cosine similarities (direction changes)
            cos_sims = [h.cosine_sim for h in history if hasattr(h, 'cosine_sim') and h.cosine_sim is not None]
            if len(cos_sims) < 3:
                uncertainties[name] = 0.0
                continue
                
            # If the gradient direction kept flipping (low or negative cosine sim), 
            # the model struggled to find a minimum, indicating high epistemic uncertainty.
            mean_sim = np.mean(cos_sims)
            
            # map mean_sim from [-1, 1] to Uncertainty [1, 0]
            # 1.0 sim (straight path) -> 0.0 uncertainty
            # 0.0 sim (orthogonal/random path) -> 0.5 uncertainty
            # -1.0 sim (oscillating path) -> 1.0 uncertainty
            uncertainty = max(0.0, min(1.0, (1.0 - mean_sim) / 2.0))
            
            # Incorporate Zombie ratio (explicit oscillation)
            zombie_ratio = history[-1].zombie_ratio if hasattr(history[-1], 'zombie_ratio') else 0.0
            
            final_uncertainty = 0.7 * uncertainty + 0.3 * zombie_ratio
            uncertainties[name] = round(float(final_uncertainty), 4)
            
        return uncertainties

    def report(self) -> None:
        """Prints a human-readable Mechanistic Interpretability report."""
        grokking = self.grokking_progress()
        shortcuts = self.detect_shortcut_learning()
        uncertainty = self.epistemic_uncertainty_profile()
        
        lines = ["\n" + "═" * 65]
        lines.append(" 🧠 Mechanistic Interpretability & Training Dynamics Report")
        lines.append("═" * 65)
        
        lines.append("\n[1] Grokking & Circuit Formation Phase")
        lines.append(f"  {'Layer':<25s} | {'Phase':<20s} | {'Progress':<10s}")
        lines.append("  " + "-" * 58)
        
        for name, info in grokking.items():
            phase = info["phase"]
            prog = f"{info['progress_score']*100:.1f}%"
            icon = "⚡" if phase == "memorization" else "🧩" if phase == "circuit_formation" else "💀"
            lines.append(f"  {icon} {name:<23s} | {phase:<20s} | {prog:<10s}")

        lines.append("\n[2] Shortcut Learning & Gradient Starvation (Pezeshki et al.)")
        if shortcuts["shortcut_detected"]:
            lines.append("  🚨 WARNING: Shortcut Circuit Detected!")
            lines.append(f"  - Dominant (Shortcut) Layers: {', '.join(shortcuts['dominant_circuits'])}")
            lines.append(f"  - Starved (Abandoned) Layers: {', '.join(shortcuts['starved_circuits'])}")
            lines.append("  * Diagnosis: The model found a spurious feature and starved deeper representations.")
        else:
            lines.append("  ✅ No catastrophic gradient starvation detected. Information flow is healthy.")

        lines.append("\n[3] Epistemic Uncertainty Profile (Gradient Variance)")
        lines.append("  (High uncertainty means the model is guessing, despite high Softmax confidence)")
        for name, uncert in uncertainty.items():
            bar_len = int(uncert * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"  {name:<25s} {bar} {uncert*100:.1f}%")
            
        lines.append("═" * 65 + "\n")
        print("\n".join(lines))
