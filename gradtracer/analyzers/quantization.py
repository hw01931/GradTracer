"""
QuantizationAdvisor — Per-layer bit-width recommendation using training dynamics.

Uses FlowTracker's weight distribution and gradient SNR data to recommend
which layers can safely use lower precision (4-bit, 8-bit) vs which need full precision.

Usage:
    tracker = FlowTracker(model)
    # ... train ...
    qa = QuantizationAdvisor(tracker)
    plan = qa.recommend_mixed_precision()
    # → {"layer1.weight": 4, "layer2.weight": 8, "layer3.weight": 16}
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from gradtracer.snapshot import SnapshotStore


class QuantizationAdvisor:
    """
    Recommends per-layer quantization precision using training dynamics.

    Key insight: layers with low gradient SNR and narrow weight distribution
    can tolerate aggressive quantization. Layers with high SNR and wide
    distribution need higher precision to preserve performance.
    """

    def __init__(self, tracker):
        """
        Args:
            tracker: A FlowTracker instance with recorded training history.
        """
        self.tracker = tracker
        self.store: SnapshotStore = tracker.store

    def sensitivity_profile(self) -> Dict[str, Dict]:
        """
        For each layer, compute quantization-relevant metrics:
        - weight_range: max - min (wider = harder to quantize)
        - weight_std: standard deviation (larger = needs more bits)
        - grad_snr: gradient signal-to-noise ratio (higher = more sensitive)

        Returns:
            {layer_name: {"weight_range": float, "weight_std": float,
                          "grad_snr": float, "recommended_bits": int}}
        """
        from gradtracer.analyzers.health import gradient_snr_per_layer

        snr_data = gradient_snr_per_layer(self.store)
        profiles = {}

        for name in self.store.layer_names:
            history = self.store.get_layer_history(name)
            if not history:
                continue

            latest = history[-1]
            w_range = latest.weight_max - latest.weight_min
            w_std = latest.weight_std

            # Get latest SNR
            snr_series = snr_data.get(name, [])
            snr = snr_series[-1] if snr_series else 0.0
            if np.isinf(snr):
                snr = 100.0  # Very high signal

            # Recommend bits based on sensitivity
            bits = self._recommend_bits(w_range, w_std, snr)

            profiles[name] = {
                "weight_range": round(w_range, 4),
                "weight_std": round(w_std, 4),
                "grad_snr": round(float(snr), 6),
                "recommended_bits": bits,
            }

        return profiles

    @staticmethod
    def _recommend_bits(w_range: float, w_std: float, snr: float) -> int:
        """
        Decision logic for bit-width recommendation.
        Improved to protect sensitive layers more aggressively.
        """
        # Improved sensitivity score (weighted more towards gradient signal)
        # We use log-scale SNR to make it less prone to extreme vanishing values
        weighted_snr = np.log1p(snr)
        
        # Combined score: Signal * Variation
        # If the layer is changing fast (SNR) or has wide weights (Std), it's sensitive.
        sensitivity = weighted_snr * (w_range + w_std * 2.0)

        # New, stricter thresholds
        if sensitivity > 0.1:      # Clearly important signal
            return 16  # Keep FP32 (Recommended for sensitive layers)
        elif sensitivity > 0.001:  # Moderate signal
            return 8   # Standard Quantization
        else:                      # Near-zero signal/dead weights
            return 4   # Highly Aggressive Quantization

    def recommend_mixed_precision(self, quantile: float = 0.85) -> Dict[str, int]:
        """
        Dynamically recommends bit-widths based on the overall sensitivity distribution 
        of the current model. No hardcoded constants.

        Heuristic: Protect the top (1-quantile) most sensitive layers (FP32), 
        normal layers (INT8), and near-zero signal layers (INT4).
        
        Args:
            quantile: The threshold to determine "High Sensitivity" (top X percentile).
        """
        profile = self.sensitivity_profile()
        if not profile:
            return {}

        # 1. Compute sensitivity scores (Signal SNR * Weight Spread)
        # Using log-normalized SNR + Z-Score of weight spread for scale-invariant sensitivity
        names = list(profile.keys())
        scores = []
        for name in names:
            p = profile[name]
            # S = log(1+SNR) * WeightRange
            s = np.log1p(p["grad_snr"]) * (p["weight_range"] + 1e-9)
            scores.append(s)
        
        scores = np.array(scores)
        
        # 2. Dynamic Thresholding using statistical quantiles
        # Top 15% (by default) are 'Critical' -> FP32/16
        # Near-zero are 'Dead/Redundant' -> INT4
        high_threshold = np.quantile(scores, quantile) if len(scores) > 1 else 0.1
        low_threshold = np.quantile(scores, 0.25) if len(scores) > 1 else 0.001
        
        plan = {}
        for name, score in zip(names, scores):
            if score >= high_threshold and score > 1e-6:
                plan[name] = 16  # High Sensitivity -> FP32/16
            elif score <= low_threshold or score < 1e-9:
                plan[name] = 4   # Near-zero Signal -> INT4
            else:
                plan[name] = 8   # Normal -> INT8
                
        return plan

    def estimated_size_reduction(self) -> Dict[str, float]:
        """
        Estimate memory savings based on the dynamic plan.
        """
        plan = self.recommend_mixed_precision()
        if not plan:
            return {"original_bits": 32, "avg_bits": 32.0, "estimated_reduction_pct": 0.0}

        total_params = 0
        weighted_bits = 0
        for name in self.store.layer_names:
            history = self.store.get_layer_history(name)
            if history:
                n = history[-1].num_params
                bits = plan.get(name, 32)
                total_params += n
                weighted_bits += n * bits

        avg_bits = weighted_bits / max(total_params, 1)
        reduction = (1.0 - avg_bits / 32.0) * 100

        return {
            "original_bits": 32,
            "avg_bits": round(avg_bits, 1),
            "estimated_reduction_pct": round(reduction, 1),
        }

    def report(self) -> None:
        """Generate a human-readable quantization recommendation report."""
        profile = self.sensitivity_profile()
        reduction = self.estimated_size_reduction()

        lines = ["─" * 55, "🔢 Quantization Advisor Report", "─" * 55]

        lines.append(f"  {'Layer':<35s} {'Range':<8s} {'SNR':<10s} {'Bits':<5s}")
        lines.append("  " + "─" * 55)

        for name, info in profile.items():
            bits = info["recommended_bits"]
            emoji = "🟢" if bits <= 4 else "🟡" if bits <= 8 else "🔴"
            lines.append(
                f"  {emoji} {name:<33s} {info['weight_range']:<8.4f} "
                f"{info['grad_snr']:<10.6f} {bits}-bit"
            )

        lines.append("")
        lines.append(f"  📊 Average precision: {reduction['avg_bits']:.1f}-bit "
                      f"(from 32-bit)")
        lines.append(f"  💾 Estimated size reduction: {reduction['estimated_reduction_pct']:.1f}%")

        report = "\n".join(lines)
        print(report)

    def to_agent_xml(self) -> str:
        """Export quantization advice as XML for AI agents."""
        from xml.sax.saxutils import escape
        profile = self.sensitivity_profile()
        reduction = self.estimated_size_reduction()

        layers_xml = []
        for name, info in profile.items():
            layers_xml.append(
                f'    <layer name="{escape(name)}" '
                f'weight_range="{info["weight_range"]}" '
                f'grad_snr="{info["grad_snr"]}" '
                f'recommended_bits="{info["recommended_bits"]}" />'
            )

        return (
            "<quantization_analysis>\n"
            f"  <estimated_avg_bits>{reduction['avg_bits']}</estimated_avg_bits>\n"
            f"  <estimated_reduction_pct>{reduction['estimated_reduction_pct']}</estimated_reduction_pct>\n"
            + "\n".join(layers_xml) + "\n"
            + "</quantization_analysis>"
        )

# ------------------------------------------------------------------
# Quantization Execution Methods
# ------------------------------------------------------------------

def apply_uniform_quantization(model: 'torch.nn.Module') -> 'torch.nn.Module':
    """Apply wholesale uniform PyTorch Dynamic INT8 Quantization."""
    import torch
    import torch.nn as nn
    return torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

def apply_mixed_precision_quantization(model: 'torch.nn.Module', recommendation: Dict[str, int]) -> 'torch.nn.Module':
    """
    Apply TRUE Heterogeneous PyTorch Native Mixed-Precision Dynamic Quantization.
    Robust layers (recommending <= 8 bits) are quantized to qint8.
    Sensitive layers (recommending 16/32 bits) are kept exact in FP32.
    """
    import torch
    import torch.nn as nn
    
    # Target all Linears by default
    qconfig_spec = {nn.Linear: torch.quantization.default_dynamic_qconfig}
    
    kept_fp32 = 0
    quantized_int8 = 0
    
    for param_name, bits in recommendation.items():
        # Strip '.weight' to map param to module name
        mod_name = param_name[:-7] if param_name.endswith(".weight") else param_name
        
        if bits > 8:
            # Disable quantization for this sensitive module
            qconfig_spec[mod_name] = None
            kept_fp32 += 1
        else:
            quantized_int8 += 1
            
    print(f"  [Quantization Executor] Routing {quantized_int8} layers to INT8, leaving {kept_fp32} layers in FP32.")
    
    return torch.quantization.quantize_dynamic(
        model, qconfig_spec=qconfig_spec, dtype=torch.qint8
    )
