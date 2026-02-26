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

from flowgrad.snapshot import SnapshotStore


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
        from flowgrad.analyzers.health import gradient_snr_per_layer

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

        Logic:
            sensitivity = normalize(snr * w_range * w_std)
            High sensitivity → 16 or 32-bit
            Medium → 8-bit
            Low → 4-bit
        """
        # Composite sensitivity score
        sensitivity = snr * max(w_range, 1e-6) * max(w_std, 1e-6)

        if sensitivity > 1.0:
            return 16  # Very sensitive, keep high precision
        elif sensitivity > 0.01:
            return 8   # Standard quantization
        else:
            return 4   # Safe to quantize aggressively

    def recommend_mixed_precision(self) -> Dict[str, int]:
        """
        Per-layer bit-width recommendation for mixed-precision quantization.

        Returns:
            {layer_name: recommended_bits (4, 8, or 16)}
        """
        profile = self.sensitivity_profile()
        return {name: info["recommended_bits"] for name, info in profile.items()}

    def estimated_size_reduction(self) -> Dict[str, float]:
        """
        Estimate memory savings from mixed-precision quantization.

        Returns:
            {"original_bits": 32, "avg_bits": float,
             "estimated_reduction_pct": float}
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
