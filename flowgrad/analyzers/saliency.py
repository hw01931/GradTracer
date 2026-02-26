"""
SaliencyAnalyzer — Dynamic, training-aware layer saliency scoring.

Unlike static L1 pruning (which only looks at weight magnitude),
SaliencyAnalyzer uses FlowTracker's training dynamics (velocity, gradient trends,
dead neuron ratio) to determine which layers are truly important.

Usage:
    tracker = FlowTracker(model)
    # ... train ...
    sa = SaliencyAnalyzer(tracker)
    priority = sa.pruning_priority()
    # → [("layer3.weight", 0.92, "Low velocity + declining gradient"), ...]
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from flowgrad.snapshot import SnapshotStore


class SaliencyAnalyzer:
    """
    Analyze layer-level saliency using training dynamics from FlowTracker.

    Scores each layer by how "alive" and important it is, using:
    - Velocity (how actively weights are changing)
    - Gradient momentum (is the gradient signal growing or dying?)
    - Dead neuron ratio (already effectively pruned)
    """

    def __init__(self, tracker):
        """
        Args:
            tracker: A FlowTracker instance with recorded training history.
        """
        self.tracker = tracker
        self.store: SnapshotStore = tracker.store

    def velocity_saliency(self) -> Dict[str, float]:
        """
        Score each layer by recent velocity (weight change rate).

        Higher velocity = still learning = more important.

        Logic:
            saliency = mean(velocity[-N:]) / max(all_velocities)
            where N = min(10, total_steps)

        Returns:
            {layer_name: saliency_score} where 0 = dead, 1 = most active
        """
        window = min(10, self.store.num_steps)
        if window == 0:
            return {}

        velocities = {}
        for name in self.store.layer_names:
            series = self.store.get_layer_series(name, "velocity")
            if series:
                recent = series[-window:]
                velocities[name] = float(np.mean(recent))
            else:
                velocities[name] = 0.0

        max_v = max(velocities.values()) if velocities else 1e-10
        if max_v < 1e-12:
            max_v = 1e-12

        return {name: v / max_v for name, v in velocities.items()}

    def gradient_momentum(self) -> Dict[str, float]:
        """
        Compute gradient momentum: is the gradient signal growing or dying?

        Logic:
            Fit a linear regression on grad_norm over the last 50% of steps.
            Negative slope = gradient is declining = layer becoming less important.

        Returns:
            {layer_name: slope} where negative = safe to prune
        """
        results = {}
        for name in self.store.layer_names:
            series = self.store.get_layer_series(name, "grad_norm")
            if len(series) < 4:
                results[name] = 0.0
                continue

            half = len(series) // 2
            recent = series[half:]
            x = np.arange(len(recent), dtype=float)
            y = np.array(recent, dtype=float)

            if np.std(y) < 1e-12:
                results[name] = 0.0
                continue

            # Simple linear regression slope
            slope = float(np.polyfit(x, y, 1)[0])
            results[name] = slope

        return results

    def dead_neuron_candidates(self) -> List[str]:
        """
        Layers with dead_ratio > 0.5 — already effectively pruned by training.

        Returns:
            List of layer names with majority dead neurons.
        """
        dead_layers = []
        for name in self.store.layer_names:
            history = self.store.get_layer_history(name)
            if history and history[-1].dead_ratio > 0.5:
                dead_layers.append(name)
        return dead_layers

    def pruning_priority(self) -> List[Tuple[str, float, str]]:
        """
        Combine all signals into a ranked pruning recommendation.

        Scoring:
            priority = (1 - velocity_saliency) * 0.4
                     + (1 if gradient_declining else 0) * 0.3
                     + dead_ratio * 0.3

        Returns:
            [(layer_name, priority_score, reason), ...] sorted highest first.
            Higher priority = prune this first.
        """
        vel_sal = self.velocity_saliency()
        grad_mom = self.gradient_momentum()
        dead_set = set(self.dead_neuron_candidates())

        results = []
        for name in self.store.layer_names:
            reasons = []

            # Velocity component (40%)
            v_score = 1.0 - vel_sal.get(name, 0.5)
            if vel_sal.get(name, 1.0) < 0.1:
                reasons.append("Near-zero velocity")

            # Gradient momentum component (30%)
            slope = grad_mom.get(name, 0)
            g_score = 0.3 if slope < 0 else 0.0
            if slope < 0:
                reasons.append("Declining gradient trend")

            # Dead neuron component (30%)
            d_score = 0.0
            history = self.store.get_layer_history(name)
            if history:
                dr = history[-1].dead_ratio
                d_score = dr * 0.3
                if name in dead_set:
                    reasons.append(f"Dead neurons: {dr*100:.0f}%")

            priority = v_score * 0.4 + g_score + d_score
            reason_str = "; ".join(reasons) if reasons else "Normal"
            results.append((name, round(priority, 3), reason_str))

        results.sort(key=lambda x: -x[1])
        return results

    def report(self) -> None:
        """Generate a human-readable saliency report."""
        lines = ["─" * 50, "🔬 Dynamic Saliency Report", "─" * 50]

        priority = self.pruning_priority()
        for name, score, reason in priority:
            emoji = "🟢" if score > 0.6 else "🟡" if score > 0.3 else "🔴"
            label = "PRUNE FIRST" if score > 0.6 else "MODERATE" if score > 0.3 else "KEEP"
            lines.append(f"  {emoji} {name:<35s} priority={score:.2f} ({label})")
            lines.append(f"     → {reason}")

        report = "\n".join(lines)
        print(report)

    def to_agent_xml(self) -> str:
        """Export saliency analysis as XML for AI agents."""
        from xml.sax.saxutils import escape
        priority = self.pruning_priority()
        findings = []
        for name, score, reason in priority:
            findings.append(
                f'    <layer name="{escape(name)}" priority="{score:.3f}" '
                f'reason="{escape(reason)}" />'
            )
        return (
            "<saliency_analysis>\n"
            + "\n".join(findings) + "\n"
            + "</saliency_analysis>"
        )
