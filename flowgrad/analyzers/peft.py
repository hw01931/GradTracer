"""
PEFTTracker — LoRA and Parameter-Efficient Fine-Tuning diagnostics.

Analyzes LoRA (Low-Rank Adaptation) rank allocation, adapter health,
and recommends optimal per-layer ranks using training dynamics.

Usage:
    tracker = FlowTracker(model)
    # ... fine-tune with LoRA adapters ...
    pt = PEFTTracker(tracker)
    ranks = pt.recommend_ranks()
    # → {"layer1": 16, "layer2": 4, "layer3": 8}
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from flowgrad.snapshot import SnapshotStore


class PEFTTracker:
    """
    Track and optimize LoRA / PEFT adapter configurations.

    Uses FlowTracker dynamics to determine:
    - Which layers benefit most from adapters (high velocity = actively learning)
    - Optimal rank per layer (high sensitivity = needs higher rank)
    - Which adapters are underutilized (can be pruned to save memory)
    """

    def __init__(self, tracker, adapter_layers: Optional[List[str]] = None):
        """
        Args:
            tracker: FlowTracker instance tracking a model with LoRA/adapters.
            adapter_layers: Optional list of adapter layer names.
                            If None, auto-detects layers containing 'lora', 'adapter'.
        """
        self.tracker = tracker
        self.store: SnapshotStore = tracker.store

        if adapter_layers is not None:
            self.adapter_layers = adapter_layers
        else:
            self.adapter_layers = self._auto_detect_adapters()

        self.base_layers = [n for n in self.store.layer_names
                           if n not in self.adapter_layers]

    def _auto_detect_adapters(self) -> List[str]:
        """Auto-detect adapter layers by name pattern."""
        keywords = ["lora", "adapter", "ia3", "prefix", "prompt"]
        adapters = []
        for name in self.store.layer_names:
            if any(kw in name.lower() for kw in keywords):
                adapters.append(name)
        return adapters

    def adapter_utilization(self) -> Dict[str, Dict]:
        """
        Measure how actively each adapter layer is being used.

        Logic:
            utilization = velocity / max_velocity_across_all_adapters
            + gradient_norm_ratio (adapter vs base layer)

        Returns:
            {adapter_layer: {"velocity": float, "utilization": float,
                             "status": "ACTIVE" | "UNDERUTILIZED" | "DEAD"}}
        """
        if not self.adapter_layers:
            return {}

        window = min(10, self.store.num_steps)
        if window == 0:
            return {}

        velocities = {}
        for name in self.adapter_layers:
            series = self.store.get_layer_series(name, "velocity")
            v = float(np.mean(series[-window:])) if series else 0.0
            velocities[name] = v

        max_v = max(velocities.values()) if velocities.values() else 1e-10
        if max_v < 1e-12:
            max_v = 1e-12

        results = {}
        for name, v in velocities.items():
            util = v / max_v
            if util > 0.3:
                status = "ACTIVE"
            elif util > 0.05:
                status = "UNDERUTILIZED"
            else:
                status = "DEAD"

            results[name] = {
                "velocity": round(v, 6),
                "utilization": round(util, 3),
                "status": status,
            }

        return results

    def recommend_ranks(self, budget: Optional[int] = None) -> Dict[str, int]:
        """
        Recommend LoRA rank per layer based on training dynamics.

        Logic:
            - High velocity + high grad_norm → needs high rank (16-64)
            - Medium activity → moderate rank (4-16)
            - Low/dead → rank 1 or remove entirely

        Args:
            budget: Optional total rank budget across all layers.

        Returns:
            {layer_name: recommended_rank}
        """
        window = min(10, self.store.num_steps)
        if window == 0:
            return {name: 8 for name in self.store.layer_names}

        # Score all layers (not just adapters — recommend where to PUT adapters)
        scores = {}
        for name in self.store.layer_names:
            v_series = self.store.get_layer_series(name, "velocity")
            g_series = self.store.get_layer_series(name, "grad_norm")

            v = float(np.mean(v_series[-window:])) if v_series else 0.0
            g = float(np.mean(g_series[-window:])) if g_series else 0.0
            scores[name] = v * g  # combined activity score

        max_score = max(scores.values()) if scores.values() else 1e-10
        if max_score < 1e-12:
            max_score = 1e-12

        # Normalize and map to rank
        ranks = {}
        for name, s in scores.items():
            normalized = s / max_score
            if normalized > 0.7:
                rank = 64
            elif normalized > 0.4:
                rank = 16
            elif normalized > 0.1:
                rank = 8
            elif normalized > 0.01:
                rank = 4
            else:
                rank = 1  # Minimal or skip

            ranks[name] = rank

        # Apply budget constraint if given
        if budget is not None:
            total = sum(ranks.values())
            if total > budget:
                scale = budget / total
                ranks = {n: max(1, int(r * scale)) for n, r in ranks.items()}

        return ranks

    def adapter_vs_base_analysis(self) -> Dict[str, Dict]:
        """
        Compare adapter and base layer dynamics.

        Shows whether adapters are learning faster/slower than base layers.

        Returns:
            {"adapter_avg_velocity": float, "base_avg_velocity": float,
             "adapter_efficiency": float, "recommendation": str}
        """
        window = min(10, self.store.num_steps)
        if window == 0:
            return {}

        def _avg_velocity(layer_list):
            vels = []
            for name in layer_list:
                series = self.store.get_layer_series(name, "velocity")
                if series:
                    vels.append(float(np.mean(series[-window:])))
            return float(np.mean(vels)) if vels else 0.0

        adapter_v = _avg_velocity(self.adapter_layers)
        base_v = _avg_velocity(self.base_layers)

        efficiency = adapter_v / max(base_v, 1e-10)

        if efficiency > 10:
            rec = "Adapters are learning actively. LoRA is working well."
        elif efficiency > 1:
            rec = "Adapters are learning slightly faster than base. Normal behavior."
        elif efficiency > 0.1:
            rec = "Adapters are slower than base layers. Consider increasing LoRA rank."
        else:
            rec = "Adapters are barely learning. Check if adapters are properly attached and unfrozen."

        return {
            "adapter_avg_velocity": round(adapter_v, 6),
            "base_avg_velocity": round(base_v, 6),
            "adapter_efficiency": round(efficiency, 3),
            "recommendation": rec,
        }

    def report(self) -> None:
        """Generate PEFT diagnostics report."""
        lines = ["─" * 55, "🔌 PEFT / LoRA Diagnostics", "─" * 55]

        lines.append(f"  Adapter layers: {len(self.adapter_layers)}")
        lines.append(f"  Base layers:    {len(self.base_layers)}")
        lines.append("")

        # Utilization
        util = self.adapter_utilization()
        if util:
            lines.append("  📊 Adapter Utilization:")
            for name, info in util.items():
                emoji = "🟢" if info["status"] == "ACTIVE" else "🟡" if info["status"] == "UNDERUTILIZED" else "🔴"
                lines.append(f"   {emoji} {name:<35s} util={info['utilization']:.2f} ({info['status']})")
            lines.append("")

        # Analysis
        analysis = self.adapter_vs_base_analysis()
        if analysis:
            lines.append("  ⚡ Adapter vs Base:")
            lines.append(f"     Adapter velocity: {analysis.get('adapter_avg_velocity', 0):.6f}")
            lines.append(f"     Base velocity:    {analysis.get('base_avg_velocity', 0):.6f}")
            lines.append(f"     Efficiency:       {analysis.get('adapter_efficiency', 0):.2f}x")
            lines.append(f"     💊 {analysis.get('recommendation', '')}")
            lines.append("")

        # Rank recommendations
        ranks = self.recommend_ranks()
        lines.append("  🎯 Recommended LoRA Ranks:")
        for name, rank in sorted(ranks.items(), key=lambda x: -x[1]):
            emoji = "🔴" if rank >= 64 else "🟡" if rank >= 8 else "🟢"
            lines.append(f"   {emoji} {name:<35s} rank={rank}")

        report = "\n".join(lines)
        print(report)

    def to_agent_xml(self) -> str:
        """Export PEFT analysis as XML for AI agents."""
        from xml.sax.saxutils import escape
        ranks = self.recommend_ranks()
        util = self.adapter_utilization()
        analysis = self.adapter_vs_base_analysis()

        layers_xml = []
        for name, rank in ranks.items():
            u = util.get(name, {})
            layers_xml.append(
                f'    <layer name="{escape(name)}" '
                f'recommended_rank="{rank}" '
                f'utilization="{u.get("utilization", "N/A")}" '
                f'status="{u.get("status", "N/A")}" />'
            )

        return (
            "<peft_analysis>\n"
            f"  <adapter_efficiency>{analysis.get('adapter_efficiency', 'N/A')}</adapter_efficiency>\n"
            f"  <recommendation>{escape(analysis.get('recommendation', ''))}</recommendation>\n"
            + "\n".join(layers_xml) + "\n"
            + "</peft_analysis>"
        )
