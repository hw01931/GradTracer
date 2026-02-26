"""
DistillationTracker — Knowledge distillation diagnostics using training dynamics.

Compares Teacher and Student FlowTracker data to identify where the student
is struggling to learn from the teacher.

Usage:
    teacher_tracker = FlowTracker(teacher_model)
    student_tracker = FlowTracker(student_model)
    # ... train both (or just student with KD) ...
    dt = DistillationTracker(teacher_tracker, student_tracker)
    gaps = dt.flow_gap()
    weights = dt.suggest_distillation_weights()
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from flowgrad.snapshot import SnapshotStore


class DistillationTracker:
    """
    Track and diagnose knowledge distillation by comparing teacher/student dynamics.

    Key insight: if the student's velocity or gradient SNR is much lower than
    the teacher's in a particular layer, the student is struggling to learn
    that layer's behavior —> increase KD loss weight for that layer.
    """

    def __init__(self, teacher_tracker, student_tracker,
                 layer_mapping: Optional[Dict[str, str]] = None):
        """
        Args:
            teacher_tracker: FlowTracker attached to the teacher model.
            student_tracker: FlowTracker attached to the student model.
            layer_mapping: Optional {teacher_layer: student_layer} mapping.
                           If None, auto-maps by position order.
        """
        self.teacher = teacher_tracker
        self.student = student_tracker
        self.t_store: SnapshotStore = teacher_tracker.store
        self.s_store: SnapshotStore = student_tracker.store

        if layer_mapping is not None:
            self.mapping = layer_mapping
        else:
            self.mapping = self._auto_map()

    def _auto_map(self) -> Dict[str, str]:
        """Auto-map teacher layers to student layers by position."""
        t_names = self.t_store.layer_names
        s_names = self.s_store.layer_names
        mapping = {}
        for i, t_name in enumerate(t_names):
            if i < len(s_names):
                mapping[t_name] = s_names[i]
        return mapping

    def flow_gap(self) -> Dict[str, Dict]:
        """
        Compare weight velocity between teacher and student.

        Logic:
            gap_ratio = |vel_student - vel_teacher| / max(vel_teacher, eps)

        Large gap = student failing to replicate teacher's learning dynamics.

        Returns:
            {student_layer: {"teacher_layer": str, "teacher_vel": float,
                             "student_vel": float, "gap_ratio": float,
                             "status": "OK" | "STRUGGLING" | "CRITICAL"}}
        """
        window = min(10, self.s_store.num_steps, self.t_store.num_steps)
        if window == 0:
            return {}

        results = {}
        for t_name, s_name in self.mapping.items():
            t_series = self.t_store.get_layer_series(t_name, "velocity")
            s_series = self.s_store.get_layer_series(s_name, "velocity")

            t_vel = float(np.mean(t_series[-window:])) if t_series else 0.0
            s_vel = float(np.mean(s_series[-window:])) if s_series else 0.0

            denominator = max(abs(t_vel), 1e-10)
            gap = abs(s_vel - t_vel) / denominator

            if gap > 5.0:
                status = "CRITICAL"
            elif gap > 2.0:
                status = "STRUGGLING"
            else:
                status = "OK"

            results[s_name] = {
                "teacher_layer": t_name,
                "teacher_vel": round(t_vel, 6),
                "student_vel": round(s_vel, 6),
                "gap_ratio": round(gap, 3),
                "status": status,
            }

        return results

    def snr_comparison(self) -> Dict[str, Dict]:
        """
        Compare gradient SNR between teacher and student.

        If student's SNR is much lower → noise is dominating that layer.

        Returns:
            {student_layer: {"teacher_snr": float, "student_snr": float,
                             "status": "OK" | "NOISY"}}
        """
        from flowgrad.analyzers.health import gradient_snr_per_layer

        t_snr = gradient_snr_per_layer(self.t_store)
        s_snr = gradient_snr_per_layer(self.s_store)

        results = {}
        for t_name, s_name in self.mapping.items():
            t_series = t_snr.get(t_name, [])
            s_series = s_snr.get(s_name, [])

            t_val = t_series[-1] if t_series else 0.0
            s_val = s_series[-1] if s_series else 0.0

            if math.isinf(t_val):
                t_val = 100.0
            if math.isinf(s_val):
                s_val = 100.0

            status = "NOISY" if (s_val < t_val * 0.1 and t_val > 0.001) else "OK"

            results[s_name] = {
                "teacher_layer": t_name,
                "teacher_snr": round(float(t_val), 6),
                "student_snr": round(float(s_val), 6),
                "status": status,
            }

        return results

    def suggest_distillation_weights(self) -> Dict[str, float]:
        """
        Suggest per-layer KD loss weights based on flow gap.

        Layers where student struggles → higher KD weight to focus learning.

        Logic:
            raw_weight = gap_ratio for each layer
            normalized via softmax → weights sum to ~1

        Returns:
            {student_layer: weight}
        """
        gaps = self.flow_gap()
        if not gaps:
            return {}

        names = list(gaps.keys())
        raw = np.array([gaps[n]["gap_ratio"] for n in names])

        # Softmax normalization (temperature=1)
        exp_raw = np.exp(raw - np.max(raw))
        weights = exp_raw / exp_raw.sum()

        return {name: round(float(w), 4) for name, w in zip(names, weights)}

    def report(self) -> None:
        """Generate a human-readable distillation diagnostics report."""
        gaps = self.flow_gap()
        snr = self.snr_comparison()
        weights = self.suggest_distillation_weights()

        lines = ["─" * 60, "🎓 Knowledge Distillation Diagnostics", "─" * 60]
        lines.append(f"  Teacher layers: {len(self.t_store.layer_names)}")
        lines.append(f"  Student layers: {len(self.s_store.layer_names)}")
        lines.append(f"  Mapped pairs:   {len(self.mapping)}")
        lines.append("")

        lines.append(f"  {'Student Layer':<30s} {'Gap':>6s} {'Status':<12s} {'KD Weight':>9s}")
        lines.append("  " + "─" * 60)

        for s_name in self.mapping.values():
            if s_name not in gaps:
                continue
            g = gaps[s_name]
            w = weights.get(s_name, 0)
            emoji = "🔴" if g["status"] == "CRITICAL" else "🟡" if g["status"] == "STRUGGLING" else "🟢"
            lines.append(
                f"  {emoji} {s_name:<28s} {g['gap_ratio']:>6.2f} "
                f"{g['status']:<12s} {w:>9.4f}"
            )

        # Critical layers alert
        critical = [s for s, g in gaps.items() if g["status"] in ("CRITICAL", "STRUGGLING")]
        if critical:
            lines.append("")
            lines.append("  💊 Prescription:")
            for s_name in critical:
                lines.append(f"     Increase KD loss weight for '{s_name}' "
                             f"(weight={weights.get(s_name, 0):.4f})")
            lines.append(f"     Consider intermediate layer matching (hint loss) "
                         f"for these layers.")

        report = "\n".join(lines)
        print(report)

    def to_agent_xml(self) -> str:
        """Export distillation analysis as XML for AI agents."""
        from xml.sax.saxutils import escape
        gaps = self.flow_gap()
        weights = self.suggest_distillation_weights()

        pairs_xml = []
        for s_name, g in gaps.items():
            pairs_xml.append(
                f'    <layer_pair student="{escape(s_name)}" '
                f'teacher="{escape(g["teacher_layer"])}" '
                f'gap_ratio="{g["gap_ratio"]}" '
                f'status="{g["status"]}" '
                f'kd_weight="{weights.get(s_name, 0):.4f}" />'
            )

        return (
            "<distillation_analysis>\n"
            + "\n".join(pairs_xml) + "\n"
            + "</distillation_analysis>"
        )
