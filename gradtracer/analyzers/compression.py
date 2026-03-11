"""
CompressionTracker — Model compression diagnostics for pruning, quantization, and LoRA.

Goal-based auto-search: set a performance floor, GradTracer finds the optimal compression.

Usage:
    tracker = CompressionTracker(model, eval_fn=lambda m: accuracy(m, X_val, y_val))

    # Manual snapshots
    tracker.snapshot("original")
    apply_pruning(model, 0.3)
    tracker.snapshot("pruned_30%", sparsity=0.3)

    # Auto search: "keep 95% performance, compress as much as possible"
    result = tracker.auto_compress(method="pruning", performance_floor=0.95)

    # Layer sensitivity profiling
    sensitivity = tracker.layer_sensitivity(method="pruning")

    tracker.report()
    tracker.plot.tradeoff_curve()
"""
from __future__ import annotations

import copy
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from gradtracer.analyzers.features import LayerProfile
# Import modular execution strategies
from gradtracer.analyzers.pruning import (
    apply_global_pruning,
    apply_heterogeneous_pruning,
    PruningAdvisor
)
from gradtracer.analyzers.quantization import (
    apply_uniform_quantization,
    apply_mixed_precision_quantization,
    QuantizationAdvisor
)

_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for CompressionTracker. "
                "Install with: pip install gradtracer[torch]"
            )
    return _torch


# ──────────────────────────────────────────────────────────────────────
#  Data Structures
# ──────────────────────────────────────────────────────────────────────

@dataclass
class LayerCompressionStats:
    """Per-layer compression statistics."""
    name: str
    total_params: int = 0
    nonzero_params: int = 0
    sparsity: float = 0.0
    weight_norm: float = 0.0
    sensitivity: Optional[float] = None  # perf drop when this layer is pruned


@dataclass
class CompressionSnapshot:
    """A snapshot of model state at one compression level."""
    name: str
    timestamp: float = 0.0

    # Model size
    total_params: int = 0
    nonzero_params: int = 0
    model_size_mb: float = 0.0
    sparsity: float = 0.0
    bits: int = 32

    # Performance
    eval_metrics: Dict[str, float] = field(default_factory=dict)

    # Layer detail
    layer_stats: Dict[str, LayerCompressionStats] = field(default_factory=dict)

    # Config
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionResult:
    """Result of auto_compress search."""
    optimal_config: Dict[str, Any]
    performance_original: float
    performance_compressed: float
    performance_retained: float  # ratio
    size_original_mb: float
    size_compressed_mb: float
    size_reduction: float  # ratio
    all_snapshots: List[CompressionSnapshot]
    recommendation: str


# ──────────────────────────────────────────────────────────────────────
#  CompressionTracker
# ──────────────────────────────────────────────────────────────────────

class CompressionTracker:
    """
    Track and optimize model compression (pruning, quantization, LoRA).

    Args:
        model: PyTorch nn.Module to compress.
        eval_fn: Callable that takes a model and returns a scalar score
                 (higher = better). Example: lambda m: accuracy(m, X_val, y_val)
    """

    def __init__(
        self,
        model,
        eval_fn: Optional[Callable] = None,
        tracker: Optional[Any] = None,
    ):
        """
        Initialize the compression tracker.

        Args:
            model: PyTorch model.
            eval_fn: A callable `fn(model) -> float` yielding a performance score.
                     Higher must be better (e.g. accuracy).
            tracker: Optional `FlowTracker` instance. Required if using 'heterogeneous' method.
        """
        self.model = model
        self.eval_fn = eval_fn
        self.tracker = tracker
        self.snapshots: List[CompressionSnapshot] = []
        self._sensitivity_cache: Optional[Dict[str, List[Tuple[float, float]]]] = None

    # ------------------------------------------------------------------
    # Measurement utilities
    # ------------------------------------------------------------------
    def _measure_model(self, model) -> Tuple[int, int, float]:
        """Measure total params, nonzero params, size in MB."""
        torch = _get_torch()
        total = 0
        nonzero = 0
        size_bytes = 0
        for p in model.parameters():
            numel = p.numel()
            total += numel
            
            # Fast check for qint8
            if p.dtype == torch.qint8 or p.dtype == torch.quint8:
                nonzero += numel
                size_bytes += numel * 1
            else:
                nonzero += (p != 0).sum().item()
                size_bytes += numel * p.element_size()
                
        return total, nonzero, size_bytes / (1024 * 1024)

    def _measure_layer_stats(self, model) -> Dict[str, LayerCompressionStats]:
        """Per-layer compression stats."""
        torch = _get_torch()
        import torch.nn as nn
        stats = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                if hasattr(module, "weight"):
                    w = module.weight.detach()
                    total = w.numel()
                    
                    if w.dtype == torch.qint8 or w.dtype == torch.quint8:
                        stats[name] = LayerCompressionStats(
                            name=name,
                            total_params=total,
                            nonzero_params=total,
                            sparsity=0.0,
                            weight_norm=0.0,
                        )
                    else:
                        nz = (w != 0).sum().item()
                        stats[name] = LayerCompressionStats(
                            name=name,
                            total_params=total,
                            nonzero_params=nz,
                            sparsity=1.0 - (nz / max(total, 1)),
                            weight_norm=w.norm().item(),
                        )
        return stats

    def _evaluate(self, model) -> float:
        """Evaluate model using eval_fn."""
        if self.eval_fn is None:
            raise ValueError(
                "eval_fn is required. Pass it to CompressionTracker() or snapshot()."
            )
        return float(self.eval_fn(model))

    # ------------------------------------------------------------------    
    # Pruning & Quantization Utilities have been moved to dedicated 
    # modules: `gradtracer.analyzers.pruning` and `gradtracer.analyzers.quantization`
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------
    def snapshot(
        self,
        name: str,
        eval_metrics: Optional[Dict[str, float]] = None,
        **config,
    ) -> CompressionSnapshot:
        """
        Record current model state as a snapshot.

        Args:
            name: Descriptive name (e.g. "original", "pruned_30%").
            eval_metrics: Optional pre-computed metrics. If None, uses eval_fn.
            **config: Extra config to store (sparsity=0.3, bits=8, lora_rank=16, etc.)
        """
        total, nonzero, size_mb = self._measure_model(self.model)
        layer_stats = self._measure_layer_stats(self.model)

        if eval_metrics is None and self.eval_fn is not None:
            score = self._evaluate(self.model)
            eval_metrics = {"score": score}

        snap = CompressionSnapshot(
            name=name,
            timestamp=time.time(),
            total_params=total,
            nonzero_params=nonzero,
            model_size_mb=size_mb,
            sparsity=1.0 - (nonzero / max(total, 1)),
            bits=config.get("bits", 32),
            eval_metrics=eval_metrics or {},
            layer_stats=layer_stats,
            config=config,
        )
        self.snapshots.append(snap)
        return snap

    # ------------------------------------------------------------------
    # Auto Compress — Goal-based search
    # ------------------------------------------------------------------
    def auto_compress(
        self,
        method: str = "pruning",
        performance_floor: float = 0.95,
        search_range: Tuple[float, float] = (0.1, 0.9),
        search_strategy: str = "binary",
        precision: float = 0.02,
        fine_tune_fn: Optional[Callable] = None,
    ) -> CompressionResult:
        """
        Automatically find the optimal compression level.

        Args:
            method: 'pruning', 'quantization', 'heterogeneous', or 'lora'.
            performance_floor: Minimum acceptable ratio of original performance (0-1).
            search_range: (min, max) range to search (ignored for quantization).
            search_strategy: 'binary' or 'grid' (ignored for quantization).
            precision: Search stops when range < precision.
            fine_tune_fn: Optional callable(model) to fine-tune after compression.

        Returns:
            CompressionResult with optimal config and all snapshots.
        """
        torch = _get_torch()

        # Baseline
        model_backup = copy.deepcopy(self.model)
        baseline_score = self._evaluate(self.model)
        baseline_total, _, baseline_size = self._measure_model(self.model)
        target_score = baseline_score * performance_floor

        # Record original
        self.model = copy.deepcopy(model_backup)
        orig_snap = self.snapshot("original")

        search_snapshots = [orig_snap]

        if method == "quantization":
            # Quantization is discrete, no search range needed for dynamic INT8
            model_copy = copy.deepcopy(model_backup)
            quantized_model = apply_uniform_quantization(model_copy)
            if fine_tune_fn is not None:
                fine_tune_fn(quantized_model)
            self.model = quantized_model
            snap = self.snapshot("quant_int8", bits=8, method="quantization")
            search_snapshots.append(snap)
        elif method == "heterogeneous":
            if self.tracker is None:
                raise ValueError("FlowTracker must be provided to CompressionTracker to use 'heterogeneous' method.")
            
            # Step 1: Find best target sparsity using heterogeneous pruning
            self._binary_search(model_backup, "heterogeneous_prune", target_score, baseline_score, search_range, precision, fine_tune_fn, search_snapshots)
            
            # Step 2: Grab the optimal pruned model from search
            optimal_prune = max(
                [s for s in search_snapshots if s.eval_metrics.get("score", 0) >= target_score],
                key=lambda s: 1 - s.model_size_mb,
                default=search_snapshots[0]
            )
            
            target_sp = optimal_prune.config.get("sparsity", 0.0)
            
            # Step 3: Rebuild hybrid from scratch
            model_copy = copy.deepcopy(model_backup)
            p_advisor = PruningAdvisor(self.tracker)
            p_plan = p_advisor.generate_pruning_plan(target_sp)
            pruned_model = apply_heterogeneous_pruning(model_copy, p_plan)
            
            q_advisor = QuantizationAdvisor(self.tracker)
            q_plan = q_advisor.recommend_mixed_precision()
            hybrid_model = apply_mixed_precision_quantization(pruned_model, q_plan)
            
            if fine_tune_fn is not None:
                fine_tune_fn(hybrid_model)
                
            self.model = hybrid_model
            snap = self.snapshot(f"heterogeneous_{target_sp:.2f}", method="heterogeneous", sparsity=round(target_sp, 3))
            search_snapshots.append(snap)
            
        elif search_strategy == "binary":
            result = self._binary_search(
                model_backup, method, target_score, baseline_score,
                search_range, precision, fine_tune_fn, search_snapshots
            )
        else:
            result = self._grid_search(
                model_backup, method, target_score, baseline_score,
                search_range, fine_tune_fn, search_snapshots
            )

        # Restore original model
        self.model = model_backup

        optimal_snap = max(
            [s for s in search_snapshots if s.eval_metrics.get("score", 0) >= target_score],
            key=lambda s: 1 - s.model_size_mb,  # Maximize size reduction
            default=search_snapshots[0],
        )

        comp_size = optimal_snap.model_size_mb
        perf = optimal_snap.eval_metrics.get("score", baseline_score)

        recommendation = self._generate_recommendation(
            method, optimal_snap, baseline_score, baseline_size, target_score
        )

        return CompressionResult(
            optimal_config=optimal_snap.config,
            performance_original=baseline_score,
            performance_compressed=perf,
            performance_retained=perf / baseline_score if baseline_score > 0 else 1.0,
            size_original_mb=baseline_size,
            size_compressed_mb=comp_size,
            size_reduction=1.0 - (comp_size / baseline_size) if baseline_size > 0 else 0,
            all_snapshots=search_snapshots,
            recommendation=recommendation,
        )

    def _binary_search(
        self, model_backup, method, target_score, baseline_score,
        search_range, precision, fine_tune_fn, snapshots
    ) -> None:
        torch = _get_torch()
        lo, hi = search_range

        while hi - lo > precision:
            mid = (lo + hi) / 2
            model_copy = copy.deepcopy(model_backup)

            if method == "pruning":
                apply_global_pruning(model_copy, mid)
                config = {"sparsity": round(mid, 3), "method": "pruning"}
            elif method == "heterogeneous_prune":
                p_advisor = PruningAdvisor(self.tracker)
                p_plan = p_advisor.generate_pruning_plan(mid)
                apply_heterogeneous_pruning(model_copy, p_plan)
                config = {"sparsity": round(mid, 3), "method": "heterogeneous_prune"}
            elif method == "lora":
                config = {"rank": int(mid), "method": "lora"}
            else:
                raise ValueError(f"Unknown method: {method}")

            if fine_tune_fn is not None:
                fine_tune_fn(model_copy)

            self.model = model_copy
            snap = self.snapshot(
                f"{method}_{mid:.2f}",
                **config,
            )
            snapshots.append(snap)
            score = snap.eval_metrics.get("score", 0)

            if score >= target_score:
                lo = mid  # can compress more
            else:
                hi = mid  # too much compression

        self.model = model_backup

    def _grid_search(
        self, model_backup, method, target_score, baseline_score,
        search_range, fine_tune_fn, snapshots
    ) -> None:
        torch = _get_torch()
        lo, hi = search_range
        steps = np.linspace(lo, hi, 9)

        for level in steps:
            model_copy = copy.deepcopy(model_backup)

            if method == "pruning":
                apply_global_pruning(model_copy, level)
                config = {"sparsity": round(float(level), 3), "method": "pruning"}
            elif method == "heterogeneous_prune":
                p_advisor = PruningAdvisor(self.tracker)
                p_plan = p_advisor.generate_pruning_plan(level)
                apply_heterogeneous_pruning(model_copy, p_plan)
                config = {"sparsity": round(float(level), 3), "method": "heterogeneous_prune"}
            else:
                config = {"rank": int(level), "method": method}

            if fine_tune_fn is not None:
                fine_tune_fn(model_copy)

            self.model = model_copy
            snap = self.snapshot(
                f"{method}_{level:.2f}",
                **config,
            )
            snapshots.append(snap)

        self.model = model_backup

    # ------------------------------------------------------------------
    # Layer Sensitivity
    # ------------------------------------------------------------------
    def layer_sensitivity(
        self,
        method: str = "pruning",
        sparsity_levels: Optional[List[float]] = None,
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Profile each layer's sensitivity to compression.

        For each layer, applies compression at various levels and measures
        performance impact.

        Args:
            method: 'pruning' (default).
            sparsity_levels: List of sparsity levels to test per layer.

        Returns:
            {layer_name: [(sparsity, performance_score), ...]}
        """
        torch = _get_torch()

        if sparsity_levels is None:
            sparsity_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

        model_backup = copy.deepcopy(self.model)
        baseline_score = self._evaluate(model_backup)

        # Get prunable layers
        import torch.nn as nn
        layer_names = []
        for name, module in model_backup.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                if hasattr(module, "weight"):
                    layer_names.append(name)

        results: Dict[str, List[Tuple[float, float]]] = {}

        for layer_name in layer_names:
            layer_results = []
            for sparsity in sparsity_levels:
                model_copy = copy.deepcopy(model_backup)
                self._apply_layer_pruning(model_copy, layer_name, sparsity)
                score = self.eval_fn(model_copy)
                drop = baseline_score - score
                layer_results.append((sparsity, float(score)))
                del model_copy

            results[layer_name] = layer_results

        self._sensitivity_cache = results
        self.model = model_backup
        return results

    def recommend_nonuniform(
        self,
        performance_floor: float = 0.95,
        sparsity_levels: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Recommend non-uniform per-layer pruning based on sensitivity.

        Assigns higher sparsity to less sensitive layers.

        Returns:
            {layer_name: recommended_sparsity}
        """
        if self._sensitivity_cache is None:
            self.layer_sensitivity(sparsity_levels=sparsity_levels)

        sensitivity = self._sensitivity_cache
        baseline_score = self._evaluate(self.model)
        target = baseline_score * performance_floor

        recommendations = {}
        for layer_name, results in sensitivity.items():
            # Find max sparsity where score >= target
            best_sparsity = 0.0
            for sparsity, score in sorted(results, key=lambda x: x[0]):
                if score >= target:
                    best_sparsity = sparsity
                else:
                    break
            recommendations[layer_name] = best_sparsity

        return recommendations

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def _generate_recommendation(
        self, method, optimal_snap, baseline_score, baseline_size, target_score
    ) -> str:
        perf = optimal_snap.eval_metrics.get("score", 0)
        retained = perf / baseline_score * 100 if baseline_score > 0 else 100
        reduction = (1 - optimal_snap.model_size_mb / baseline_size) * 100 if baseline_size > 0 else 0

        if method == "pruning":
            sp = optimal_snap.config.get("sparsity", 0)
            return (
                f"Apply L1 unstructured pruning at {sp*100:.0f}% sparsity. "
                f"Performance: {retained:.1f}% retained, size: {reduction:.1f}% reduction. "
                f"Fine-tune for 5-10 epochs to recover ~1-2% accuracy."
            )
        else:
            return f"Optimal config: {optimal_snap.config}. Performance: {retained:.1f}% retained."

    def report(self) -> None:
        """Generate comprehensive compression diagnostic report."""
        lines = []
        lines.append("=" * 60)
        lines.append("  GradTracer — Compression Diagnostic Report")
        lines.append("=" * 60)
        lines.append("")

        lines.append(f"📊 Snapshots recorded: {len(self.snapshots)}")
        lines.append("")

        if not self.snapshots:
            lines.append("  No snapshots recorded yet.")
            lines.append("=" * 60)
            report = "\n".join(lines)
            print(report)
            return

        # Snapshot comparison table
        lines.append("─" * 60)
        lines.append("📋 Snapshot Comparison")
        lines.append("─" * 60)

        header = f"  {'Version':<20s} {'Params':>10s} {'Sparsity':>10s} {'Size(MB)':>10s} {'Score':>10s} {'vs Orig':>10s}"
        lines.append(header)
        lines.append("  " + "─" * 70)

        orig_score = self.snapshots[0].eval_metrics.get("score", 1.0) if self.snapshots else 1.0

        for snap in self.snapshots:
            score = snap.eval_metrics.get("score", 0)
            retained = score / orig_score * 100 if orig_score > 0 else 100
            emoji = "✅" if retained >= 95 else "🟡" if retained >= 90 else "❌"

            params_str = f"{snap.nonzero_params / 1000:.1f}K" if snap.nonzero_params < 1e6 else f"{snap.nonzero_params / 1e6:.2f}M"
            lines.append(
                f"  {snap.name:<20s} {params_str:>10s} {snap.sparsity*100:>9.1f}% "
                f"{snap.model_size_mb:>9.2f} {score:>10.4f} {retained:>8.1f}% {emoji}"
            )
        lines.append("")

        # Best point
        valid = [s for s in self.snapshots if s.eval_metrics.get("score", 0) >= orig_score * 0.95]
        if valid and len(self.snapshots) > 1:
            best = max(valid, key=lambda s: s.sparsity)
            retained = best.eval_metrics.get("score", 0) / orig_score * 100
            reduction = (1 - best.model_size_mb / self.snapshots[0].model_size_mb) * 100

            lines.append("─" * 60)
            lines.append("🎯 Optimal Point (≥95% performance retained)")
            lines.append("─" * 60)
            lines.append(f"  ✅ Config: {best.name}")
            lines.append(f"  📦 Size: {self.snapshots[0].model_size_mb:.2f}MB → {best.model_size_mb:.2f}MB ({reduction:.1f}% reduction)")
            lines.append(f"  📊 Performance: {retained:.1f}% retained")
            lines.append("")

        # Layer sensitivity
        if self._sensitivity_cache:
            lines.append("─" * 60)
            lines.append("🔬 Layer Sensitivity (most → least sensitive)")
            lines.append("─" * 60)

            baseline = self._evaluate(self.model)
            layer_max_drops = {}
            for layer_name, results in self._sensitivity_cache.items():
                # Drop at 50% sparsity
                drops = [(sp, baseline - sc) for sp, sc in results]
                drop_at_50 = next((d for s, d in drops if abs(s - 0.5) < 0.1), 0)
                layer_max_drops[layer_name] = drop_at_50

            sorted_layers = sorted(layer_max_drops.items(), key=lambda x: -x[1])
            for name, drop in sorted_layers:
                pct = drop / baseline * 100 if baseline > 0 else 0
                emoji = "🔴" if pct > 5 else "🟡" if pct > 2 else "🟢"
                label = "SENSITIVE!" if pct > 5 else "moderate" if pct > 2 else "safe to prune"
                lines.append(f"  {emoji} {name:<30s} 50% pruning → {pct:>5.1f}% drop ({label})")

            # Non-uniform recommendation
            rec = self.recommend_nonuniform()
            lines.append("")
            lines.append("  💊 Non-uniform pruning recommendation:")
            total_pruned = 0
            total_params = 0
            for layer_name, sp in rec.items():
                lines.append(f"     {layer_name}: {sp*100:.0f}%")
            lines.append("")

        lines.append("=" * 60)
        report = "\n".join(lines)
        print(report)

    @property
    def plot(self):
        from gradtracer.viz.plots import CompressionPlotAPI
        return CompressionPlotAPI(self)

    @property
    def summary(self) -> Dict[str, Any]:
        result = {
            "num_snapshots": len(self.snapshots),
            "snapshots": [
                {"name": s.name, "sparsity": s.sparsity,
                 "size_mb": s.model_size_mb, "score": s.eval_metrics.get("score", None)}
                for s in self.snapshots
            ],
        }
        return result

    def __repr__(self):
        return f"CompressionTracker(snapshots={len(self.snapshots)})"
