"""
Weights & Biases (W&B) Integration for GradTracer.

Provides seamless logging of GradTracer diagnostics (Zombie percent, popularity bias,
dead embeddings, layer health) directly into W&B dashboards.
"""
from typing import Optional, Any
import warnings

def log_to_wandb(tracker: Any, step: Optional[int] = None, prefix: str = "gradtracer"):
    """
    Extracts metrics from a GradTracer tracker and logs them to Weights & Biases.
    
    Compatible with:
    - EmbeddingTracker
    - FlowTracker
    - FlowManager
    
    Args:
        tracker: The GradTracer instance.
        step: Optional training step to log against.
        prefix: Prefix for the metric names in W&B (e.g., "gradtracer").
    """
    try:
        import wandb
    except ImportError:
        warnings.warn("wandb is not installed. Please install it to use `log_to_wandb`.")
        return

    if wandb.run is None:
        warnings.warn("wandb.init() has not been called. W&B logging skipped.")
        return
        
    metrics = {}
    
    # 1. Check if it's an EmbeddingTracker
    if hasattr(tracker, "zombie_embeddings") and hasattr(tracker, "popularity_bias"):
        s = tracker.summary()
        metrics.update({
            f"{prefix}/{tracker.name}/zombie_pct": s["zombie_pct"],
            f"{prefix}/{tracker.name}/dead_pct": s["dead_pct"],
            f"{prefix}/{tracker.name}/coverage_pct": s["coverage_pct"],
            f"{prefix}/{tracker.name}/popularity_gini": s["gini"],
        })
        
        # If AutoFix is active, log the cumulative interventions
        if getattr(tracker, "auto_fix", False) and getattr(tracker, "audit_logger", None):
            audit = tracker.audit_logger.summary()
            metrics[f"{prefix}/{tracker.name}/autofix_total_interventions"] = audit.get("total_events", 0)

    # 2. Check if it's a FlowTracker (Deep Learning health)
    elif hasattr(tracker, "store") and hasattr(tracker, "export_for_agent"):
        from gradtracer.analyzers.health import layer_health_score
        health_scores = layer_health_score(tracker.store)
        
        # Log average health
        avg_health = sum(health_scores.values()) / max(len(health_scores), 1)
        metrics[f"{prefix}/global_avg_health"] = avg_health
        
        # Log individual layer health generically (could be large, so we pick worst 3)
        worst_layers = sorted(health_scores.items(), key=lambda x: x[1])[:3]
        for i, (name, score) in enumerate(worst_layers):
            metrics[f"{prefix}/worst_layer_{i}_health"] = score

    # 3. Check if it's a FlowManager (Multi-tracker ecosystem)
    elif hasattr(tracker, "trackers"):
        for name, sub_tracker in tracker.trackers.items():
            # Recursively log sub-trackers using their specific names
            sub_metrics = _extract_metrics(sub_tracker, f"{prefix}/{name}")
            metrics.update(sub_metrics)

    # Dispatch to wandb
    if metrics:
        wandb.log(metrics, step=step)


def _extract_metrics(tracker: Any, prefix: str) -> dict:
    """Helper to extract metrics without importing wandb again."""
    metrics = {}
    
    if hasattr(tracker, "summary"):
        s = tracker.summary()
        metrics.update({
            f"{prefix}/zombie_pct": s.get("zombie_pct", 0),
            f"{prefix}/dead_pct": s.get("dead_pct", 0),
            f"{prefix}/coverage_pct": s.get("coverage_pct", 0),
            f"{prefix}/popularity_gini": s.get("gini", 0),
        })
        
        if getattr(tracker, "auto_fix", False) and getattr(tracker, "audit_logger", None):
            audit = tracker.audit_logger.summary()
            metrics[f"{prefix}/autofix_total_interventions"] = audit.get("total_events", 0)
            
    return metrics
