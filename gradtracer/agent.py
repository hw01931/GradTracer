"""
GradTracer Agent Mode — Structured XML export for AI coding assistants.

Generates diagnostic output optimized for AI agents (Cursor, Copilot, Antigravity).
Employs Statistical Causal Reasoning by combining multiple metrics (LR, Loss Trend, SNR)
to generate high-confidence prescriptions via <causal_model> tags.

The XML includes:
  1. Experiment history (cross-run comparison)
  2. Environment context (optimizer, scheduler, architecture)
  3. Training state summary (loss trend, current LR)
  4. Causal Meta-Diagnostics (multi-variate reasoning, false-positive suppression)
"""
from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape

from gradtracer.history import HistoryTracker


# ======================================================================
#  Context Scanner — extracts optimizer, scheduler, architecture info
# ======================================================================

def _scan_optimizer(tracker) -> Dict[str, Any]:
    """Extract optimizer info as a dict for reasoning."""
    opt = getattr(tracker, "optimizer", None)
    if opt is None:
        return {"name": "Not provided", "lr": 0.0, "is_adaptive": False, "str": "Not provided"}
    
    name = type(opt).__name__
    pg = opt.param_groups[0]
    lr = pg.get("lr", 0.0)
    wd = pg.get("weight_decay", 0)
    betas = pg.get("betas", None)
    
    parts = [f"lr={lr}", f"weight_decay={wd}"]
    if betas:
        parts.append(f"betas={betas}")
        
    is_adaptive = name in ["Adam", "AdamW", "RMSprop", "Adagrad"]
    
    return {
        "name": name,
        "lr": lr,
        "is_adaptive": is_adaptive,
        "str": f"{name}({', '.join(parts)})"
    }


def _scan_scheduler(tracker) -> Dict[str, Any]:
    """Extract LR scheduler info."""
    sched = getattr(tracker, "scheduler", None)
    if sched is None:
        return {"name": "None", "current_lr": None, "str": "None"}
    
    name = type(sched).__name__
    try:
        current_lr = sched.get_last_lr()[0]
        return {"name": name, "current_lr": current_lr, "str": f"{name}(current_lr={current_lr})"}
    except Exception:
        return {"name": name, "current_lr": None, "str": name}


def _scan_architecture(model) -> Dict[str, Any]:
    """Scan model for structural summary."""
    total_params = 0
    trainable_params = 0
    norm_layers = 0
    dropout_layers = 0
    activation_types = set()

    for m in model.modules():
        cls_name = type(m).__name__.lower()
        if "norm" in cls_name:
            norm_layers += 1
        if "dropout" in cls_name:
            dropout_layers += 1
        if any(act in cls_name for act in ["relu", "gelu", "silu", "tanh", "sigmoid", "leaky"]):
            activation_types.add(type(m).__name__)

    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "normalization_layers": norm_layers,
        "dropout_layers": dropout_layers,
        "activations": sorted(activation_types) if activation_types else ["None detected"],
    }


import json

# ======================================================================
#  Agent Exporter with Causal Meta-Diagnostics (JSON Standard)
# ======================================================================

class AgentExporter:
    """
    Exports GradTracer diagnostics into structured XML for AI agents.
    Uses Causal Reasoning to combine multiple metrics and suppress false positives.
    """

    @classmethod
    def export_dl(
        cls,
        tracker,
        run_name: str = "current_run",
        include_history: bool = True,
        save: bool = True,
    ) -> str:
        from gradtracer.analyzers.velocity import detect_stagnation, detect_explosion
        from gradtracer.analyzers.health import layer_health_score, gradient_snr_per_layer

        store = tracker.store

        # ── 1. Gather Cross-Signal Context ──────────────────────────
        opt_info = _scan_optimizer(tracker)
        sched_info = _scan_scheduler(tracker)
        arch = _scan_architecture(tracker.model)
        
        losses = store.get_loss_history()
        valid_losses = [l for l in losses if l is not None]
        
        loss_trend = 0.0
        loss_is_improving = False
        if len(valid_losses) > 1:
            loss_trend = valid_losses[-1] - valid_losses[0]
            recent_trend = valid_losses[-1] - valid_losses[-min(5, len(valid_losses))]
            loss_is_improving = recent_trend < -0.01

        current_lr = sched_info.get("current_lr") or opt_info.get("lr", 0.0)
        is_adaptive = opt_info.get("is_adaptive", False)

        # ── 2. Causal Engine: Evaluate Rules ────────────────────────
        findings_json = []
        issue_summaries = []

        # A. Inconsistency Detection (Zero LR but loss changing)
        if current_lr == 0.0 and loss_is_improving:
            findings_json.append({
                "type": "INCONSISTENCY_DETECTED",
                "premise": "Learning rate is 0.0 but loss is significantly decreasing.",
                "implies": "Major inconsistency: weights might be frozen while other params learn, or scheduler is detached.",
                "action": "CHECK_CODE_LOGIC",
                "expected_effect": "Resolve ghost updates and ensure reproducible training.",
                "confidence": 0.95
            })
            issue_summaries.append("LR_INCONSISTENCY")

        # B. Low SNR Analysis (False Positive Suppression)
        snr_data = gradient_snr_per_layer(store)
        low_snr_layers = {name: s[-1] for name, s in snr_data.items() if s and 0 < s[-1] < 0.01}
        
        if low_snr_layers:
            avg_snr = sum(low_snr_layers.values()) / len(low_snr_layers)
            if loss_is_improving and is_adaptive:
                # Suppression Logic: Benign Noise in Adaptive Optimizer
                findings_json.append({
                    "type": "BENIGN_NOISE",
                    "premise": f"Low gradient SNR (<0.01) detected in {len(low_snr_layers)} layers, BUT loss is actively decreasing and Adam/Adaptive optimizer is used.",
                    "implies": "Benign noise. Adaptive optimizers absorb raw variance. The model is in a stable fine-tuning phase despite high stochasticity.",
                    "action": "NO_INTERVENTION_REQUIRED",
                    "expected_effect": "Suppress false alarms. Continue training.",
                    "confidence": 0.85
                })
            else:
                # Legitimate Low SNR Alert
                findings_json.append({
                    "type": "HARMFUL_LOW_SNR",
                    "premise": f"Low SNR (<0.01) in {len(low_snr_layers)} layers and loss is NOT improving.",
                    "implies": "Optimizer is taking random walks. Gradient noise is dominating the descent signal.",
                    "action": "INCREASE_BATCH_SIZE_OR_REDUCE_LR",
                    "expected_effect": "Reduce gradient variance to restore directional descent.",
                    "risk": "May slow down epoch processing time if batch size is increased.",
                    "confidence": 0.75
                })
                issue_summaries.append("LOW_SNR")

        # C. Stagnation Analysis
        stagnant = detect_stagnation(store)
        if stagnant:
            if loss_is_improving:
                findings_json.append({
                    "type": "SAFE_STAGNATION",
                    "premise": f"{len(stagnant)} layers have near-zero velocity, but overall loss is decreasing.",
                    "implies": "These specific layers have converged early or are redundant, while other layers drive the loss.",
                    "action": "CONSIDER_PRUNING",
                    "expected_effect": "Remove dead weights to compress model without hurting current descent trajectory.",
                    "confidence": 0.80
                })
            else:
                findings_json.append({
                    "type": "HARMFUL_STAGNATION",
                    "premise": f"{len(stagnant)} layers stagnated and loss is plateaued.",
                    "implies": "Learning has halted prematurely due to vanishing gradients or excessive regularization.",
                    "action": "INCREASE_LR_OR_REMOVE_WEIGHT_DECAY",
                    "expected_effect": "Force parameters out of local plateau.",
                    "confidence": 0.90
                })
                issue_summaries.append("STAGNATION")

        # D. Dead Neurons
        dead_layers = []
        for name in store.layer_names:
            history = store.get_layer_history(name)
            if history and history[-1].dead_ratio > 0.5:
                dead_layers.append(name)
                
        if dead_layers:
            findings_json.append({
                "type": "DEAD_NEURON_COLLAPSE",
                "premise": f">50% parameters near zero in layers: {', '.join(dead_layers[:3])}...",
                "implies": "Activation collapse (e.g., dying ReLU) or extreme sparsity regime.",
                "action": "REPLACE_RELU_WITH_LEAKYRELU_OR_GELU",
                "expected_effect": "Allow gradients to flow through negative pre-activations to revive neurons.",
                "confidence": 0.88
            })
            issue_summaries.append("DEAD_NEURONS")

        # ── 3. Build Run Data for History ───────────────────────────
        health = layer_health_score(store)
        run_data = {
            "run_id": run_name,
            "optimizer": opt_info["str"],
            "total_params": f"{arch['total_params']/1e6:.2f}M",
            "steps": store.num_steps,
            "final_loss": round(valid_losses[-1], 4) if valid_losses else None,
            "issues": issue_summaries,
            "avg_health": round(sum(health.values()) / max(len(health), 1), 1),
        }

        if save:
            HistoryTracker.append_run(run_data)

        # ── 4. Assemble JSON ─────────────────────────────────────────
        report = {
            "gradtracer_agent_report": {
                "environment": {
                    "optimizer": opt_info["str"],
                    "lr_scheduler": sched_info["str"]
                },
                "model_architecture": {
                    "total_params": f"{arch['total_params']/1e6:.2f}M",
                    "trainable_params": f"{arch['trainable_params']/1e6:.2f}M",
                    "normalization_layers": arch["normalization_layers"],
                    "dropout_layers": arch["dropout_layers"],
                    "activations": arch["activations"]
                }
            }
        }

        # A. History
        if include_history:
            past = HistoryTracker.get_recent_runs(n=5)
            past = [r for r in past if r.get("run_id") != run_name]
            if past:
                report["gradtracer_agent_report"]["experiment_history"] = past[-3:]

        # B. Training State
        state = {"current_step": store.num_steps}
        if valid_losses:
            state["initial_loss"] = round(valid_losses[0], 4)
            state["current_loss"] = round(valid_losses[-1], 4)
            if len(valid_losses) > 1:
                state["loss_trend"] = round(loss_trend, 4)
                state["min_loss"] = round(min(valid_losses), 4)
        report["gradtracer_agent_report"]["training_state"] = state

        # C. Causal Diagnostics
        if findings_json:
            report["gradtracer_agent_report"]["diagnostics"] = findings_json
        else:
            report["gradtracer_agent_report"]["diagnostics"] = {
                "status": "HEALTHY — No critical issues or inconsistencies detected."
            }

        # D. Layer Health Summary
        health_lines = []
        for name in sorted(health, key=lambda k: health[k]):
            score = health[name]
            status = "CRITICAL" if score < 40 else "WARNING" if score < 70 else "HEALTHY"
            health_lines.append({
                "layer": name,
                "health": round(score, 0),
                "status": status
            })
        report["gradtracer_agent_report"]["layer_health_summary"] = health_lines

        return json.dumps(report, indent=2)

    @classmethod
    def export_embedding(cls, tracker, save: bool = False) -> str:
        """
        Export causal JSON diagnostics specifically for an EmbeddingTracker.
        """
        report = {
            "gradtracer_embedding_report": {
                "layer": tracker.name
            }
        }
        findings_json = []
        
        summary = tracker.summary()
        
        # Matrix Stats
        report["gradtracer_embedding_report"]["embedding_stats"] = {
            "num_embeddings": summary["num_embeddings"],
            "active_coverage_pct": round(summary["coverage_pct"], 1),
            "popularity_gini": round(summary["gini"], 3)
        }
        
        # 1. Zombie Embeddings
        if summary["zombie_pct"] > 5.0:
            findings_json.append({
                "type": "ZOMBIE_EMBEDDINGS",
                "premise": f"{summary['zombie_pct']:.1f}% of embeddings have high update velocity but strictly alternating/negative cosine similarity between steps.",
                "implies": "Optimizer is oscillating. Conflicting gradients from different users/items are pulling these embeddings back and forth without generalizing.",
                "action": "DECREASE_LR_BETA_OR_USE_SPARSE_ADAM",
                "expected_effect": "Smooth out the trajectory of rare/conflicted items and prevent representation collapse.",
                "confidence": 0.92
            })
            
        # 2. Dead Embeddings
        if summary["dead_pct"] > 50.0:
            findings_json.append({
                "type": "DEAD_EMBEDDINGS",
                "premise": f"{summary['dead_pct']:.1f}% of embeddings have never received a gradient update.",
                "implies": "Severe cold-start sparsity or broken dataloader negative sampling.",
                "action": "DOWNSAMPLE_NEGATIVES_OR_USE_HASHING_TRICK",
                "expected_effect": "Increase sample efficiency and reduce memory footprint of dead parameters.",
                "confidence": 0.88
            })
            
        # 3. Popularity Bias
        if summary["gini"] > 0.8:
            findings_json.append({
                "type": "POPULARITY_BIAS",
                "premise": f"Exposure distribution is highly skewed (Gini coefficient: {summary['gini']:.2f} > 0.8).",
                "implies": "Model is predominantly optimizing for top-popular items. Long-tail embeddings will suffer from inadequate learning.",
                "action": "APPLY_LOG_Q_CORRECTION_OR_INVERSE_FREQUENCY_SAMPLING",
                "expected_effect": "Debias the softmax logits and improve long-tail recommendation coverage.",
                "risk": "May slightly drop overall accuracy (HR@10) on heavily skewed test sets.",
                "confidence": 0.95
            })

        # Assemble Diagnostics
        if findings_json:
            report["gradtracer_embedding_report"]["diagnostics"] = findings_json
        else:
            report["gradtracer_embedding_report"]["diagnostics"] = {
                "status": "HEALTHY — Embedding matrix shows stable learning dynamics and adequate coverage."
            }

        return json.dumps(report, indent=2)

