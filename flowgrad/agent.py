"""
FlowGrad Agent Mode — Structured XML export for AI coding assistants.

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

from flowgrad.history import HistoryTracker


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


# ======================================================================
#  XML Builder Helpers
# ======================================================================

def _tag(name: str, content: str, indent: int = 2, **attrs) -> str:
    """Multi-line XML tag with optional attributes."""
    sp = " " * indent
    attr_str = "".join([f' {k}="{escape(str(v))}"' for k, v in attrs.items()])
    inner = textwrap.indent(content.strip(), sp + "  ")
    return f"{sp}<{name}{attr_str}>\n{inner}\n{sp}</{name}>"


def _line(name: str, value: Any, indent: int = 2, **attrs) -> str:
    """Single-line XML tag with optional attributes."""
    sp = " " * indent
    attr_str = "".join([f' {k}="{escape(str(v))}"' for k, v in attrs.items()])
    return f"{sp}<{name}{attr_str}>{escape(str(value))}</{name}>"


# ======================================================================
#  Agent Exporter with Causal Meta-Diagnostics
# ======================================================================

class AgentExporter:
    """
    Exports FlowGrad diagnostics into structured XML for AI agents.
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
        from flowgrad.analyzers.velocity import detect_stagnation, detect_explosion
        from flowgrad.analyzers.health import layer_health_score, gradient_snr_per_layer

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
        findings_xml = []
        issue_summaries = []

        # A. Inconsistency Detection (Zero LR but loss changing)
        if current_lr == 0.0 and loss_is_improving:
            xml = "\n".join([
                _line("premise", "Learning rate is 0.0 but loss is significantly decreasing.", 0),
                _line("implies", "Major inconsistency: weights might be frozen while other params learn, or scheduler is detached.", 0),
                _line("action", "CHECK_CODE_LOGIC", 0),
                _line("expected_effect", "Resolve ghost updates and ensure reproducible training.", 0),
                _line("confidence", "0.95", 0)
            ])
            findings_xml.append(_tag("causal_model", xml, type="INCONSISTENCY_DETECTED", indent=4))
            issue_summaries.append("LR_INCONSISTENCY")

        # B. Low SNR Analysis (False Positive Suppression)
        snr_data = gradient_snr_per_layer(store)
        low_snr_layers = {name: s[-1] for name, s in snr_data.items() if s and 0 < s[-1] < 0.01}
        
        if low_snr_layers:
            avg_snr = sum(low_snr_layers.values()) / len(low_snr_layers)
            if loss_is_improving and is_adaptive:
                # Suppression Logic: Benign Noise in Adaptive Optimizer
                xml = "\n".join([
                    _line("premise", f"Low gradient SNR (<0.01) detected in {len(low_snr_layers)} layers, BUT loss is actively decreasing and Adam/Adaptive optimizer is used.", 0),
                    _line("implies", "Benign noise. Adaptive optimizers absorb raw variance. The model is in a stable fine-tuning phase despite high stochasticity.", 0),
                    _line("action", "NO_INTERVENTION_REQUIRED", 0),
                    _line("expected_effect", "Suppress false alarms. Continue training.", 0),
                    _line("confidence", "0.85", 0)
                ])
                findings_xml.append(_tag("causal_model", xml, type="BENIGN_NOISE", indent=4))
            else:
                # Legitimate Low SNR Alert
                xml = "\n".join([
                    _line("premise", f"Low SNR (<0.01) in {len(low_snr_layers)} layers and loss is NOT improving.", 0),
                    _line("implies", "Optimizer is taking random walks. Gradient noise is dominating the descent signal.", 0),
                    _line("action", "INCREASE_BATCH_SIZE_OR_REDUCE_LR", 0),
                    _line("expected_effect", "Reduce gradient variance to restore directional descent.", 0),
                    _line("risk", "May slow down epoch processing time if batch size is increased.", 0),
                    _line("confidence", "0.75", 0)
                ])
                findings_xml.append(_tag("causal_model", xml, type="HARMFUL_LOW_SNR", indent=4))
                issue_summaries.append("LOW_SNR")

        # C. Stagnation Analysis
        stagnant = detect_stagnation(store)
        if stagnant:
            if loss_is_improving:
                xml = "\n".join([
                    _line("premise", f"{len(stagnant)} layers have near-zero velocity, but overall loss is decreasing.", 0),
                    _line("implies", "These specific layers have converged early or are redundant, while other layers drive the loss.", 0),
                    _line("action", "CONSIDER_PRUNING", 0),
                    _line("expected_effect", "Remove dead weights to compress model without hurting current descent trajectory.", 0),
                    _line("confidence", "0.80", 0)
                ])
                findings_xml.append(_tag("causal_model", xml, type="SAFE_STAGNATION", indent=4))
            else:
                xml = "\n".join([
                    _line("premise", f"{len(stagnant)} layers stagnated and loss is plateaued.", 0),
                    _line("implies", "Learning has halted prematurely due to vanishing gradients or excessive regularization.", 0),
                    _line("action", "INCREASE_LR_OR_REMOVE_WEIGHT_DECAY", 0),
                    _line("expected_effect", "Force parameters out of local plateau.", 0),
                    _line("confidence", "0.90", 0)
                ])
                findings_xml.append(_tag("causal_model", xml, type="HARMFUL_STAGNATION", indent=4))
                issue_summaries.append("STAGNATION")

        # D. Dead Neurons
        dead_layers = []
        for name in store.layer_names:
            history = store.get_layer_history(name)
            if history and history[-1].dead_ratio > 0.5:
                dead_layers.append(name)
                
        if dead_layers:
            xml = "\n".join([
                _line("premise", f">50% parameters near zero in layers: {', '.join(dead_layers[:3])}...", 0),
                _line("implies", "Activation collapse (e.g., dying ReLU) or extreme sparsity regime.", 0),
                _line("action", "REPLACE_RELU_WITH_LEAKYRELU_OR_GELU", 0),
                _line("expected_effect", "Allow gradients to flow through negative pre-activations to revive neurons.", 0),
                _line("confidence", "0.88", 0)
            ])
            findings_xml.append(_tag("causal_model", xml, type="DEAD_NEURON_COLLAPSE", indent=4))
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

        # ── 4. Assemble XML ─────────────────────────────────────────
        xml_parts = ['<flowgrad_agent_report>']

        # A. History
        if include_history:
            past = HistoryTracker.get_recent_runs(n=5)
            past = [r for r in past if r.get("run_id") != run_name]
            if past:
                hist_lines = []
                for r in past[-3:]:
                    issues_str = ", ".join(r.get("issues", [])) or "None"
                    hist_lines.append(
                        _tag("run", "\n".join([
                            _line("id", r.get("run_id", "?"), 0),
                            _line("timestamp", r.get("timestamp", "?"), 0),
                            _line("optimizer", r.get("optimizer", "?"), 0),
                            _line("steps", r.get("steps", "?"), 0),
                            _line("final_loss", r.get("final_loss", "?"), 0),
                            _line("issues", issues_str, 0),
                            _line("avg_health", r.get("avg_health", "?"), 0),
                        ]), indent=4)
                    )
                xml_parts.append(_tag("experiment_history", "\n".join(hist_lines)))

        # B. Environment
        env = "\n".join([
            _line("optimizer", opt_info["str"], 4),
            _line("lr_scheduler", sched_info["str"], 4),
        ])
        xml_parts.append(_tag("environment", env))

        # C. Architecture
        arch_lines = "\n".join([
            _line("total_params", f"{arch['total_params']/1e6:.2f}M", 4),
            _line("trainable_params", f"{arch['trainable_params']/1e6:.2f}M", 4),
            _line("normalization_layers", arch["normalization_layers"], 4),
            _line("dropout_layers", arch["dropout_layers"], 4),
            _line("activations", ", ".join(arch["activations"]), 4),
        ])
        xml_parts.append(_tag("model_architecture", arch_lines))

        # D. Training State
        state_lines = [_line("current_step", store.num_steps, 4)]
        if valid_losses:
            state_lines.append(_line("initial_loss", f"{valid_losses[0]:.4f}", 4))
            state_lines.append(_line("current_loss", f"{valid_losses[-1]:.4f}", 4))
            if len(valid_losses) > 1:
                state_lines.append(_line("loss_trend", f"{loss_trend:+.4f}", 4))
                state_lines.append(_line("min_loss", f"{min(valid_losses):.4f}", 4))
        xml_parts.append(_tag("training_state", "\n".join(state_lines)))

        # E. Causal Diagnostics
        if findings_xml:
            xml_parts.append(_tag("diagnostics", "\n".join(findings_xml)))
        else:
            status = _line("status", "HEALTHY — No critical issues or inconsistencies detected.", 0)
            xml_parts.append(_tag("diagnostics", status))

        # F. Layer Health Summary
        health_lines = []
        for name in sorted(health, key=lambda k: health[k]):
            score = health[name]
            status = "CRITICAL" if score < 40 else "WARNING" if score < 70 else "HEALTHY"
            health_lines.append(
                f'    <layer name="{escape(name)}" health="{score:.0f}" status="{status}" />'
            )
        xml_parts.append(_tag("layer_health_summary", "\n".join(health_lines)))

        xml_parts.append("</flowgrad_agent_report>")
        return "\n\n".join(xml_parts)

    @classmethod
    def export_embedding(cls, tracker, save: bool = False) -> str:
        """
        Export causal XML diagnostics specifically for an EmbeddingTracker.
        """
        xml_parts = [f'<flowgrad_embedding_report layer="{escape(tracker.name)}">']
        findings_xml = []
        
        summary = tracker.summary()
        
        # Matrix Stats
        stats = "\n".join([
            _line("num_embeddings", summary["num_embeddings"], 4),
            _line("active_coverage_pct", f"{summary['coverage_pct']:.1f}", 4),
            _line("popularity_gini", f"{summary['gini']:.3f}", 4),
        ])
        xml_parts.append(_tag("embedding_stats", stats))
        
        # 1. Zombie Embeddings
        if summary["zombie_pct"] > 5.0:
            xml = "\n".join([
                _line("premise", f"{summary['zombie_pct']:.1f}% of embeddings have high update velocity but strictly alternating/negative cosine similarity between steps.", 0),
                _line("implies", "Optimizer is oscillating. Conflicting gradients from different users/items are pulling these embeddings back and forth without generalizing.", 0),
                _line("action", "DECREASE_LR_BETA_OR_USE_SPARSE_ADAM", 0),
                _line("expected_effect", "Smooth out the trajectory of rare/conflicted items and prevent representation collapse.", 0),
                _line("confidence", "0.92", 0)
            ])
            findings_xml.append(_tag("causal_model", xml, type="ZOMBIE_EMBEDDINGS", indent=4))
            
        # 2. Dead Embeddings
        if summary["dead_pct"] > 50.0:
            xml = "\n".join([
                _line("premise", f"{summary['dead_pct']:.1f}% of embeddings have never received a gradient update.", 0),
                _line("implies", "Severe cold-start sparsity or broken dataloader negative sampling.", 0),
                _line("action", "DOWNSAMPLE_NEGATIVES_OR_USE_HASHING_TRICK", 0),
                _line("expected_effect", "Increase sample efficiency and reduce memory footprint of dead parameters.", 0),
                _line("confidence", "0.88", 0)
            ])
            findings_xml.append(_tag("causal_model", xml, type="DEAD_EMBEDDINGS", indent=4))
            
        # 3. Popularity Bias
        if summary["gini"] > 0.8:
            xml = "\n".join([
                _line("premise", f"Exposure distribution is highly skewed (Gini coefficient: {summary['gini']:.2f} > 0.8).", 0),
                _line("implies", "Model is predominantly optimizing for top-popular items. Long-tail embeddings will suffer from inadequate learning.", 0),
                _line("action", "APPLY_LOG_Q_CORRECTION_OR_INVERSE_FREQUENCY_SAMPLING", 0),
                _line("expected_effect", "Debias the softmax logits and improve long-tail recommendation coverage.", 0),
                _line("risk", "May slightly drop overall accuracy (HR@10) on heavily skewed test sets.", 0),
                _line("confidence", "0.95", 0)
            ])
            findings_xml.append(_tag("causal_model", xml, type="POPULARITY_BIAS", indent=4))

        # Assemble Diagnostics
        if findings_xml:
            xml_parts.append(_tag("diagnostics", "\n".join(findings_xml)))
        else:
            status = _line("status", "HEALTHY — Embedding matrix shows stable learning dynamics and adequate coverage.", 0)
            xml_parts.append(_tag("diagnostics", status))

        xml_parts.append("</flowgrad_embedding_report>")
        return "\n\n".join(xml_parts)

