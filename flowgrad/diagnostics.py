"""
Diagnostics engine — automated training diagnosis with text prescriptions.

Analyzes collected training data and produces actionable recommendations.
"""
from __future__ import annotations

from typing import List

from flowgrad.snapshot import BoostingStore, SnapshotStore


# ──────────────────────────────────────────────────────────────────────
#  DL Diagnostics
# ──────────────────────────────────────────────────────────────────────

def generate_dl_report(store: SnapshotStore, top_k: int = 5) -> str:
    """
    Generate a comprehensive text diagnostic report for DL training.

    Returns:
        Formatted text report with findings and prescriptions.
    """
    from flowgrad.analyzers.velocity import detect_stagnation, detect_explosion
    from flowgrad.analyzers.health import layer_health_score, gradient_snr_per_layer

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("  FlowGrad — DL Training Diagnostic Report")
    lines.append("=" * 60)
    lines.append("")

    # Overview
    lines.append(f"📊 Total steps tracked: {store.num_steps}")
    lines.append(f"📐 Layers tracked: {len(store.layer_names)}")
    losses = store.get_loss_history()
    valid_losses = [l for l in losses if l is not None]
    if valid_losses:
        lines.append(f"📉 Loss: {valid_losses[0]:.6f} → {valid_losses[-1]:.6f} "
                      f"(Δ = {valid_losses[0] - valid_losses[-1]:+.6f})")
        lines.append(f"   Min loss: {min(valid_losses):.6f} at step "
                      f"{valid_losses.index(min(valid_losses)) + 1}")
    lines.append("")

    # Health scores
    lines.append("─" * 60)
    lines.append("🏥 Layer Health Scores")
    lines.append("─" * 60)
    scores = layer_health_score(store)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])

    for name, score in sorted_scores:
        bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
        emoji = "🟢" if score >= 70 else "🟡" if score >= 40 else "🔴"
        lines.append(f"  {emoji} {name:<40s} [{bar}] {score:.0f}/100")
    lines.append("")

    # Alerts
    alerts_found = False
    lines.append("─" * 60)
    lines.append("⚠️  Alerts & Prescriptions")
    lines.append("─" * 60)

    # Stagnation
    stagnant = detect_stagnation(store)
    if stagnant:
        alerts_found = True
        for alert in stagnant[:top_k]:
            lines.append(f"  🧊 STAGNATION: '{alert['name']}'")
            lines.append(f"     Velocity ≈ {alert['current_velocity']:.2e} "
                          f"since step {alert['stagnant_since_step']}")
            lines.append(f"     💊 Prescription: Consider increasing learning rate "
                          f"or removing weight decay for this layer.")
            lines.append("")

    # Explosions
    explosions = detect_explosion(store)
    if explosions:
        alerts_found = True
        # Deduplicate by layer
        seen = set()
        for alert in explosions:
            key = (alert["name"], alert["type"])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"  💥 {alert['type'].upper()}: '{alert['name']}'")
            lines.append(f"     Value: {alert['value']:.2e} at step {alert['step']}")
            if alert["type"] == "gradient_explosion":
                lines.append(f"     💊 Prescription: Add gradient clipping "
                              f"(max_norm=1.0) or reduce learning rate by 50%.")
            else:
                lines.append(f"     💊 Prescription: Reduce learning rate or "
                              f"check for numerical instability.")
            lines.append("")

    # Low SNR
    snr_data = gradient_snr_per_layer(store)
    low_snr_layers = []
    for name, series in snr_data.items():
        if series and series[-1] < 0.01 and series[-1] > 0:
            low_snr_layers.append((name, series[-1]))

    if low_snr_layers:
        alerts_found = True
        low_snr_layers.sort(key=lambda x: x[1])
        for name, snr_val in low_snr_layers[:top_k]:
            lines.append(f"  📡 LOW GRADIENT SNR: '{name}'")
            lines.append(f"     SNR = {snr_val:.4e}")
            lines.append(f"     💊 Prescription: Increase batch size or "
                          f"check if this layer's learning rate is too high.")
            lines.append("")

    # Dead neurons
    for name in store.layer_names:
        latest = store.get_layer_history(name)
        if latest and latest[-1].dead_ratio > 0.5:
            alerts_found = True
            lines.append(f"  💀 DEAD NEURONS: '{name}'")
            lines.append(f"     {latest[-1].dead_ratio * 100:.1f}% of parameters are near-zero")
            lines.append(f"     💊 Prescription: Consider using LeakyReLU or "
                          f"PReLU instead of ReLU. Check initialization.")
            lines.append("")

    if not alerts_found:
        lines.append("  ✅ No critical issues detected. Training looks healthy!")
        lines.append("")

    lines.append("=" * 60)
    report = "\n".join(lines)
    return report


# ──────────────────────────────────────────────────────────────────────
#  Boosting Diagnostics
# ──────────────────────────────────────────────────────────────────────

def generate_boosting_report(store: BoostingStore) -> str:
    """
    Generate a text diagnostic report for boosting model training.

    Returns:
        Formatted text report with findings and prescriptions.
    """
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("  FlowGrad — Boosting Model Diagnostic Report")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"📊 Total rounds tracked: {store.num_rounds}")
    lines.append(f"📐 Features tracked: {len(store.get_all_feature_names())}")
    lines.append("")

    # Eval metrics summary
    datasets = store.get_all_dataset_names()
    metrics = store.get_all_metric_names()

    if datasets and metrics:
        lines.append("─" * 60)
        lines.append("📈 Evaluation Metrics Summary")
        lines.append("─" * 60)
        for ds in datasets:
            for met in metrics:
                series = store.get_eval_metric_series(ds, met)
                valid_series = [v for v in series if v == v]  # filter NaN
                if valid_series:
                    lines.append(f"  {ds}/{met}: {valid_series[0]:.6f} → {valid_series[-1]:.6f} "
                                  f"(best: {min(valid_series):.6f})")
        lines.append("")

    # Overfitting detection
    if len(datasets) >= 2 and metrics:
        lines.append("─" * 60)
        lines.append("🔍 Overfitting Analysis")
        lines.append("─" * 60)
        met = metrics[0]

        # Find train and valid
        train_ds, valid_ds = None, None
        for ds in datasets:
            dl = ds.lower()
            if "train" in dl or "learn" in dl:
                train_ds = ds
            elif "valid" in dl or "test" in dl or "eval" in dl:
                valid_ds = ds
        if not train_ds:
            train_ds = datasets[0]
        if not valid_ds:
            valid_ds = datasets[1]

        train_s = store.get_eval_metric_series(train_ds, met)
        valid_s = store.get_eval_metric_series(valid_ds, met)
        min_len = min(len(train_s), len(valid_s))

        if min_len > 10:
            # Check if gap is increasing in last 25%
            quarter = max(min_len // 4, 1)
            early_gaps = [abs(train_s[i] - valid_s[i]) for i in range(quarter)]
            late_gaps = [abs(train_s[i] - valid_s[i]) for i in range(min_len - quarter, min_len)]
            avg_early = sum(early_gaps) / len(early_gaps) if early_gaps else 0
            avg_late = sum(late_gaps) / len(late_gaps) if late_gaps else 0

            if avg_late > avg_early * 2 and avg_late > 0.01:
                lines.append(f"  ⚠️ OVERFITTING DETECTED")
                lines.append(f"     Train-Valid gap grew from {avg_early:.4f} to {avg_late:.4f}")
                # Find best valid point
                best_valid_idx = valid_s.index(min(valid_s))
                lines.append(f"     💊 Prescription: Best validation at round {best_valid_idx + 1}. "
                              f"Set num_boost_round={best_valid_idx + 1} or use early_stopping.")
            else:
                lines.append(f"  ✅ No significant overfitting detected.")
        else:
            lines.append(f"  ℹ️ Not enough rounds ({min_len}) for overfitting analysis.")
        lines.append("")

    # Feature importance analysis
    features = store.get_all_feature_names()
    if features:
        lines.append("─" * 60)
        lines.append("🏆 Feature Importance Analysis")
        lines.append("─" * 60)

        # Top features by total importance
        totals = {}
        for feat in features:
            totals[feat] = sum(store.get_feature_importance_series(feat))
        sorted_feats = sorted(totals.items(), key=lambda x: -x[1])

        lines.append("  Top 10 features (total importance):")
        for i, (feat, imp) in enumerate(sorted_feats[:10], 1):
            bar_len = int(imp / max(totals.values()) * 20) if totals.values() else 0
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"  {i:2d}. {feat:<30s} [{bar}] {imp:.2f}")
        lines.append("")

        # Feature drift analysis
        if store.num_rounds >= 2:
            lines.append("  📊 Feature Drift (importance shift between first/second half):")
            drifts = {}
            for feat in features:
                series = store.get_feature_importance_series(feat)
                half = len(series) // 2
                if half > 0:
                    first_avg = sum(series[:half]) / half
                    second_avg = sum(series[half:]) / max(len(series) - half, 1)
                    if first_avg > 0:
                        drifts[feat] = (second_avg - first_avg) / first_avg * 100
                    elif second_avg > 0:
                        drifts[feat] = 100.0
                    else:
                        drifts[feat] = 0.0

            rising = sorted(drifts.items(), key=lambda x: -x[1])[:5]
            declining = sorted(drifts.items(), key=lambda x: x[1])[:5]

            lines.append("    📈 Rising:")
            for feat, pct in rising:
                if pct > 0:
                    lines.append(f"       ↑ {feat}: +{pct:.1f}%")
            lines.append("    📉 Declining:")
            for feat, pct in declining:
                if pct < 0:
                    lines.append(f"       ↓ {feat}: {pct:.1f}%")
            lines.append("")

        # Unused features
        unused = [f for f, t in totals.items() if t < 0.01]
        if unused:
            lines.append(f"  🗑️ Potentially unused features ({len(unused)}):")
            for feat in unused[:10]:
                lines.append(f"     - {feat}")
            lines.append(f"     💊 Prescription: Consider removing these features "
                          f"to reduce model complexity and training time.")
            lines.append("")

    lines.append("=" * 60)
    report = "\n".join(lines)
    return report
