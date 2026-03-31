"""
GradTracer Visualization Module

Provides human-readable, visual diagnostics for embeddings, trees, and dense layers.
"""
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_embedding_diagnostics(tracker, top_k: int = 20, save_path: Optional[str] = None):
    """
    Plots a human-readable visual diagnostic for an EmbeddingTracker.
    
    Creates a 1x3 panel figure showing:
    1. Exposure Frequency (Popularity Bias)
    2. Embedding Velocity (Zombie vs Healthy)
    3. Oscillation Scores
    """
    # Defensive check: Ensure there's data to plot
    if tracker.steps == 0 or np.sum(tracker.freqs) == 0:
        print("⚠️ No data tracked yet. Run training steps with the tracker active before plotting.")
        return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Identify specific sets
    dead_idx = tracker.dead_embeddings()
    zombie_idx = tracker.zombie_embeddings()
    
    # Sort by frequency (Popularity)
    sorted_idx = np.argsort(tracker.freqs)[::-1]
    plot_idx = sorted_idx[:top_k]
    
    # 1. Frequency Distribution Bar Plot
    freq_vals = tracker.freqs[plot_idx]
    
    colors1 = []
    for idx in plot_idx:
        if idx in dead_idx: colors1.append("gray")
        elif idx in zombie_idx: colors1.append("red")
        else: colors1.append("steelblue")
        
    axes[0].bar(range(len(plot_idx)), freq_vals, color=colors1)
    axes[0].set_title(f"Top {top_k} Exposed Embeddings", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Embedding Rank (by popularity)")
    axes[0].set_ylabel("Update Frequency")
    axes[0].set_xticks(range(len(plot_idx)))
    axes[0].set_xticklabels([f"ID:{i}" for i in plot_idx], rotation=90)
    
    # 2. Velocity Space (Scatter)
    active_mask = tracker.freqs > 0
    active_vels = tracker.velocities[active_mask]
    active_freqs = tracker.freqs[active_mask]
    active_oscil = tracker.oscillation_scores[active_mask]
    
    axes[1].scatter(active_freqs, active_vels, alpha=0.4, color="steelblue", label="Healthy")
    
    if len(zombie_idx) > 0:
        z_freqs = tracker.freqs[zombie_idx]
        z_vels = tracker.velocities[zombie_idx]
        axes[1].scatter(z_freqs, z_vels, color="red", alpha=0.9, edgecolor="black", label="Zombies (Oscillating)")
        
    axes[1].set_title("Velocity vs Exposure", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Exposure Frequency")
    axes[1].set_ylabel("EMA Velocity (Change Magnitude)")
    axes[1].legend()
    
    # 3. Oscillation Distribution
    valid_oscil = tracker.oscillation_scores[tracker.freqs > 5]
    if len(valid_oscil) > 0:
        sns.histplot(valid_oscil, bins=30, ax=axes[2], color="purple", kde=True)
        axes[2].axvline(x=-0.3, color="red", linestyle="--", label="Zombie Threshold (-0.3)")
    
    axes[2].set_title("Oscillation Distribution (Cos Sim)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("EMA Cosine Similarity to Prev Step")
    axes[2].set_ylabel("Count")
    axes[2].legend()
    
    plt.tight_layout()
    
    # Add an explanatory text box at the bottom
    caption = (
        "🧠 HUMAN-READABLE DIAGNOSTIC GUIDE:\n"
        "• Blue (Healthy): These embeddings are learning efficiently. They move when updated.\n"
        "• Red (Zombies): High update frequency, but direction constantly reverses (Oscillation < -0.3). They are stuck in a tug-of-war. Action: Use SparseAdam or reduce LR.\n"
        "• Gray/Zero (Dead): Never updated. Check dataloader negative sampling. Action: Downsample or Hash."
    )
    plt.figtext(0.5, -0.15, caption, wrap=True, horizontalalignment='center', fontsize=11, 
                bbox={"facecolor":"#f9f9f9", "alpha":0.8, "pad":10, "boxstyle":"round,pad=1"})
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"📊 Saved visualization to {save_path}")
    else:
        plt.show()

class DLPlotAPI:
    """Provides human-readable visualization for deep learning training dynamics."""
    
    def __init__(self, store):
        self.store = store

    def velocity_heatmap(self, save_path: Optional[str] = None):
        """Plots a heatmap of layer velocity over time."""
        sns.set_theme(style="white")
        layers = self.store.layer_names
        if not layers: return
        
        data = []
        for name in layers:
            data.append(self.store.get_layer_series(name, "velocity"))
            
        plt.figure(figsize=(12, 8))
        sns.heatmap(data, yticklabels=layers, cmap="YlGnBu")
        plt.title("Layer Velocity Heatmap")
        if save_path: plt.savefig(save_path); plt.close()
        else: plt.show()

    def weight_distribution(self, layer_name: str, save_path: Optional[str] = None):
        """Plots the weight distribution of a specific layer."""
        series = self.store.get_layer_series(layer_name, "weight_mean")
        plt.figure(figsize=(10, 6))
        sns.histplot(series, kde=True)
        plt.title(f"Weight Distribution: {layer_name}")
        if save_path: plt.savefig(save_path); plt.close()
        else: plt.show()

    def gradient_flow(self, save_path: Optional[str] = None):
        """Plots the gradient norm flow across all layers."""
        plt.figure(figsize=(12, 6))
        for name in self.store.layer_names:
            series = self.store.get_layer_series(name, "grad_norm")
            plt.plot(series, label=name, alpha=0.6)
        plt.title("Gradient Norm Flow")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        if save_path: plt.savefig(save_path); plt.close()
        else: plt.show()

    def gradient_snr(self, save_path: Optional[str] = None):
        """Plots the Gradient SNR for all layers."""
        from gradtracer.analyzers.health import gradient_snr_per_layer
        snr_dict = gradient_snr_per_layer(self.store)
        plt.figure(figsize=(12, 6))
        for name, series in snr_dict.items():
            plt.plot(series, label=name)
        plt.title("Gradient Signal-to-Noise Ratio (SNR)")
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        if save_path: plt.savefig(save_path); plt.close()
        else: plt.show()

    def health_dashboard(self, save_path: Optional[str] = None):
        """Plots a summary of layer health scores."""
        from gradtracer.analyzers.health import layer_health_score
        scores = layer_health_score(self.store)
        names = list(scores.keys())
        vals = list(scores.values())
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=vals, y=names, palette="RdYlGn")
        plt.axvline(x=70, color='green', linestyle='--')
        plt.axvline(x=40, color='red', linestyle='--')
        plt.title("Layer Health Dashboard")
        if save_path: plt.savefig(save_path); plt.close()
        else: plt.show()

    def full_report(self, save_path: Optional[str] = None):
        """Generates all visualizations into one dashboard."""
        self.mechanistic(save_path=save_path)

    def mechanistic(self, save_path: Optional[str] = None):
        """
        Generates a beautiful 1x3 XAI Dashboard displaying:
        1. Grokking Progress (Memorization vs Circuit Formation)
        2. Gradient Starvation (Shortcut Learning Flow)
        3. Model Epistemic Uncertainty
        """
        # We need the InterpretationAdvisor, but we only have store.
        # We can construct a dummy tracker or just pass a mock.
        from gradtracer.analyzers.interpretation import InterpretationAdvisor
        class MockTracker:
            def __init__(self, s): self.store = s
        
        advisor = InterpretationAdvisor(MockTracker(self.store))
        
        grokking = advisor.grokking_progress()
        shortcuts = advisor.detect_shortcut_learning()
        uncertainty = advisor.epistemic_uncertainty_profile()
        
        sns.set_theme(style="darkgrid", palette="pastel")
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 1. Grokking Progress Bar Chart
        layers = list(grokking.keys())
        prog_scores = [grokking[l]["progress_score"] for l in layers]
        phases = [grokking[l]["phase"] for l in layers]
        
        colors = []
        for p in phases:
            if p == "memorization": colors.append("#ff9999") # Red
            elif p == "circuit_formation": colors.append("#99ff99") # Green
            else: colors.append("#99ccff") # Blue
            
        axes[0].barh(layers, prog_scores, color=colors, edgecolor='black', linewidth=0.5)
        axes[0].set_title("1. Grokking & Circuit Formation", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Formation Progress (0 -> 1)")
        axes[0].set_xlim(0, 1.1)
        axes[0].invert_yaxis()
        
        # 2. Epistemic Uncertainty
        uncert_scores = [uncertainty.get(l, 0.0) for l in layers]
        sns.heatmap(np.array(uncert_scores).reshape(-1, 1), annot=True, cmap="YlOrRd", 
                    yticklabels=layers, xticklabels=["Uncertainty"], ax=axes[1], cbar=False)
        axes[1].set_title("2. Epistemic Uncertainty Profile", fontsize=14, fontweight="bold")
        
        # 3. Gradient Flow (Starvation)
        num_steps = len(self.store.get_layer_series(layers[0], "grad_norm"))
        if num_steps > 0:
            total_flow = np.zeros(num_steps)
            for name in layers:
                total_flow += np.array(self.store.get_layer_series(name, "grad_norm"))
            total_flow += 1e-12
            
            for name in layers:
                rel_flow = np.array(self.store.get_layer_series(name, "grad_norm")) / total_flow
                
                # Highlight dominant/starved
                lw, alpha = 2.0, 0.7
                if name in shortcuts["dominant_circuits"]:
                    lw, alpha = 4.0, 1.0
                elif name in shortcuts["starved_circuits"]:
                    alpha = 0.3
                    
                axes[2].plot(rel_flow, label=name if lw == 4.0 else None, linewidth=lw, alpha=alpha)
                
            axes[2].set_title("3. Gradient Flow & Starvation", fontsize=14, fontweight="bold")
            axes[2].set_xlabel("Training Step")
            axes[2].set_ylabel("Relative Flow Share")
            if len(shortcuts["dominant_circuits"]) > 0:
                axes[2].legend(title="Dominant Shortcut")
                
        plt.tight_layout()
        
        # Add interpretation text
        caption = "🔧 DYNAMICS XAI >> Green=Grokking, Red=Memorization | High Uncertainty=Model is Guessing | Starvation=Shortcut taking over"
        plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=12,
                    bbox={"facecolor":"#333333", "alpha":0.9, "pad":10, "boxstyle":"round,pad=0.5"}, color="white")
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"📊 Mechanistic XAI Dashboard saved to {save_path}")
        else:
            plt.show()
