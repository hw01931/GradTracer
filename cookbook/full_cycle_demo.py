"""
GradTracer Master Cookbook: The Full Life-Cycle of a DL Model
From Training Dynamics to Mechanistic Interpretability & Strategic Compression.

Scenario: A Recommendation Engine struggling with (1) Long-tail bias and (2) Spurious Shortcuts.
1. [Diagnose] Initial noisy training with Zombie embeddings.
2. [Fix] Auto-Fixing representational collapse.
3. [XAI] Interpreting the 'why' (Grokking, Gradient Starvation).
4. [Compress] Strategic Pruning & Quantization based on Fisher Saliency.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# GradTracer Core Components
from gradtracer.tracker import FlowTracker
from gradtracer.analyzers.embedding import EmbeddingTracker
from gradtracer.analyzers.compression import CompressionTracker
from gradtracer.analyzers.interpretation import InterpretationAdvisor

# --- PHASE 0: The Problem (Messy Real-world Data) ---
def generate_complex_data(num_samples=10000):
    np.random.seed(42)
    # 20 Dense features (Complex) + 1 Shortcut feature (Easy/Spurious)
    dense_X = np.random.randn(num_samples, 20).astype(np.float32)
    shortcut_X = np.random.randn(num_samples, 1).astype(np.float32)
    
    # Label depends heavily on dense, but shortcut is highly correlated (Watermark effect)
    y = ((dense_X[:, 0] + dense_X[:, 1]*2.0) > 0).astype(np.longlong)
    shortcut_X[y == 1] += 5.0 # This is the shortcut "leakage"
    
    X = np.concatenate([dense_X, shortcut_X], axis=1)
    
    # Categorical/Embedding features (Skewed/Long-tail)
    user_ids = np.random.zipf(1.1, num_samples).astype(np.longlong) % 1000
    return torch.tensor(X), torch.tensor(user_ids), torch.tensor(y)

class MasterModel(nn.Module):
    def __init__(self, num_users=1000):
        super().__init__()
        self.embedding = nn.Embedding(num_users, 32)
        # Deep semantic branch
        self.deep_layers = nn.Sequential(
            nn.Linear(21, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.out = nn.Linear(32 + 64, 2)
        
    def forward(self, x, user_id):
        emb = self.embedding(user_id)
        feat = self.deep_layers(x)
        combined = torch.cat([emb, feat], dim=1)
        return self.out(combined)

# --- STEP 1: INITIAL TRAINING & DIAGNOSIS ---
def run_master_cookbook():
    X, user_ids, y = generate_complex_data()
    loader = DataLoader(TensorDataset(X, user_ids, y), batch_size=256, shuffle=True)
    
    model = MasterModel()
    optimizer = optim.Adam(model.parameters(), lr=0.02) # Higher LR to induce Zombie behavior
    
    # 1. Attach FlowTracker (Dense) & EmbeddingTracker (Sparse)
    tracker = FlowTracker(model, run_name="Cookbook_Initial", track_interval=1)
    emb_tracker = EmbeddingTracker(model.embedding, name="User_Embedding")
    
    print("\n🏁 [STEP 1] INITIAL TRAINING (No Adaptation)...")
    for epoch in range(5):
        model.train()
        for bx, bu, by in loader:
            optimizer.zero_grad()
            out = model(bx, bu)
            loss = nn.CrossEntropyLoss()(out, by)
            loss.backward()
            optimizer.step()
            tracker.step(loss.item())
            emb_tracker.step()
        print(f"  Epoch {epoch+1} Complete.")

    # --- STEP 2: MECHANISTIC XAI (What happened?) ---
    print("\n🧐 [STEP 2] INTERPRETATION & CAUSAL REASONING...")
    advisor = InterpretationAdvisor(tracker)
    advisor.report()
    
    # Export for AI Agents (JSON)
    agent_json = tracker.export_for_agent(save=True)
    with open("cookbook_agent_report.json", "w") as f:
        f.write(agent_json)
    print("  ✅ Causal JSON Report exported for AI Agents/IDE integration.")

    # --- STEP 3: INTERVENTION (Auto-Fixing representational collapse) ---
    print("\n🛠️ [STEP 3] ADAPTIVE INTERVENTION (Auto-Fix Zombies)...")
    zombies = emb_tracker.zombie_embeddings()
    print(f"  Detected {len(zombies)} oscillating Zombie embeddings. Applying real-time penalty...")
    emb_tracker.auto_fix = True # Activates dynamic gradient scaling
    
    # Continue training with Auto-Fix
    for epoch in range(2):
        for bx, bu, by in loader:
            optimizer.zero_grad()
            out = model(bx, bu)
            loss = nn.CrossEntropyLoss()(out, by)
            loss.backward()
            optimizer.step()
            tracker.step(loss.item())
            emb_tracker.step()
            
    # --- STEP 4: STRATEGIC COMPRESSION (Fisher Information) ---
    print("\n📉 [STEP 4] JOINT STRATEGIC COMPRESSION (Pruning + Mixed Precision)...")
    model.to("cpu")
    ct = CompressionTracker(model, tracker=tracker)
    # Using Fisher Information to prune 60% and quantize 85% of what's left
    ct.apply_joint_compression(target_sparsity=0.6)
    
    # --- STEP 5: VISUALIZATION DASHBOARDS ---
    print("\n🎨 [STEP 5] GENERATING ULTIMATE DASHBOARDS...")
    tracker.plot.mechanistic("cookbook_xai_dashboard.png")
    # Using Embedding viz too
    from gradtracer.viz.plots import plot_embedding_diagnostics
    plot_embedding_diagnostics(emb_tracker, save_path="cookbook_embedding_dist.png")
    
    print("\n" + "═"*60)
    print("🏆 COOKBOOK COMPLETE!")
    print("═"*60)
    print("Files Generated:")
    print("1. cookbook_agent_report.json (For AI 에이전트)")
    print("2. cookbook_xai_dashboard.png (Grokking & Shortcuts)")
    print("3. cookbook_embedding_dist.png (Popularity & Zombie status)")
    print("\nGradTracer provided the data-driven foundation to find, explain, and fix model flaws.")

if __name__ == "__main__":
    run_master_cookbook()
