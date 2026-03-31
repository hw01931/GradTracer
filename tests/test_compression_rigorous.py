"""
Rigorous Production Benchmark: GradTracer vs SOTA Pruning
Evaluates:
1. Accuracy/MSE (Real-world Tabular Data)
2. Model Size (MB)
3. Inference Latency (ms)
4. Peak Memory (MB)

Strategies:
- Baseline (FP32)
- Naive (Uniform 80%)
- Standard (Global L1 Magnitude 80%) - SOTA
- GradTracer (Fisher-based 80% + Mixed Precision)
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import copy

from gradtracer.tracker import FlowTracker
from gradtracer.analyzers.compression import CompressionTracker
from gradtracer.analyzers.pruning import apply_global_pruning

# 1. Real-world Data: California Housing
def get_data():
    data = fetch_california_housing()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    return train_ds, test_ds, X.shape[1]

# 2. Medium-sized MLP (Deep enough for redundant layers)
class DeepMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.layers(x).squeeze()

def run_rigorous_benchmark():
    print("--- Starting Rigorous Compression Benchmark (California Housing) ---")
    train_ds, test_ds, input_dim = get_data()
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1024)
    
    # Baseline Training
    model = DeepMLP(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    # Track with GradTracer to collect Fisher info
    tracker = FlowTracker(model, track_interval=1)
    
    print("Training original model...")
    for epoch in range(10):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            tracker.step(loss.item())
    
    def eval_fn(m):
        m.eval()
        mse = 0
        with torch.no_grad():
            for bx, by in test_loader:
                mse += criterion(m(bx), by).item()
        return mse / len(test_loader)

    # Begin Comparison
    comp_tracker = CompressionTracker(model, eval_fn=eval_fn, tracker=tracker)
    
    # 1. Baseline
    print("\n[1/4] Recording Baseline...")
    snap_orig = comp_tracker.snapshot("Baseline (FP32)")
    
    # 2. Naive (Uniform 80%)
    print("[2/4] Applying Naive Uniform Pruning (80%)...")
    model_naive = copy.deepcopy(model)
    # Uniform 80% per layer
    from gradtracer.analyzers.pruning import apply_heterogeneous_pruning
    plan_naive = {name: 0.8 for name, _ in model_naive.named_modules() if isinstance(_, nn.Linear)}
    apply_heterogeneous_pruning(model_naive, plan_naive)
    comp_naive = CompressionTracker(model_naive, eval_fn=eval_fn)
    snap_naive = comp_naive.snapshot("Naive (Uniform 80%)")
    
    # 3. Standard (Global L1 Magnitude 80%)
    print("[3/4] Applying SOTA Global L1 Pruning (80%)...")
    model_sota = copy.deepcopy(model)
    apply_global_pruning(model_sota, sparsity=0.8)
    comp_sota = CompressionTracker(model_sota, eval_fn=eval_fn)
    snap_sota = comp_sota.snapshot("Global L1 (SOTA 80%)")
    
    # 4. GradTracer (Fisher-based 80% + Mixed Precision)
    print("[4/4] Applying GradTracer Joint Compression (80%)...")
    model_gt = copy.deepcopy(model)
    comp_gt = CompressionTracker(model_gt, eval_fn=eval_fn, tracker=tracker)
    comp_gt.apply_joint_compression(target_sparsity=0.8)
    snap_gt = comp_gt.snapshot("GradTracer (Fisher 80%)")
    
    # Final Report
    print("\n" + "="*80)
    print(f"{'Strategy':<25} {'MSE (Lower Better)':<20} {'Size (MB)':<10} {'Latency (ms)':<15}")
    print("-" * 80)
    for s in [snap_orig, snap_naive, snap_sota, snap_gt]:
        mse = s.eval_metrics.get('score', 0)
        print(f"{s.name:<25} {mse:<20.4f} {s.model_size_mb:<10.2f} {s.inference_latency_ms:<15.2f}")
    print("=" * 80)

if __name__ == "__main__":
    run_rigorous_benchmark()
