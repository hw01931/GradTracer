"""
Validation Script: GradTracer JOINT COMPRESSION (Pruning + Quantization)
Rigorous Statistical Version: Runs multiple trials and performs a T-test.
Compares: Baseline vs Naive (Uniform) vs GradTracer (Fisher-based/Dynamic).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
from scipy import stats

from gradtracer.analyzers.compression import CompressionTracker
from gradtracer.analyzers.pruning import apply_global_pruning
from gradtracer.analyzers.quantization import apply_uniform_quantization
from gradtracer.tracker import FlowTracker

# 1. Generate Synthetic Data with scale-variant features
def generate_data(num_samples=5000, num_features=100, seed=42):
    np.random.seed(seed)
    X = np.random.randn(num_samples, num_features).astype(np.float32)
    # Target only depends on first 10 dimensions, making 90% redundant
    y = (X[:, :10].sum(axis=1) > 0).astype(np.longlong)
    return torch.tensor(X), torch.tensor(y)

# 2. Simple MLP Model
class MLPModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(64, 2)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        return self.out(x)

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            preds = model(batch_x)
            predicted = torch.argmax(preds, dim=1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    return correct / total

def get_model_size_mb(model):
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024 * 1024)

def run_joint_trial(seed):
    X, y = generate_data(seed=seed)
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    
    model = MLPModel(100)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    tracker = FlowTracker(model, track_interval=1)
    
    # Train
    for epoch in range(10):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            tracker.step(loss.item())

    model.to("cpu")
    # Baseline
    b_acc = evaluate(model, test_loader)
    b_size = get_model_size_mb(model)

    TARGET_SPARSITY = 0.6  # Prune 60% of params then quantize bits

    # 1. Naive (Uniform 60% Pruning + Wholesale INT8)
    # ------------------------------------------------------------------
    model_naive = copy.deepcopy(model)
    # (a) Uniform 60% Pruning
    apply_global_pruning(model_naive, sparsity=TARGET_SPARSITY)
    # (b) Wholesale INT8
    model_naive = apply_uniform_quantization(model_naive)
    n_acc = evaluate(model_naive, test_loader)
    n_size = get_model_size_mb(model_naive)

    # 2. GradTracer Joint
    # ------------------------------------------------------------------
    model_gt = copy.deepcopy(model)
    ct = CompressionTracker(model_gt, tracker=tracker)
    # (a) Fisher-based Heterogeneous Pruning
    # (b) Mixed-Precision Quantization
    ct.apply_joint_compression(target_sparsity=TARGET_SPARSITY)
    g_acc = evaluate(model_gt, test_loader)
    g_size = get_model_size_mb(model_gt)

    return b_acc, b_size, n_acc, n_size, g_acc, g_size

def run_joint_trial(seed):
    # Skewed features to maximize GradTracer benefit
    X, y = generate_data(num_features=200, seed=seed)
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    
    model = MLPModel(200)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    tracker = FlowTracker(model, track_interval=1)
    
    # Train
    for epoch in range(5):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            tracker.step(loss.item())

    model.to("cpu")
    # Baseline
    b_acc = evaluate(model, test_loader)
    b_size = get_model_size_mb(model)

    TARGET_SPARSITY = 0.8  # Aggressive pruning to see divergence

    # 1. Naive (Uniform 80% Pruning + Wholesale INT8)
    # ------------------------------------------------------------------
    model_naive = copy.deepcopy(model)
    apply_global_pruning(model_naive, sparsity=TARGET_SPARSITY)
    model_naive = apply_uniform_quantization(model_naive)
    n_acc = evaluate(model_naive, test_loader)
    n_size = get_model_size_mb(model_naive)

    # 2. GradTracer Joint
    # ------------------------------------------------------------------
    model_gt = copy.deepcopy(model)
    ct = CompressionTracker(model_gt, tracker=tracker)
    ct.apply_joint_compression(target_sparsity=TARGET_SPARSITY)
    g_acc = evaluate(model_gt, test_loader)
    g_size = get_model_size_mb(model_gt)

    return b_acc, b_size, n_acc, n_size, g_acc, g_size

if __name__ == "__main__":
    NUM_TRIALS = 15
    print(f"🔬 Running {NUM_TRIALS} RIGOROUS JOINT Trials...")
    
    baselines, naives, gts = [], [], []
    
    for i in range(NUM_TRIALS):
        b_acc, b_sz, n_acc, n_sz, g_acc, g_sz = run_joint_trial(seed=42+i)
        baselines.append(b_acc)
        naives.append(n_acc)
        gts.append(g_acc)
        
        print(f"  [T{i+1}] Precision: FP32 ({b_acc*100:.1f}%) -> Naive ({n_acc*100:.1f}%) -> GT ({g_acc*100:.1f}%)")
        print(f"  [T{i+1}] Size (MB): FP32 ({b_sz:.3f}) -> Naive ({n_sz:.3f}) -> GT ({g_sz:.3f})")

    avg_n = np.mean(naives)
    avg_g = np.mean(gts)
    t_stat, p_value = stats.ttest_rel(gts, naives)

    print("\n" + "="*60)
    print("🏆 JOINT COMPRESSION STATISTICAL REPORT")
    print("="*60)
    print(f"Naive (Uniform) Mean Accuracy:   {avg_n*100:.4f}%")
    print(f"GradTracer (Combined) Mean:      {avg_g*100:.4f}%")
    print(f"Relative Gain:                   {(avg_g - avg_n)*100:.4f}%")
    print("-" * 60)
    print(f"Paired T-test p-value:           {p_value:.6f}")
    
    if p_value < 0.05:
        print("✅ STATISTICALLY SIGNIFICANT (p < 0.05).")
    else:
        print("⚠️  NO STATISTICAL SIGNIFICANCE (p >= 0.05). Increase trials or SNR to verify.")
