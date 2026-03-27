"""
Validation Script: GradTracer Heterogeneous Pruning vs Naive Magnitude Pruning
Rigorous Statistical Version: Runs multiple trials and performs a T-test.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
from scipy import stats

from gradtracer.analyzers.pruning import apply_global_pruning, apply_heterogeneous_pruning, PruningAdvisor
from gradtracer.tracker import FlowTracker

# 1. Generate Synthetic Data
def generate_data(num_samples=5000, num_features=50, seed=42):
    np.random.seed(seed)
    X = np.random.randn(num_samples, num_features).astype(np.float32)
    # Target depends on first 10 features (others are absolute noise)
    y = (np.sum(X[:, :10], axis=1) > 0).astype(np.longlong)
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

def run_trial(seed):
    X, y = generate_data(seed=seed)
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    
    model = MLPModel(50)
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

    # Pre-Pruning Baseline
    baseline_acc = evaluate(model, test_loader)

    TARGET_SPARSITY = 0.7 # High sparsity to see divergence

    # 1. Naive
    model_naive = copy.deepcopy(model)
    apply_global_pruning(model_naive, sparsity=TARGET_SPARSITY)
    naive_acc = evaluate(model_naive, test_loader)
    
    # 2. GradTracer
    model_gt = copy.deepcopy(model)
    advisor = PruningAdvisor(tracker)
    pruning_plan = advisor.generate_pruning_plan(target_sparsity=TARGET_SPARSITY)
    apply_heterogeneous_pruning(model_gt, pruning_plan)
    gt_acc = evaluate(model_gt, test_loader)
    
    return baseline_acc, naive_acc, gt_acc

if __name__ == "__main__":
    NUM_TRIALS = 10
    results_baseline = []
    results_naive = []
    results_gt = []

    print(f"🔬 Running {NUM_TRIALS} Trials for Statistical Significance Test...")
    for i in range(NUM_TRIALS):
        b, n, g = run_trial(seed=42+i)
        results_baseline.append(b)
        results_naive.append(n)
        results_gt.append(g)
        print(f"  Trial {i+1}: Baseline {b:.3f} | Naive {n:.3f} | GradTracer {g:.3f}")

    avg_n = np.mean(results_naive)
    avg_g = np.mean(results_gt)
    
    # Paired T-test
    t_stat, p_value = stats.ttest_rel(results_gt, results_naive)

    print("\n" + "="*50)
    print("🏆 STATISTICAL VALIDATION SUMMARY")
    print("="*50)
    print(f"Naive Avg Accuracy:      {avg_n*100:.2f}%")
    print(f"GradTracer Avg Accuracy: {avg_g*100:.2f}%")
    print(f"Delta Improvement:       {(avg_g - avg_n)*100:.3f}%")
    print("-" * 50)
    print(f"Paired T-test p-value:   {p_value:.6f}")
    
    if p_value < 0.05:
        print("✅ SUCCESS: The improvement is STATISTICALLY SIGNIFICANT (p < 0.05).")
    else:
        print("⚠️  WARNING: The improvement is NOT statistically significant (p >= 0.05).")
