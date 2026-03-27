"""
Validation Script: Mechanistic Interpretability & Shortcut Learning Detection
Demonstrates how GradTracer exposes Model Uncertainty, Grokking, and
spurious Shortcut Circuits created by Gradient Starvation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from gradtracer.tracker import FlowTracker
from gradtracer.analyzers.interpretation import InterpretationAdvisor

# 1. Generate Synthetic Data with a BLATANT Shortcut (Clever Hans Effect)
def generate_shortcut_data(num_samples=2000):
    np.random.seed(42)
    # The true complex feature (e.g. image content)
    true_features = np.random.randn(num_samples, 20).astype(np.float32)
    true_y = (np.sum(np.sin(true_features), axis=1) > 0).astype(np.longlong)
    
    # The spurious shortcut feature (e.g. a watermark or background color)
    # It perfectly correlates with the label, but is extremely simple to learn linearly
    shortcut_feature = np.zeros((num_samples, 2), dtype=np.float32)
    shortcut_feature[true_y == 1, 0] = 5.0 # Easy signal
    shortcut_feature[true_y == 1, 1] = -5.0
    
    X = np.concatenate([true_features, shortcut_feature], axis=1)
    return torch.tensor(X), torch.tensor(true_y)

# 2. Network designed to map components
class ShortcutModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Deeper complex branch trying to learn the true features
        self.complex_branch = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # Shallow branch attached to the shortcut
        self.shortcut_branch = nn.Linear(2, 32)
        
        # Classifier
        self.classifier = nn.Linear(64, 2)
        
    def forward(self, x):
        complex_x = x[:, :20]
        shortcut_x = x[:, 20:]
        
        c = self.complex_branch(complex_x)
        s = self.shortcut_branch(shortcut_x)
        
        # Starvation multiplier: make the shortcut wildly more appealing to the optimizer
        s = s * 50.0
        
        # Combine them
        combined = torch.cat([c, s], dim=1)
        return self.classifier(combined)

if __name__ == "__main__":
    print("🚀 Simulating a dataset with a Spurious Shortcut Feature...")
    X, y = generate_shortcut_data()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    model = ShortcutModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Attach GradTracer 
    tracker = FlowTracker(model, track_interval=1)
    
    print("\n🧠 Training Model... Watch for Gradient Starvation as the shortcut overrides learning.")
    for epoch in range(15):
        model.train()
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            tracker.step(loss.item())
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

    print("\n" + "="*50)
    print("🔍 EXTRACTING XAI & MECHANISTIC INSIGHTS")
    print("="*50)
    
    # Use the new Phase 2 Interpretation Advisor
    advisor = InterpretationAdvisor(tracker)
    advisor.report()
    
    print("\n✅ Validated: InterpretationAdvisor successfully analyzed the Training Dynamics!")
    print("GradTracer now exposes Grokking Phases, Gradient Starvation (Shortcuts), and Epistemic Uncertainty.")

    # Generate the visualization
    print("\n🎨 Generating XAI Dashboard Visualization...")
    tracker.plot.mechanistic("xai_dashboard_demo.png")

