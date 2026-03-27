"""
Validation Script: GradTracer Auto-Fix vs Baseline
Demonstrates that intercepting 'Zombie' embeddings dynamically improves generalization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

from gradtracer.analyzers.embedding import EmbeddingTracker

# 1. Generate Synthetic Data with Conflicting Gradients (Zombies)
def generate_data(num_users=1000, num_items=500, num_samples=30000):
    np.random.seed(42)
    
    users = np.random.randint(0, num_users, num_samples)
    
    # 80% normal items, 20% "Zombie" items
    # Zombie items are seen very frequently but have completely random signals
    normal_items = np.arange(0, int(num_items * 0.8))
    zombie_items = np.arange(int(num_items * 0.8), num_items)
    
    items = np.zeros(num_samples, dtype=int)
    labels = np.zeros(num_samples, dtype=np.float32)
    
    for i in range(num_samples):
        if np.random.rand() < 0.6:
            # Normal interaction
            item = np.random.choice(normal_items)
            # Consistent label based on item mod
            label = 1.0 if item % 2 == 0 else 0.0
        else:
            # Zombie interaction
            item = np.random.choice(zombie_items)
            # Conflicting label (completely random 50/50)
            label = float(np.random.choice([0, 1]))
            
        items[i] = item
        labels[i] = label
        
    return torch.tensor(users), torch.tensor(items), torch.tensor(labels)

# 2. Matrix Factorization Model
class MFModel(nn.Module):
    def __init__(self, num_users, num_items, dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Initialize
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        
    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)
        dot = (u * i).sum(dim=1) + self.bias
        return torch.sigmoid(dot)

# 3. Training Loop
def train(model, dataloader, epochs=10, lr=0.01, use_gradtracer=False):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    tracker = None
    if use_gradtracer:
        # Attach GradTracer to the item embedding to fix zombie items dynamically
        tracker = EmbeddingTracker(model.item_emb, name="item_emb", auto_fix=True, track_interval=1)
        
    for epoch in range(epochs):
        total_loss = 0
        for batch_users, batch_items, batch_labels in dataloader:
            optimizer.zero_grad()
            preds = model(batch_users, batch_items)
            loss = criterion(preds, batch_labels)
            loss.backward()
            
            # The backward hook in EmbeddingTracker intercepts gradients here
            optimizer.step()
            
            if tracker:
                tracker.step()
                
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f}")
        
    if tracker:
        print("\n--- GradTracer Report ---")
        tracker.report()
        tracker.detach()

def evaluate(model, dataloader):
    model.eval()
    criterion = nn.BCELoss()
    total_loss = 0
    with torch.no_grad():
        for batch_users, batch_items, batch_labels in dataloader:
            preds = model(batch_users, batch_items)
            loss = criterion(preds, batch_labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)


if __name__ == "__main__":
    print("Generating Dataset with Long-Tail (Popularity Bias) Distribution...")
    users, items, labels = generate_data()
    
    # 80/20 Split
    train_size = int(0.8 * len(users))
    dataset = TensorDataset(users, items, labels)
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1024, shuffle=False)
    
    NUM_USERS, NUM_ITEMS = 1000, 500
    EPOCHS = 20
    LR = 0.05 # High learning rate to intentionally cause oscillations in rare items
    
    baseline_model = MFModel(NUM_USERS, NUM_ITEMS)
    gt_model = deepcopy(baseline_model)
    
    print("\n" + "="*50)
    print("📈 Training BASELINE Model (No GradTracer)")
    print("="*50)
    train(baseline_model, train_loader, epochs=EPOCHS, lr=LR, use_gradtracer=False)
    base_test_loss = evaluate(baseline_model, test_loader)
    
    print("\n" + "="*50)
    print("🌊 Training GRADTRACER Model (Auto-Fix=True)")
    print("="*50)
    train(gt_model, train_loader, epochs=EPOCHS, lr=LR, use_gradtracer=True)
    gt_test_loss = evaluate(gt_model, test_loader)
    
    print("\n" + "="*50)
    print("🏆 FINAL VALIDATION RESULTS")
    print("="*50)
    print(f"Baseline Test Loss:   {base_test_loss:.5f}")
    print(f"GradTracer Test Loss: {gt_test_loss:.5f}")
    
    if gt_test_loss < base_test_loss:
        improvement = (base_test_loss - gt_test_loss) / base_test_loss * 100
        print(f"✅ Success! GradTracer Auto-Fix improved generalization by {improvement:.2f}%")
        print("By penalizing oscillating 'Zombie' embeddings, GradTracer prevented representation collapse.")
    else:
        print("❌ GradTracer did not improve the loss in this run.")
