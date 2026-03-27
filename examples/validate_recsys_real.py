"""
Validation Script: GradTracer vs MovieLens 100k
Demonstrates EmbeddingTracker & Auto-Fix on a real-world sparse dataset.
"""

import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
from gradtracer.analyzers.embedding import EmbeddingTracker

# 1. MovieLens 100k Robust Downloader
def download_movielens():
    url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
    path = "ml-100k.data"
    if not os.path.exists(path):
        print(f"Downloading MovieLens 100k from {url}...")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(f"Download failed: {e}. Generating synthetic power-law data instead.")
            return generate_synthetic_ml()
    
    df = pd.read_csv(path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    return df

def generate_synthetic_ml(num_users=943, num_items=1682, num_samples=100000):
    """Fallback generator using Zipf (power-law) distribution matching ML-100k."""
    users = np.random.randint(0, num_users, num_samples)
    items = np.random.zipf(1.1, num_samples) % num_items
    ratings = np.random.randint(1, 6, num_samples)
    return pd.DataFrame({'user_id': users, 'item_id': items, 'rating': ratings})

# 2. Preprocessing
def prepare_data(df):
    user_ids = torch.tensor(df['user_id'].values - 1, dtype=torch.long)
    item_ids = torch.tensor(df['item_id'].values - 1, dtype=torch.long)
    ratings = torch.tensor(df['rating'].values, dtype=torch.float32)
    return user_ids, item_ids, ratings

# 3. Model: Neural Matrix Factorization
class NCF(nn.Module):
    def __init__(self, num_users, num_items, dim=16):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        self.fc = nn.Sequential(
            nn.Linear(dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)
        return self.fc(torch.cat([u, i], dim=-1)).squeeze()

# 4. Global Validation Logic
def run_validation():
    df = download_movielens()
    num_users = df['user_id'].max()
    num_items = df['item_id'].max()
    
    u, i, r = prepare_data(df)
    dataset = TensorDataset(u, i, r)
    train_size = int(0.8 * len(dataset))
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    
    loader = DataLoader(train_set, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1024)

    def train_model(auto_fix=False):
        model = NCF(num_users, num_items)
        optimizer = optim.Adam(model.parameters(), lr=0.02)
        criterion = nn.MSELoss()
        tracker = EmbeddingTracker(model.item_emb, auto_fix=auto_fix) if auto_fix else None
        
        for epoch in range(5):
            for bu, bi, br in loader:
                optimizer.zero_grad()
                loss = criterion(model(bu, bi), br)
                loss.backward()
                optimizer.step()
                if tracker: tracker.step()
        
        if tracker: tracker.report()
        return model

    print("\n--- Training Baseline ---")
    base_model = train_model(auto_fix=False)
    
    print("\n--- Training GradTracer (AutoFix) ---")
    gt_model = train_model(auto_fix=True)
    
    def evaluate(model):
        model.eval()
        mse = 0
        with torch.no_grad():
            for bu, bi, br in test_loader:
                mse += nn.MSELoss()(model(bu, bi), br).item()
        return mse / len(test_loader)

    print(f"\nBaseline MSE: {evaluate(base_model):.4f}")
    print(f"GradTracer MSE: {evaluate(gt_model):.4f}")

if __name__ == "__main__":
    run_validation()
