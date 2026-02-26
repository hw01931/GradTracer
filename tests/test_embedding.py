"""Tests for v0.5 EmbeddingTracker module."""
import torch
import torch.nn as nn
from flowgrad import EmbeddingTracker
from flowgrad.agent import AgentExporter


def test_embedding_dead_detection():
    # 10 embeddings, dim 4
    emb = nn.Embedding(10, 4)
    tracker = EmbeddingTracker(emb)
    
    optimizer = torch.optim.SGD(emb.parameters(), lr=0.1)
    
    # Only update indices 0, 1, 2
    for _ in range(5):
        optimizer.zero_grad()
        indices = torch.tensor([0, 1, 2])
        out = emb(indices).sum()
        out.backward()
        optimizer.step()
        tracker.step()
        
    dead = tracker.dead_embeddings()
    assert 0 not in dead
    assert 1 not in dead
    assert 2 not in dead
    assert 3 in dead
    assert len(dead) == 7


def test_embedding_zombie_oscillation():
    # 5 embeddings, dim 2
    emb = nn.Embedding(5, 2)
    tracker = EmbeddingTracker(emb)
    
    optimizer = torch.optim.SGD(emb.parameters(), lr=1.0)
    
    # Force index 0 to oscillate by providing alternating gradients
    for i in range(10):
        optimizer.zero_grad()
        indices = torch.tensor([0, 1])
        out = emb(indices)
        
        # Loss that forces gradients back and forth for index 0
        target = torch.tensor([[1.0, 1.0], [0.1, 0.1]])
        if i % 2 == 1:
            target[0] = -target[0] # oscillate target
            
        loss = ((out - target)**2).sum()
        loss.backward()
        optimizer.step()
        tracker.step()
        
    zombies = tracker.zombie_embeddings(threshold=-0.1)
    assert 0 in zombies  # index 0 should be zombie
    
    summary = tracker.summary()
    assert summary["zombie_count"] >= 1


def test_embedding_popularity_bias():
    emb = nn.Embedding(10, 4)
    tracker = EmbeddingTracker(emb)
    optimizer = torch.optim.SGD(emb.parameters(), lr=0.1)
    
    # Highly skewed exposure: index 0 gets 90 updates, index 1 gets 10 updates
    for _ in range(9):
        optimizer.zero_grad()
        out = emb(torch.tensor([0] * 10)).sum()
        out.backward()
        optimizer.step()
        tracker.step()
        
    for _ in range(1):
        optimizer.zero_grad()
        out = emb(torch.tensor([1] * 10)).sum()
        out.backward()
        optimizer.step()
        tracker.step()
        
    bias = tracker.popularity_bias()
    assert bias["gini"] > 0.0
    assert bias["coverage"] == 0.2  # 2 out of 10
    
    
def test_embedding_report_and_xml():
    emb = nn.Embedding(10, 4)
    tracker = EmbeddingTracker(emb)
    optimizer = torch.optim.SGD(emb.parameters(), lr=0.1)
    
    for _ in range(2):
        optimizer.zero_grad()
        out = emb(torch.tensor([0, 1])).sum()
        out.backward()
        optimizer.step()
        tracker.step()
        
    report = tracker.report()
    assert "Embedding Dynamics" in report
    
    xml = AgentExporter.export_embedding(tracker)
    assert "<flowgrad_embedding_report" in xml
    assert "DEAD_EMBEDDINGS" in xml  # Since 80% are dead
