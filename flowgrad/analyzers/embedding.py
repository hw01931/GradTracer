"""
FlowGrad Embedding Tracker — RecSys Diagnostics.

Tracks per-embedding training dynamics (velocity, frequency, zombie states)
to diagnose representation collapse, cold-start failures, and embedding drift.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import warnings

import numpy as np

if TYPE_CHECKING:
    pass

_torch = None

def _get_torch():
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            raise ImportError("PyTorch required for EmbeddingTracker.")
    return _torch


class EmbeddingTracker:
    """
    Tracks dynamics of a specific `nn.Embedding` matrix.
    
    Identifies:
    - Dead Embeddings: Never updated or zero gradients
    - Zombie Embeddings: High update velocity but oscillating direction (failing to learn)
    - Frequency-Aware Saliency: Velocity normalized by exposure frequency
    """
    def __init__(self, embedding_layer, name: str = "embedding"):
        self.layer = embedding_layer
        self.name = name
        self.num_embeddings = embedding_layer.num_embeddings
        self.embedding_dim = embedding_layer.embedding_dim
        
        # State tracking per embedding ID
        # Using dicts or numpy arrays; arrays are faster for dense updates, 
        # but dicts are better for huge sparse updates. We'll use numpy arrays 
        # for simplicity since N is usually < 1M.
        self.freqs = np.zeros(self.num_embeddings, dtype=np.int32)
        self.velocities = np.zeros(self.num_embeddings, dtype=np.float32)
        
        # To detect oscillation, we need prev gradient or prev delta.
        # We store the normalized direction of the previous update.
        self._prev_deltas = np.zeros((self.num_embeddings, self.embedding_dim), dtype=np.float32)
        # Cosine similarity EMA between consecutive updates
        self.oscillation_scores = np.zeros(self.num_embeddings, dtype=np.float32) 
        
        self._prev_weights = None
        self.steps = 0
        
        # Register backward hook to get active indices
        self._active_indices = None
        self._hook = self.layer.weight.register_hook(self._grad_hook)

    def _grad_hook(self, grad):
        """Captures which embeddings are being updated this step."""
        torch = _get_torch()
        # For sparse gradients or dense gradients, find non-zero rows
        with torch.no_grad():
            if grad.is_sparse:
                indices = grad._indices()[0].unique().cpu().numpy()
            else:
                # Dense grad: indices where norm > 0
                norms = grad.norm(dim=1)
                indices = torch.nonzero(norms > 1e-8, as_tuple=True)[0].cpu().numpy()
            self._active_indices = indices

    def step(self):
        """Call this after optimizer.step()"""
        torch = _get_torch()
        self.steps += 1
        
        with torch.no_grad():
            curr_weights = self.layer.weight.detach().float().cpu().numpy()
            
            if self._prev_weights is not None and self._active_indices is not None and len(self._active_indices) > 0:
                idx = self._active_indices
                
                # Update frequency
                self.freqs[idx] += 1
                
                # Calculate deltas (W_t - W_{t-1})
                deltas = curr_weights[idx] - self._prev_weights[idx]
                norms = np.linalg.norm(deltas, axis=1) + 1e-8
                
                # Update EMA velocities
                self.velocities[idx] = 0.9 * self.velocities[idx] + 0.1 * norms
                
                # Calculate oscillation (cosine similarity with prev delta)
                normalized_deltas = deltas / norms[:, None]
                prev_normalized = self._prev_deltas[idx]
                
                cos_sims = np.sum(normalized_deltas * prev_normalized, axis=1)
                # cos_sim ~= -1 means it bounced back exactly opposite
                # cos_sim ~= 1 means it kept going same direction
                
                # Update EMA oscillation score
                # If they just appeared for the first time, prev is 0, cos_sim is 0
                valid_mask = np.linalg.norm(prev_normalized, axis=1) > 0
                valid_idx = idx[valid_mask]
                if len(valid_idx) > 0:
                     self.oscillation_scores[valid_idx] = (
                         0.8 * self.oscillation_scores[valid_idx] + 0.2 * cos_sims[valid_mask]
                     )
                
                # Save normalized direction for next time
                self._prev_deltas[idx] = normalized_deltas
            
            self._prev_weights = curr_weights.copy()
            self._active_indices = None

    def dead_embeddings(self) -> List[int]:
        """Returns indices of embeddings that have never been updated."""
        return np.where(self.freqs == 0)[0].tolist()
        
    def zombie_embeddings(self, threshold: float = -0.3) -> List[int]:
        """
        Returns indices of embeddings oscillating back and forth.
        Usually indicates learning rate is too high or conflicting gradients for this item.
        """
        # Must have been updated at least a few times to be a zombie
        mask = (self.freqs > 5) & (self.oscillation_scores < threshold)
        return np.where(mask)[0].tolist()

    def frequency_aware_saliency(self) -> np.ndarray:
        """
        Velocity normalized by exposure frequency.
        Identifies embeddings that move a lot relative to how rarely they are seen.
        """
        eps = 1.0
        return self.velocities / (self.freqs + eps)
        
    def popularity_bias(self) -> Dict[str, float]:
        """
        Compute Gini coefficient and entropy of the exposure distribution.
        """
        f = self.freqs[self.freqs > 0]
        if len(f) == 0:
            return {"gini": 0.0, "entropy": 0.0, "coverage": 0.0}
            
        f_norm = f / f.sum()
        entropy = -np.sum(f_norm * np.log(f_norm + 1e-8))
        
        # Gini
        f_sorted = np.sort(f)
        n = len(f)
        cumx = np.cumsum(f_sorted, dtype=float)
        gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
        
        coverage = len(f) / self.num_embeddings
        
        return {
            "gini": float(gini),
            "entropy": float(entropy),
            "coverage": float(coverage)
        }
        
    def summary(self) -> Dict[str, Any]:
        dead = self.dead_embeddings()
        zombies = self.zombie_embeddings()
        bias = self.popularity_bias()
        
        return {
            "num_embeddings": self.num_embeddings,
            "dead_count": len(dead),
            "dead_pct": len(dead) / self.num_embeddings * 100,
            "zombie_count": len(zombies),
            "zombie_pct": len(zombies) / self.num_embeddings * 100,
            "coverage_pct": bias["coverage"] * 100,
            "gini": bias["gini"]
        }

    def report(self) -> None:
        s = self.summary()
        lines = []
        lines.append("=" * 60)
        lines.append(f"  FlowGrad — Embedding Dynamics ('{self.name}')")
        lines.append("=" * 60)
        lines.append(f"📊 Matrix: {self.num_embeddings} x {self.embedding_dim}")
        lines.append(f"🔍 Coverage (updated at least once): {s['coverage_pct']:.1f}%")
        lines.append(f"📉 Dead items: {s['dead_count']} ({s['dead_pct']:.1f}%)")
        lines.append(f"🧟 Zombie items (oscillating): {s['zombie_count']} ({s['zombie_pct']:.1f}%)")
        lines.append(f"📈 Popularity Gini: {s['gini']:.3f} (1.0 = highly skewed)")
        
        lines.append("")
        lines.append("⚠️  Alerts & Prescriptions")
        if s['dead_pct'] > 5.0:
             lines.append(f"  💀 HIGH DEAD RATE ({s['dead_pct']:.1f}%)")
             lines.append(f"     💊 Recommendation: Downsample negative items, or apply hashing trick.")
        if s['zombie_pct'] > 2.0:
             lines.append(f"  🧟 ZOMBIE COLLAPSE ({s['zombie_pct']:.1f}%)")
             lines.append(f"     💊 Recommendation: Reduce learning rate for sparse parameters (use SparseAdam), or increase batch size to smooth conflicting gradients.")
        if s['gini'] > 0.4:
             lines.append(f"  🎯 EXTREME POPULARITY BIAS (Gini: {s['gini']:.2f})")
             lines.append(f"     💊 Recommendation: Apply log-Q correction to logits, or use inverse/log-frequency sampling for positive items.")
             
        if not (s['dead_pct'] > 5.0 or s['zombie_pct'] > 2.0 or s['gini'] > 0.4):
             lines.append("  ✅ Embedding dynamics are healthy.")
        
        lines.append("=" * 60)
        rep = "\n".join(lines)
        print(rep)
        
    def detach(self):
        if hasattr(self, '_hook'):
            self._hook.remove()
