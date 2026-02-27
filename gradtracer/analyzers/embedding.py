"""
GradTracer Embedding Tracker — RecSys Diagnostics.

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
    
    In AutoFix mode, it dynamically intercepts and scales down gradients 
    if an embedding oscillates severely without contributing to global loss, using empirical Bayesian scaling.
    """
    def __init__(self, embedding_layer, name: str = "embedding", auto_fix: bool = False, track_interval: int = 1):
        self.layer = embedding_layer
        self.name = name
        self.num_embeddings = embedding_layer.num_embeddings
        self.embedding_dim = embedding_layer.embedding_dim
        self.auto_fix = auto_fix
        self.track_interval = track_interval
        
        # State tracking per embedding ID
        self.freqs = np.zeros(self.num_embeddings, dtype=np.int32)
        self.velocities = np.zeros(self.num_embeddings, dtype=np.float32)
        
        self._prev_deltas = np.zeros((self.num_embeddings, self.embedding_dim), dtype=np.float32)
        self.oscillation_scores = np.zeros(self.num_embeddings, dtype=np.float32) 
        
        # We need this to apply penalties in AutoFix
        # Zombie mask for the *next* update phase
        self._zombie_mask_tensor = None
        
        self._prev_weights = None
        self.steps = 0
        
        # Register backward hook to intercept gradients
        self._active_indices = None
        self._hook = self.layer.weight.register_hook(self._grad_hook)

    def _grad_hook(self, grad):
        """Captures active indices and actively intercepts gradients if auto_fix is enabled."""
        torch = _get_torch()
        
        # 1. Capture active indices for step() logging
        # Only pull to CPU if we are logging this interval to save D2H overhead
        if self.steps % self.track_interval == 0:
            with torch.no_grad():
                if grad.is_sparse:
                    indices = grad._indices()[0].unique().cpu().numpy()
                else:
                    norms = grad.norm(dim=1)
                    indices = torch.nonzero(norms > 1e-8, as_tuple=True)[0].cpu().numpy()
                self._active_indices = indices

        # 2. ⚡ AutoFix: Intercept and scale oscillating gradients in-place in GPU
        if self.auto_fix and self._zombie_mask_tensor is not None:
            # _zombie_mask_tensor is constructed at the end of step(), shape (num_embeddings, 1)
            # It contains scaling factors (e.g. 0.1 for zombies, 1.0 for healthy)
            with torch.no_grad():
                if grad.is_sparse:
                    # Sparse scaling
                    indices = grad._indices()[0]
                    scales = self._zombie_mask_tensor[indices].to(grad._values().device)
                    grad._values().mul_(scales)
                else:
                    # Dense scaling
                    grad.mul_(self._zombie_mask_tensor.to(grad.device))
        
        return grad

    def step(self):
        """Call this after optimizer.step()"""
        torch = _get_torch()
        
        # Skip heavy NumPy logic if it's not the interval
        if self.steps % self.track_interval != 0:
            self.steps += 1
            return
            
        self.steps += 1    
        
        with torch.no_grad():
            curr_weights = self.layer.weight.detach().float().cpu().numpy()
            
            if self._prev_weights is not None and self._active_indices is not None and len(self._active_indices) > 0:
                idx = self._active_indices
                
                # Update frequency
                local_freq = np.zeros(self.num_embeddings, dtype=np.int32)
                local_freq[idx] = 1
                
                # Calculate deltas (W_t - W_{t-1})
                deltas = curr_weights[idx] - self._prev_weights[idx]
                local_norms = np.zeros(self.num_embeddings, dtype=np.float32)
                local_norms[idx] = np.linalg.norm(deltas, axis=1) + 1e-8
                
                # DDP Support: Aggregate stats across all active GPUs
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    device = self.layer.weight.device
                    # We send arrays to GPU for fast NCCL reduce
                    freq_t = torch.tensor(local_freq, dtype=torch.int32, device=device)
                    norm_t = torch.tensor(local_norms, dtype=torch.float32, device=device)
                    
                    dist.all_reduce(freq_t, op=dist.ReduceOp.SUM)
                    dist.all_reduce(norm_t, op=dist.ReduceOp.SUM)
                    
                    # Bring back consolidated stats
                    global_freq = freq_t.cpu().numpy()
                    global_norms = norm_t.cpu().numpy()
                    
                    # In DDP, many embeddings might have been updated globally
                    global_idx_mask = global_freq > 0
                    idx = np.where(global_idx_mask)[0]
                    
                    # For metrics, we average the norms across GPUs that saw the item
                    active_counts = global_freq[idx]
                    norms = global_norms[idx] / active_counts
                    
                    self.freqs[idx] += active_counts
                else:
                    self.freqs[idx] += 1
                    norms = local_norms[idx]
                
                # Update EMA velocities
                self.velocities[idx] = 0.9 * self.velocities[idx] + 0.1 * norms
                
                # Calculate oscillation (cosine similarity with prev delta)
                normalized_deltas = deltas / norms[:, None]
                prev_normalized = self._prev_deltas[idx]
                
                cos_sims = np.sum(normalized_deltas * prev_normalized, axis=1)
                
                valid_mask = np.linalg.norm(prev_normalized, axis=1) > 0
                valid_idx = idx[valid_mask]
                if len(valid_idx) > 0:
                     self.oscillation_scores[valid_idx] = (
                         0.8 * self.oscillation_scores[valid_idx] + 0.2 * cos_sims[valid_mask]
                     )
                
                # Save normalized direction for next time
                self._prev_deltas[idx] = normalized_deltas
                
                # ── Auto-Fix Construction ──
                # Use a Bayesian-inspired empirical weighting. Instead of blindly slicing zombies,
                # we only penalize if they oscillate aggressively (cos_sim < -0.3) AND have seen decent frequency
                if self.auto_fix:
                    scales = np.ones(self.num_embeddings, dtype=np.float32)
                    
                    # Zombie penalty (Oscillation)
                    # For embeddings heavily zig-zagging, their variance isn't helping loss,
                    # so we scale down their gradients to act as local LR decay.
                    # We use self.zombie_embeddings() which includes the momentum safety threshold.
                    zombies = self.zombie_embeddings()
                    if len(zombies) > 0:
                        scales[zombies] = 0.1
                        
                    # Dead revival (Inject minor exploration momentum if never updated)
                    # In a real setup, we might add uniform noise to their gradients, 
                    # but scaling them up aggressively when they do get hit prevents vanishing.
                    revivals = np.where(self.freqs == 0)[0]
                    if len(revivals) > 0:
                        scales[revivals] = 1.5
                        
                    self._zombie_mask_tensor = torch.tensor(scales).unsqueeze(1)
            
            self._prev_weights = curr_weights.copy()
            self._active_indices = None

    def dead_embeddings(self) -> List[int]:
        """Returns indices of embeddings that have never been updated."""
        return np.where(self.freqs == 0)[0].tolist()
        
    def zombie_embeddings(self, threshold: float = -0.3) -> List[int]:
        """
        Returns indices of embeddings oscillating back and forth.
        Usually indicates learning rate is too high or conflicting gradients for this item.
        Safe-guarded against numerical noise in converged embeddings via a velocity threshold.
        """
        # 🛡️ Momentum Defense: Protect early-converged embeddings from numerical noise.
        # If an embedding's velocity is below 50% of the median active velocity, 
        # it has already converged and its micro-vibrations are just noise.
        if np.max(self.velocities) > 0:
            active_vels = self.velocities[self.freqs > 0]
            velocity_threshold = np.median(active_vels) * 0.5 if len(active_vels) > 0 else 0.0
        else:
            velocity_threshold = 0.0
            
        # Must have been updated at least a few times, be oscillating, AND have high enough momentum
        mask = (self.freqs > 5) & (self.oscillation_scores < threshold) & (self.velocities > velocity_threshold)
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
        lines.append(f"  GradTracer — Embedding Dynamics ('{self.name}')")
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
