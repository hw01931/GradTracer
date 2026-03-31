"""
FlowTracker — Core DL training dynamics tracker.

Hooks into a PyTorch model and automatically tracks per-layer weight/gradient
statistics at every step.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from gradtracer.snapshot import LayerSnapshot, SnapshotStore, StepRecord
from gradtracer.memory import MemoryTracker
import time

if TYPE_CHECKING:
    pass  # torch types only for type checking

# Lazy torch import to avoid hard dependency at module level
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for FlowTracker. "
                "Install it with: pip install gradtracer[torch]"
            )
    return _torch


class FlowTracker:
    """
    One-line training dynamics tracker for PyTorch models.

    Usage:
        tracker = FlowTracker(model)
        for epoch in range(100):
            loss = train_one_epoch(model, loader, optimizer)
            tracker.step(loss=loss.item())
        tracker.report()
        tracker.plot.velocity_heatmap()

    Args:
        model: PyTorch nn.Module to track.
        track_gradients: Whether to capture gradient statistics during backward.
        track_weights: Whether to capture weight statistics at each step.
        track_dead_neurons: Whether to compute dead neuron ratio.
        include_bias: Whether to include bias parameters (default: False for cleaner viz).
        device: Device for computations ('cpu' or 'cuda'). Defaults to model's device.
    """

    def __init__(
        self,
        model,
        track_gradients: bool = True,
        track_weights: bool = True,
        track_dead_neurons: bool = True,
        include_bias: bool = False,
        device: Optional[str] = None,
        optimizer=None,
        scheduler=None,
        run_name: str = "current_run",
        track_interval: int = 1,
        hook_interval: int = 1,
        mode: str = "full",  # 'full' or 'light' (only tracks high-variance layers)
    ):
        torch = _get_torch()

        self.model = model
        self.track_gradients = track_gradients
        self.track_weights = track_weights
        self.track_dead_neurons = track_dead_neurons
        self.include_bias = include_bias
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.run_name = run_name
        self.track_interval = track_interval
        self.hook_interval = hook_interval
        self.mode = mode
        self.store = SnapshotStore()
        self.memory = MemoryTracker()
        self._start_time = None
        self._overhead_ms = 0.0
        self._total_steps_tracked = 0

        self._step_count = 0
        self._hooks = []
        self._grad_stats: Dict[str, Dict[str, float]] = {}
        self._prev_weights: Dict[str, np.ndarray] = {}
        self._prev_velocity: Dict[str, float] = {}

        # Discover trackable parameters
        self._param_names: List[str] = []
        self._params: Dict[str, Any] = {}
        for name, param in model.named_parameters():
            if not self.include_bias and "bias" in name:
                continue
            if param.requires_grad:
                self._param_names.append(name)
                self._params[name] = param

        # Light mode: Only track the top 25% largest parameters (bottlenecks)
        if self.mode == "light" and self._param_names:
            sorted_params = sorted(
                self._param_names, 
                key=lambda n: self._params[n].numel(), 
                reverse=True
            )
            num_to_keep = max(1, len(sorted_params) // 4)
            self._param_names = sorted_params[:num_to_keep]
            # Keep only selected params
            self._params = {n: self._params[n] for n in self._param_names}

        # Register gradient hooks
        if self.track_gradients:
            self._register_grad_hooks()

    def export_for_agent(self, include_history: bool = True, save: bool = True) -> str:
        """
        Exports training context and diagnostics as structured XML for AI Assistants
        (Cursor, Copilot, Antigravity, etc.).

        The output includes:
        - Experiment history (previous runs from .gradtracer/history.jsonl)
        - Current environment (optimizer, scheduler, architecture)
        - Training state (step, loss)
        - Diagnostics with math/logic explanations and prescriptions

        Args:
            include_history: Whether to include previous experiment runs.
            save: Whether to save this run to the history file.

        Returns:
            XML string optimized for AI agent parsing.
        """
        from gradtracer.agent import AgentExporter
        return AgentExporter.export_dl(
            self,
            run_name=self.run_name,
            include_history=include_history,
            save=save
        )

    def _register_grad_hooks(self):
        """Register backward hooks on all tracked parameters."""
        torch = _get_torch()

        for name, param in self._params.items():

            def _make_hook(pname):
                def hook(grad):
                    # We store the raw tensor on device to avoid backward-pass sync.
                    # It will be processed and moved to CPU in step() only when needed.
                    if self._step_count % self.hook_interval == 0:
                        self._grad_stats[pname] = grad.detach()
                return hook

            h = param.register_hook(_make_hook(name))
            self._hooks.append(h)

    def get_estimated_resource_usage(self) -> Dict[str, Any]:
        """
        Estimate the VRAM and CPU overhead before starting the training loop.
        Useful for production planning and avoiding OOM.
        """
        torch = _get_torch()
        with torch.no_grad():
            num_params = sum(p.numel() for p in self._params.values())
        
            # 1. State Memory: Each tracked parameter needs a 'prev_weight' buffer (float32)
            state_mem_mb = (num_params * 4) / (1024 * 1024)
            
            # 2. Gradient Buffer: Temporary storage for gradients before processing
            grad_buf_mb = (num_params * 4) / (1024 * 1024)
            
            return {
                "estimated_vram_mb": state_mem_mb + grad_buf_mb,
                "estimated_cpu_mem_mb": state_mem_mb,
                "tracked_parameters": len(self._param_names),
                "total_tracked_elements": num_params,
                "recommendation": "High" if state_mem_mb > 500 else "Low"
            }

    def step(self, loss: Optional[float] = None, metrics: Optional[Dict[str, float]] = None):
        """
        Record one training step. Call this once per epoch or per N steps.

        Args:
            loss: Current loss value (optional but recommended).
            metrics: Dict of additional metrics to track (e.g. {'val_loss': 0.5}).
        """
        t0 = time.time()
        torch = _get_torch()
        self._step_count += 1
        
        # Update memory statistics
        self.memory.update()
        
        # Distributed aggregation
        if torch.distributed.is_initialized():
            # Sync loss
            if loss is not None:
                loss_t = torch.tensor([loss], device=torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
                torch.distributed.all_reduce(loss_t, op=torch.distributed.ReduceOp.SUM)
                loss = loss_t.item() / torch.distributed.get_world_size()

        record = StepRecord(
            step=self._step_count,
            loss=loss,
            metrics=metrics or {},
        )

        if self._step_count % self.track_interval != 0:
            self.store.add_step(record)
            return

        for name in self._param_names:
            param = self._params[name]
            snap = LayerSnapshot(name=name, step=self._step_count)

            # Weight statistics
            if self.track_weights:
                with torch.no_grad():
                    w = param.detach().float().cpu()
                    snap.weight_norm = w.norm().item()
                    snap.weight_mean = w.mean().item()
                    snap.weight_std = w.std().item() if w.numel() > 1 else 0.0
                    snap.weight_min = w.min().item()
                    snap.weight_max = w.max().item()
                    snap.num_params = w.numel()

                    # Dead neuron ratio
                    if self.track_dead_neurons:
                        snap.dead_ratio = (w.abs() < 1e-6).float().mean().item()

                    # Velocity and acceleration
                    w_np = w.numpy().flatten()
                    if name in self._prev_weights:
                        delta = w_np - self._prev_weights[name]
                        velocity = float(np.linalg.norm(delta))
                        snap.velocity = velocity

                        if name in self._prev_velocity:
                            snap.acceleration = abs(velocity - self._prev_velocity[name])

                        self._prev_velocity[name] = velocity

                    self._prev_weights[name] = w_np.copy()

            # Gradient statistics (Non-blocking processing)
            if self.track_gradients and name in self._grad_stats:
                g = self._grad_stats[name]
                with torch.no_grad():
                    # Move to CPU only for the summary statistics computation
                    g_cpu = g.float().cpu()
                    snap.grad_norm = g_cpu.norm().item()
                    snap.grad_mean = g_cpu.mean().item()
                    snap.grad_std = g_cpu.std().item() if g_cpu.numel() > 1 else 0.0
                    snap.grad_min = g_cpu.min().item()
                    snap.grad_max = g_cpu.max().item()
                
                # Distributed Sync for Gradient Norms
                if torch.distributed.is_initialized():
                    gn_t = torch.tensor([snap.grad_norm**2], device=g.device)
                    torch.distributed.all_reduce(gn_t, op=torch.distributed.ReduceOp.SUM)
                    snap.grad_norm = (gn_t.item() ** 0.5) / (torch.distributed.get_world_size() ** 0.5)
                
                # Critical: Remove from dict to free GPU/Memory
                del self._grad_stats[name]

            record.layers[name] = snap

        self.store.add_step(record)
        
        # Record overhead for this step
        self._overhead_ms += (time.time() - t0) * 1000
        self._total_steps_tracked += 1

    def report(self, top_k: int = 5) -> None:
        """
        Generate and print a text diagnostic report of the training so far.
        """
        from gradtracer.diagnostics import generate_dl_report
        rep = generate_dl_report(
            self.store, 
            memory_summary=self.memory.get_summary(), 
            overhead_ms=self._overhead_ms,
            top_k=top_k
        )
        print(rep)

    @property
    def plot(self):
        """Access the visualization API."""
        from gradtracer.viz.plots import DLPlotAPI
        return DLPlotAPI(self.store)

    @property
    def history(self) -> SnapshotStore:
        """Access raw collected data."""
        return self.store

    @property
    def summary(self) -> Dict[str, Any]:
        """Quick summary dict for programmatic access."""
        result = {
            "total_steps": self.store.num_steps,
            "num_layers": len(self._param_names),
            "layer_names": self._param_names,
        }

        losses = self.store.get_loss_history()
        valid_losses = [l for l in losses if l is not None]
        if valid_losses:
            result["loss_first"] = valid_losses[0]
            result["loss_last"] = valid_losses[-1]
            result["loss_min"] = min(valid_losses)
            result["loss_improvement"] = valid_losses[0] - valid_losses[-1]

        return result

    def detach(self):
        """Remove all hooks from the model."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._grad_stats.clear()

    def __del__(self):
        try:
            self.detach()
        except Exception:
            pass

    def __repr__(self):
        return (
            f"FlowTracker(layers={len(self._param_names)}, "
            f"steps={self.store.num_steps})"
        )
