"""
FlowTracker — Core DL training dynamics tracker.

Hooks into a PyTorch model and automatically tracks per-layer weight/gradient
statistics at every step.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from flowgrad.snapshot import LayerSnapshot, SnapshotStore, StepRecord

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
                "Install it with: pip install flowgrad[torch]"
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
        self.store = SnapshotStore()

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

        # Register gradient hooks
        if self.track_gradients:
            self._register_grad_hooks()

    def export_for_agent(self, include_history: bool = True, save: bool = True) -> str:
        """
        Exports training context and diagnostics as structured XML for AI Assistants
        (Cursor, Copilot, Antigravity, etc.).

        The output includes:
        - Experiment history (previous runs from .flowgrad/history.jsonl)
        - Current environment (optimizer, scheduler, architecture)
        - Training state (step, loss)
        - Diagnostics with math/logic explanations and prescriptions

        Args:
            include_history: Whether to include previous experiment runs.
            save: Whether to save this run to the history file.

        Returns:
            XML string optimized for AI agent parsing.
        """
        from flowgrad.agent import AgentExporter
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
                    with torch.no_grad():
                        g = grad.detach().float().cpu()
                        self._grad_stats[pname] = {
                            "norm": g.norm().item(),
                            "mean": g.mean().item(),
                            "std": g.std().item() if g.numel() > 1 else 0.0,
                            "min": g.min().item(),
                            "max": g.max().item(),
                        }
                return hook

            h = param.register_hook(_make_hook(name))
            self._hooks.append(h)

    def step(self, loss: Optional[float] = None, metrics: Optional[Dict[str, float]] = None):
        """
        Record one training step. Call this once per epoch or per N steps.

        Args:
            loss: Current loss value (optional but recommended).
            metrics: Dict of additional metrics to track (e.g. {'val_loss': 0.5}).
        """
        torch = _get_torch()
        self._step_count += 1
        record = StepRecord(
            step=self._step_count,
            loss=loss,
            metrics=metrics or {},
        )

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

            # Gradient statistics
            if self.track_gradients and name in self._grad_stats:
                gs = self._grad_stats[name]
                snap.grad_norm = gs["norm"]
                snap.grad_mean = gs["mean"]
                snap.grad_std = gs["std"]
                snap.grad_min = gs["min"]
                snap.grad_max = gs["max"]

            record.layers[name] = snap

        self.store.add_step(record)

    def report(self, top_k: int = 5) -> str:
        """
        Generate and print a text diagnostic report of the training so far.

        Returns:
            str: Formatted diagnostic report.
        """
        from flowgrad.diagnostics import generate_dl_report
        rep = generate_dl_report(self.store, top_k=top_k)
        print(rep)
        return rep

    @property
    def plot(self):
        """Access the visualization API."""
        from flowgrad.viz.plots import DLPlotAPI
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
