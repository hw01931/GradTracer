"""
Snapshot data structures for tracking parameter states over time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LayerSnapshot:
    """Snapshot of a single layer's parameter state at one training step."""

    name: str
    step: int

    # Weight statistics
    weight_norm: float = 0.0
    weight_mean: float = 0.0
    weight_std: float = 0.0
    weight_min: float = 0.0
    weight_max: float = 0.0
    num_params: int = 0

    # Gradient statistics (filled if tracking gradients)
    grad_norm: float = 0.0
    grad_mean: float = 0.0
    grad_std: float = 0.0
    grad_min: float = 0.0
    grad_max: float = 0.0

    # Derived dynamics (computed after 2+ steps)
    velocity: float = 0.0           # ||W_t - W_{t-1}||
    acceleration: float = 0.0       # ||ΔW_t - ΔW_{t-1}||

    # Neuron health (for ReLU-like activations)
    dead_ratio: float = 0.0         # fraction of near-zero params


@dataclass
class StepRecord:
    """All layer snapshots for a single training step."""

    step: int
    loss: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    layers: Dict[str, LayerSnapshot] = field(default_factory=dict)


@dataclass
class BoostingSnapshot:
    """Snapshot of a boosting model at one training round."""

    round: int
    # eval metrics: {dataset_name: {metric_name: value}}
    eval_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # feature importance: {feature_name: importance}
    feature_importance: Dict[str, float] = field(default_factory=dict)

# Alias for backward compatibility
BoostingRoundRecord = BoostingSnapshot


class SnapshotStore:
    """Storage and retrieval for training snapshots."""

    def __init__(self):
        self.steps: List[StepRecord] = []
        self._layer_names: List[str] = []

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def layer_names(self) -> List[str]:
        return self._layer_names

    def add_step(self, record: StepRecord):
        if not self._layer_names and record.layers:
            self._layer_names = list(record.layers.keys())
        self.steps.append(record)

    def get_layer_history(self, layer_name: str) -> List[LayerSnapshot]:
        """Get all snapshots for a specific layer across time."""
        return [s.layers[layer_name] for s in self.steps if layer_name in s.layers]

    def get_loss_history(self) -> List[Optional[float]]:
        return [s.loss for s in self.steps]

    def get_metric_history(self, name: str) -> List[float]:
        return [s.metrics.get(name, math.nan) for s in self.steps]

    def get_layer_series(self, layer_name: str, attr: str) -> List[float]:
        """Get a time series of a layer attribute (e.g. 'weight_norm')."""
        history = self.get_layer_history(layer_name)
        return [getattr(snap, attr, math.nan) for snap in history]


class BoostingStore:
    """Storage for boosting model training records."""

    def __init__(self):
        self.rounds: List[BoostingSnapshot] = []
        self._feature_names: List[str] = []
        self._dataset_names: List[str] = []
        self._metric_names: List[str] = []

    @property
    def num_rounds(self) -> int:
        return len(self.rounds)

    def add_round(self, record: BoostingSnapshot):
        if not self._feature_names and record.feature_importance:
            self._feature_names = list(record.feature_importance.keys())
        if not self._dataset_names and record.eval_metrics:
            self._dataset_names = list(record.eval_metrics.keys())
            for ds in self._dataset_names:
                if record.eval_metrics[ds]:
                    self._metric_names = list(record.eval_metrics[ds].keys())
                    break
        self.rounds.append(record)

    def get_feature_importance_series(self, feature: str) -> List[float]:
        """Get importance of a feature across all rounds."""
        return [r.feature_importance.get(feature, 0.0) for r in self.rounds]

    def get_eval_metric_series(self, dataset: str, metric: str) -> List[float]:
        """Get an eval metric series for a dataset across rounds."""
        return [
            r.eval_metrics.get(dataset, {}).get(metric, math.nan)
            for r in self.rounds
        ]

    def get_all_feature_names(self) -> List[str]:
        return self._feature_names

    def get_all_dataset_names(self) -> List[str]:
        return self._dataset_names

    def get_all_metric_names(self) -> List[str]:
        return self._metric_names
