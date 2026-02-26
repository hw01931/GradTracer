"""
SklearnTracker — Training dynamics tracker for scikit-learn models.

Supports:
  - GradientBoosting* (warm_start iteration tracking)
  - RandomForest / ExtraTrees (per-tree analysis)
  - Any model with partial_fit() (incremental learning)
  - Any model (before/after training comparison)

Usage:
    from flowgrad import SklearnTracker

    # Gradient Boosting (warm_start)
    tracker = SklearnTracker()
    model = GradientBoostingClassifier(n_estimators=200, warm_start=True)
    tracker.track_warm_start(model, X_train, y_train, X_val, y_val, step_size=10)
    tracker.report()

    # RandomForest (post-hoc tree analysis)
    model = RandomForestClassifier(n_estimators=100).fit(X, y)
    tracker = SklearnTracker.from_forest(model, feature_names=X.columns.tolist())
    tracker.report()
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from flowgrad.snapshot import BoostingRoundRecord, BoostingStore


class SklearnTracker:
    """
    Unified sklearn model training tracker.

    Tracks feature importance evolution and eval metrics across
    training stages for scikit-learn models.
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names
        self.store = BoostingStore()
        self._model_type: Optional[str] = None

    # ------------------------------------------------------------------
    # Warm-start tracking (GradientBoosting, HistGradientBoosting)
    # ------------------------------------------------------------------
    def track_warm_start(
        self,
        model,
        X_train, y_train,
        X_val=None, y_val=None,
        step_size: int = 10,
        max_estimators: Optional[int] = None,
        scoring: str = "auto",
    ) -> "SklearnTracker":
        """
        Track a warm_start compatible model by fitting in increments.

        Works with: GradientBoostingClassifier/Regressor,
                    HistGradientBoostingClassifier/Regressor,
                    RandomForestClassifier/Regressor (warm_start=True)

        Args:
            model: sklearn estimator with warm_start=True.
            X_train, y_train: Training data.
            X_val, y_val: Optional validation data.
            step_size: Number of estimators to add per step.
            max_estimators: Total estimators target. If None, uses model's n_estimators.
            scoring: Metric to use. 'auto' picks based on model type.
        """
        from sklearn.metrics import (
            accuracy_score, mean_squared_error, log_loss, r2_score
        )

        self._model_type = type(model).__name__

        if not getattr(model, "warm_start", False):
            model.warm_start = True

        if max_estimators is None:
            max_estimators = getattr(model, "n_estimators", 100)

        # Detect task type
        is_classifier = hasattr(model, "predict_proba") or hasattr(model, "classes_")

        if self.feature_names is None:
            if hasattr(X_train, "columns"):
                self.feature_names = list(X_train.columns)
            else:
                n_features = X_train.shape[1] if hasattr(X_train, "shape") else len(X_train[0])
                self.feature_names = [f"feature_{i}" for i in range(n_features)]

        current = step_size
        while current <= max_estimators:
            model.n_estimators = current
            model.fit(X_train, y_train)

            record = BoostingRoundRecord(round=current)

            # Eval metrics
            if is_classifier:
                train_pred = model.predict(X_train)
                train_acc = accuracy_score(y_train, train_pred)
                record.eval_metrics["train"] = {"accuracy": train_acc}

                if hasattr(model, "predict_proba"):
                    try:
                        train_proba = model.predict_proba(X_train)
                        record.eval_metrics["train"]["log_loss"] = log_loss(y_train, train_proba)
                    except Exception:
                        pass

                if X_val is not None and y_val is not None:
                    val_pred = model.predict(X_val)
                    val_acc = accuracy_score(y_val, val_pred)
                    record.eval_metrics["valid"] = {"accuracy": val_acc}

                    if hasattr(model, "predict_proba"):
                        try:
                            val_proba = model.predict_proba(X_val)
                            record.eval_metrics["valid"]["log_loss"] = log_loss(y_val, val_proba)
                        except Exception:
                            pass
            else:
                train_pred = model.predict(X_train)
                train_mse = mean_squared_error(y_train, train_pred)
                train_r2 = r2_score(y_train, train_pred)
                record.eval_metrics["train"] = {"mse": train_mse, "r2": train_r2}

                if X_val is not None and y_val is not None:
                    val_pred = model.predict(X_val)
                    val_mse = mean_squared_error(y_val, val_pred)
                    val_r2 = r2_score(y_val, val_pred)
                    record.eval_metrics["valid"] = {"mse": val_mse, "r2": val_r2}

            # Feature importance
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                record.feature_importance = {
                    name: float(imp)
                    for name, imp in zip(self.feature_names, importances)
                }

            self.store.add_round(record)
            current += step_size

        return self

    # ------------------------------------------------------------------
    # Forest analysis (per-tree importance)
    # ------------------------------------------------------------------
    @classmethod
    def from_forest(
        cls,
        model,
        feature_names: Optional[List[str]] = None,
        X_val=None, y_val=None,
    ) -> "SklearnTracker":
        """
        Analyze a trained RandomForest/ExtraTrees by inspecting individual trees.

        Creates a synthetic "round-by-round" view where each round = adding one tree.
        """
        tracker = cls(feature_names=feature_names)
        tracker._model_type = type(model).__name__

        if not hasattr(model, "estimators_"):
            raise ValueError("Model must be fitted first (no estimators_ found).")

        n_features = model.n_features_in_
        if tracker.feature_names is None:
            if hasattr(model, "feature_names_in_"):
                tracker.feature_names = list(model.feature_names_in_)
            else:
                tracker.feature_names = [f"feature_{i}" for i in range(n_features)]

        # Analyze each tree
        for idx, tree in enumerate(model.estimators_):
            record = BoostingRoundRecord(round=idx + 1)

            # Per-tree feature importance
            if hasattr(tree, "feature_importances_"):
                imp = tree.feature_importances_
            elif hasattr(tree, "tree_"):
                # Single tree in forest
                imp = tree.tree_.compute_feature_importances(normalize=True)
            else:
                continue

            record.feature_importance = {
                name: float(v) for name, v in zip(tracker.feature_names, imp)
            }

            tracker.store.add_round(record)

        return tracker

    # ------------------------------------------------------------------
    # Incremental learning (partial_fit)
    # ------------------------------------------------------------------
    def track_partial_fit(
        self,
        model,
        X_batches, y_batches,
        classes=None,
        X_val=None, y_val=None,
    ) -> "SklearnTracker":
        """
        Track models with partial_fit (SGDClassifier, MiniBatchKMeans, etc).

        Args:
            model: sklearn model with partial_fit().
            X_batches, y_batches: Lists of (X_batch, y_batch) per step.
            classes: Class labels (required for first partial_fit of classifiers).
        """
        from sklearn.metrics import accuracy_score, mean_squared_error

        self._model_type = type(model).__name__

        if self.feature_names is None:
            first_X = X_batches[0]
            if hasattr(first_X, "columns"):
                self.feature_names = list(first_X.columns)
            else:
                self.feature_names = [f"feature_{i}" for i in range(first_X.shape[1])]

        is_classifier = hasattr(model, "predict_proba") or (classes is not None)

        for step, (X_batch, y_batch) in enumerate(zip(X_batches, y_batches), 1):
            kwargs = {}
            if classes is not None and step == 1:
                kwargs["classes"] = classes

            model.partial_fit(X_batch, y_batch, **kwargs)

            record = BoostingRoundRecord(round=step)

            # Eval on batch
            pred = model.predict(X_batch)
            if is_classifier:
                record.eval_metrics["train"] = {
                    "accuracy": float(accuracy_score(y_batch, pred))
                }
            else:
                record.eval_metrics["train"] = {
                    "mse": float(mean_squared_error(y_batch, pred))
                }

            # Validation
            if X_val is not None and y_val is not None:
                val_pred = model.predict(X_val)
                if is_classifier:
                    record.eval_metrics["valid"] = {
                        "accuracy": float(accuracy_score(y_val, val_pred))
                    }
                else:
                    record.eval_metrics["valid"] = {
                        "mse": float(mean_squared_error(y_val, val_pred))
                    }

            # Feature importance (if available, e.g. SGDClassifier has coef_)
            if hasattr(model, "feature_importances_"):
                record.feature_importance = {
                    n: float(v)
                    for n, v in zip(self.feature_names, model.feature_importances_)
                }
            elif hasattr(model, "coef_"):
                coef = np.abs(model.coef_).flatten()
                if len(coef) == len(self.feature_names):
                    record.feature_importance = {
                        n: float(v) for n, v in zip(self.feature_names, coef)
                    }

            self.store.add_round(record)

        return self

    # ------------------------------------------------------------------
    # Analysis & reporting (delegates to shared boosting infrastructure)
    # ------------------------------------------------------------------
    def report(self) -> None:
        from flowgrad.diagnostics import generate_boosting_report
        rep = generate_boosting_report(self.store)
        print(rep)

    @property
    def plot(self):
        from flowgrad.viz.plots import BoostingPlotAPI
        return BoostingPlotAPI(self.store)

    @property
    def history(self) -> BoostingStore:
        return self.store

    @property
    def summary(self) -> Dict[str, Any]:
        result = {
            "model_type": self._model_type,
            "total_rounds": self.store.num_rounds,
            "features_tracked": len(self.store.get_all_feature_names()),
        }
        return result

    def __repr__(self):
        return (
            f"SklearnTracker(model={self._model_type}, "
            f"rounds={self.store.num_rounds})"
        )
