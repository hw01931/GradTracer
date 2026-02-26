"""
BoostingTracker — Training dynamics tracker for XGBoost, LightGBM, and CatBoost.

Provides unified callback adapters for all three boosting frameworks.
Tracks per-round feature importance shifts and evaluation metrics.

Usage:
    tracker = BoostingTracker()

    # XGBoost
    model = xgb.train(params, dtrain, callbacks=[tracker.as_xgb_callback()])

    # LightGBM
    model = lgb.train(params, dtrain, callbacks=[tracker.as_lgb_callback()])

    # CatBoost
    model = CatBoostClassifier(**params)
    model.fit(X, y, callbacks=[tracker.as_catboost_callback()])

    tracker.report()
    tracker.plot.feature_drift()
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from flowgrad.snapshot import BoostingRoundRecord, BoostingStore


class BoostingTracker:
    """
    Unified boosting model training tracker.

    Records feature importance and eval metrics at each boosting round
    for XGBoost, LightGBM, and CatBoost.

    Args:
        track_every: Record every N rounds (default 1 = every round).
        feature_names: Optional list of feature names. Auto-detected if possible.
    """

    def __init__(
        self,
        track_every: int = 1,
        feature_names: Optional[List[str]] = None,
    ):
        self.track_every = track_every
        self.feature_names = feature_names
        self.store = BoostingStore()
        self._framework: Optional[str] = None

    # ------------------------------------------------------------------
    # XGBoost callback
    # ------------------------------------------------------------------
    def as_xgb_callback(self):
        """
        Return an XGBoost TrainingCallback instance.

        Usage:
            model = xgb.train(params, dtrain, callbacks=[tracker.as_xgb_callback()])
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "XGBoost is required. Install: pip install flowgrad[xgboost]"
            )

        tracker = self
        tracker._framework = "xgboost"

        class FlowGradXGBCallback(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                if (epoch + 1) % tracker.track_every != 0:
                    return False  # continue training

                record = BoostingRoundRecord(round=epoch + 1)

                # Eval metrics
                for dataset_name, metrics in evals_log.items():
                    record.eval_metrics[dataset_name] = {}
                    for metric_name, values in metrics.items():
                        if isinstance(values, list) and values:
                            record.eval_metrics[dataset_name][metric_name] = values[-1]
                        else:
                            record.eval_metrics[dataset_name][metric_name] = float(values)

                # Feature importance
                try:
                    importance = model.get_score(importance_type="gain")
                    if importance:
                        record.feature_importance = {
                            k: float(v) for k, v in importance.items()
                        }
                except Exception:
                    pass

                tracker.store.add_round(record)
                return False  # continue training

        return FlowGradXGBCallback()

    # ------------------------------------------------------------------
    # LightGBM callback
    # ------------------------------------------------------------------
    def as_lgb_callback(self):
        """
        Return a LightGBM callback function.

        Usage:
            model = lgb.train(params, dtrain, callbacks=[tracker.as_lgb_callback()])
        """
        tracker = self
        tracker._framework = "lightgbm"

        def _lgb_callback(env):
            iteration = env.iteration + 1

            if iteration % tracker.track_every != 0:
                return

            record = BoostingRoundRecord(round=iteration)

            # Eval metrics
            if env.evaluation_result_list:
                for item in env.evaluation_result_list:
                    # LightGBM format: (dataset_name, metric_name, value, is_higher_better)
                    if len(item) >= 3:
                        ds_name, metric_name, value = item[0], item[1], item[2]
                        if ds_name not in record.eval_metrics:
                            record.eval_metrics[ds_name] = {}
                        record.eval_metrics[ds_name][metric_name] = float(value)

            # Feature importance
            try:
                model = env.model
                importance = model.feature_importance(importance_type="gain")
                names = model.feature_name()
                if importance is not None and names:
                    record.feature_importance = {
                        name: float(imp) for name, imp in zip(names, importance)
                    }
            except Exception:
                pass

            tracker.store.add_round(record)

        _lgb_callback.order = 100  # Run after built-in callbacks
        return _lgb_callback

    # ------------------------------------------------------------------
    # CatBoost callback
    # ------------------------------------------------------------------
    def as_catboost_callback(self):
        """
        Return a CatBoost callback instance.

        Usage:
            model = CatBoostClassifier(**params)
            model.fit(X, y, callbacks=[tracker.as_catboost_callback()])
        """
        tracker = self
        tracker._framework = "catboost"

        class FlowGradCatBoostCallback:
            def after_iteration(self, info):
                iteration = info.iteration + 1

                if iteration % tracker.track_every != 0:
                    return True  # continue training

                record = BoostingRoundRecord(round=iteration)

                # Eval metrics
                if hasattr(info, "metrics") and info.metrics:
                    for ds_name, metrics in info.metrics.items():
                        record.eval_metrics[ds_name] = {}
                        for metric_name, value in metrics.items():
                            if isinstance(value, (list, tuple)):
                                record.eval_metrics[ds_name][metric_name] = float(value[-1])
                            else:
                                record.eval_metrics[ds_name][metric_name] = float(value)

                tracker.store.add_round(record)
                return True  # continue training

        return FlowGradCatBoostCallback()

    # ------------------------------------------------------------------
    # Manual step (framework-agnostic)
    # ------------------------------------------------------------------
    def step(
        self,
        round_num: int,
        eval_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        feature_importance: Optional[Dict[str, float]] = None,
    ):
        """
        Manually record a boosting round (for unsupported frameworks).

        Args:
            round_num: Current boosting round number.
            eval_metrics: {dataset: {metric: value}}
            feature_importance: {feature: importance}
        """
        record = BoostingRoundRecord(
            round=round_num,
            eval_metrics=eval_metrics or {},
            feature_importance=feature_importance or {},
        )
        self.store.add_round(record)

    # ------------------------------------------------------------------
    # Analysis & reporting
    # ------------------------------------------------------------------
    def report(self) -> None:
        """Generate and print a text diagnostic report."""
        from flowgrad.diagnostics import generate_boosting_report
        rep = generate_boosting_report(self.store)
        print(rep)

    @property
    def plot(self):
        """Access the boosting visualization API."""
        from flowgrad.viz.plots import BoostingPlotAPI
        return BoostingPlotAPI(self.store)

    @property
    def history(self) -> BoostingStore:
        """Access raw collected data."""
        return self.store

    @property
    def summary(self) -> Dict[str, Any]:
        """Quick summary for programmatic access."""
        result = {
            "framework": self._framework,
            "total_rounds": self.store.num_rounds,
            "features_tracked": len(self.store.get_all_feature_names()),
        }

        # Feature importance drift: top movers
        feature_names = self.store.get_all_feature_names()
        if feature_names and self.store.num_rounds >= 2:
            drifts = {}
            for feat in feature_names:
                series = self.store.get_feature_importance_series(feat)
                if len(series) >= 2:
                    first_half = sum(series[: len(series) // 2]) / max(len(series) // 2, 1)
                    second_half = sum(series[len(series) // 2 :]) / max(len(series) - len(series) // 2, 1)
                    if first_half > 0:
                        drifts[feat] = (second_half - first_half) / first_half
                    else:
                        drifts[feat] = second_half
            result["top_rising_features"] = sorted(drifts.items(), key=lambda x: -x[1])[:5]
            result["top_declining_features"] = sorted(drifts.items(), key=lambda x: x[1])[:5]

        return result

    def __repr__(self):
        return (
            f"BoostingTracker(framework={self._framework}, "
            f"rounds={self.store.num_rounds})"
        )
