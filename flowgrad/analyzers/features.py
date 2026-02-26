"""
Feature Engineering Analyzer — FlowGrad's key differentiator.

Goes beyond static feature importance by analyzing:
  - Feature interactions (synergy between feature pairs)
  - Feature redundancy (near-duplicate detection)
  - Feature clustering (group related features)
  - Automatic feature combination suggestions

Unlike `df.corr()` or `model.feature_importances_`:
  - Detects NON-LINEAR interactions (not just correlation)
  - Measures SYNERGY (A+B together > A alone + B alone)
  - Suggests CONCRETE new features to engineer

Usage:
    from flowgrad.analyzers.features import FeatureAnalyzer

    analyzer = FeatureAnalyzer(model, X_train, y_train, feature_names=X.columns)
    analyzer.interactions(top_k=10)         # Top feature pairs by interaction strength
    analyzer.suggest_features()             # Suggest new engineered features
    analyzer.redundant_features()           # Find near-duplicates
    analyzer.feature_clusters()             # Group related features
    analyzer.plot.interaction_heatmap()      # Visualize
"""
from __future__ import annotations

import warnings
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from sklearn.metrics import mutual_info_score
except ImportError:
    mutual_info_score = None


class FeatureAnalyzer:
    """
    Analyze feature interactions, redundancy, and suggest engineering ideas.

    Args:
        model: A fitted sklearn-compatible model (must have .predict()).
        X: Feature matrix (numpy array or pandas DataFrame).
        y: Target vector.
        feature_names: List of feature names. Auto-detected from DataFrame.
        task: 'classification' or 'regression'. Auto-detected if possible.
    """

    def __init__(
        self,
        model,
        X,
        y,
        feature_names: Optional[List[str]] = None,
        task: Optional[str] = None,
    ):
        self.model = model
        self.X = np.asarray(X)
        self.y = np.asarray(y)

        # Feature names
        if feature_names is not None:
            self.feature_names = list(feature_names)
        elif hasattr(X, "columns"):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f"feature_{i}" for i in range(self.X.shape[1])]

        # Task type
        if task:
            self.task = task
        elif hasattr(model, "predict_proba") or hasattr(model, "classes_"):
            self.task = "classification"
        else:
            self.task = "regression"

        # Cache
        self._base_score: Optional[float] = None
        self._individual_scores: Dict[str, float] = {}
        self._interaction_cache: Dict[Tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # Core: model performance scorer
    # ------------------------------------------------------------------
    def _score(self, X_modified: np.ndarray) -> float:
        """Score model on modified feature matrix."""
        from sklearn.metrics import accuracy_score, mean_squared_error
        pred = self.model.predict(X_modified)
        if self.task == "classification":
            return accuracy_score(self.y, pred)
        else:
            return -mean_squared_error(self.y, pred)  # negative for consistency (higher=better)

    def _get_base_score(self) -> float:
        if self._base_score is None:
            self._base_score = self._score(self.X)
        return self._base_score

    def _permute_feature(self, col_idx: int, seed: int = 42) -> np.ndarray:
        """Create X with column col_idx randomly permuted (breaks its predictive power)."""
        rng = np.random.RandomState(seed)
        X_perm = self.X.copy()
        X_perm[:, col_idx] = rng.permutation(X_perm[:, col_idx])
        return X_perm

    # ------------------------------------------------------------------
    # 1. Feature Interaction Detection
    # ------------------------------------------------------------------
    def interactions(
        self,
        top_k: int = 10,
        method: str = "permutation",
        sample_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect feature interactions by measuring synergy.

        Synergy(A, B) = importance(A, B together) - importance(A alone) - importance(B alone)
        If synergy > 0 → features are more powerful together than individually.

        Args:
            top_k: Number of top interactions to return.
            method: 'permutation' (default) or 'correlation'.
            sample_size: Subsample data for speed (default: min(1000, n_samples)).

        Returns:
            List of {feat_a, feat_b, synergy_score, interaction_strength}
        """
        if method == "correlation":
            return self._interaction_by_correlation(top_k)

        # Subsample for speed
        n = self.X.shape[0]
        if sample_size is None:
            sample_size = min(1000, n)

        if sample_size < n:
            idx = np.random.RandomState(42).choice(n, sample_size, replace=False)
            X_sample = self.X[idx]
            y_sample = self.y[idx]
        else:
            X_sample = self.X
            y_sample = self.y

        base = self._score_with(X_sample, y_sample)
        n_features = len(self.feature_names)

        # Individual importance (permutation-based)
        individual_drop = {}
        for i in range(n_features):
            X_perm = X_sample.copy()
            rng = np.random.RandomState(42)
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            score_perm = self._score_with(X_perm, y_sample)
            individual_drop[i] = base - score_perm  # how much performance drops

        # Pairwise interaction: permute both, compare to sum of individual drops
        results = []
        pairs = list(combinations(range(n_features), 2))

        # Limit pairs for large feature sets
        if len(pairs) > 500:
            # Prioritize top individually important features
            top_features = sorted(individual_drop.keys(),
                                   key=lambda k: individual_drop[k], reverse=True)[:20]
            pairs = list(combinations(top_features, 2))

        for i, j in pairs:
            X_perm = X_sample.copy()
            rng = np.random.RandomState(42)
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            score_both = self._score_with(X_perm, y_sample)

            pair_drop = base - score_both
            synergy = pair_drop - individual_drop[i] - individual_drop[j]

            results.append({
                "feat_a": self.feature_names[i],
                "feat_b": self.feature_names[j],
                "synergy_score": round(synergy, 6),
                "interaction_strength": round(abs(synergy), 6),
                "individual_a": round(individual_drop[i], 6),
                "individual_b": round(individual_drop[j], 6),
                "combined_drop": round(pair_drop, 6),
            })

        results.sort(key=lambda x: x["interaction_strength"], reverse=True)
        return results[:top_k]

    def _score_with(self, X: np.ndarray, y: np.ndarray) -> float:
        """Score model with given X, y."""
        from sklearn.metrics import accuracy_score, mean_squared_error
        pred = self.model.predict(X)
        if self.task == "classification":
            return accuracy_score(y, pred)
        else:
            return -mean_squared_error(y, pred)

    def _interaction_by_correlation(self, top_k: int) -> List[Dict]:
        """Fast interaction detection using mutual information / correlation."""
        n_features = len(self.feature_names)
        results = []

        for i, j in combinations(range(n_features), 2):
            # Pearson correlation
            corr = np.corrcoef(self.X[:, i], self.X[:, j])[0, 1]

            # Residual correlation with target
            corr_i_y = np.corrcoef(self.X[:, i], self.y)[0, 1] if np.std(self.X[:, i]) > 0 else 0
            corr_j_y = np.corrcoef(self.X[:, j], self.y)[0, 1] if np.std(self.X[:, j]) > 0 else 0

            # Product feature correlation with target
            product = self.X[:, i] * self.X[:, j]
            if np.std(product) > 0:
                corr_prod_y = np.corrcoef(product, self.y)[0, 1]
            else:
                corr_prod_y = 0

            # Synergy: does the product capture more than individuals?
            synergy = abs(corr_prod_y) - max(abs(corr_i_y), abs(corr_j_y))

            results.append({
                "feat_a": self.feature_names[i],
                "feat_b": self.feature_names[j],
                "synergy_score": round(float(synergy), 6),
                "interaction_strength": round(abs(float(synergy)), 6),
                "pair_correlation": round(float(corr), 4),
                "product_target_corr": round(float(corr_prod_y), 4),
            })

        results.sort(key=lambda x: x["interaction_strength"], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # 2. Feature Combination Suggestions
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_vif(X: np.ndarray, col_idx: int) -> float:
        """Compute Variance Inflation Factor for a single column."""
        y_col = X[:, col_idx]
        other_cols = np.delete(X, col_idx, axis=1)

        if other_cols.shape[1] == 0 or np.std(y_col) < 1e-10:
            return 1.0

        # VIF = 1 / (1 - R²)
        try:
            # Fast OLS via normal equation
            X_aug = np.column_stack([np.ones(len(other_cols)), other_cols])
            beta = np.linalg.lstsq(X_aug, y_col, rcond=None)[0]
            y_hat = X_aug @ beta
            ss_res = np.sum((y_col - y_hat) ** 2)
            ss_tot = np.sum((y_col - y_col.mean()) ** 2)
            r_squared = 1 - ss_res / max(ss_tot, 1e-10)
            r_squared = min(r_squared, 0.9999)  # cap to avoid inf
            return 1.0 / (1.0 - r_squared)
        except Exception:
            return 1.0

    def suggest_features(
        self,
        top_k: int = 10,
        operations: Optional[List[str]] = None,
        sample_size: Optional[int] = None,
        collinearity_check: bool = True,
        vif_threshold: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """
        Suggest new feature combinations that could improve the model.

        Tests A*B, A/B, A-B, A+B and scores each against the target.
        Optionally checks collinearity (VIF) to flag redundant combinations.

        Args:
            top_k: Number of suggestions to return.
            operations: List of operations to try. Default: ['multiply', 'divide', 'subtract', 'add'].
            sample_size: Subsample for speed.
            collinearity_check: If True, compute VIF for each suggestion.
            vif_threshold: VIF above this → collinearity warning.

        Returns:
            List of dicts with expression, lift, vif_score, collinearity_warning.
        """
        if operations is None:
            operations = ["multiply", "divide", "subtract", "add"]

        n_features = len(self.feature_names)

        # Subsample
        n = self.X.shape[0]
        if sample_size is None:
            sample_size = min(2000, n)
        if sample_size < n:
            idx = np.random.RandomState(42).choice(n, sample_size, replace=False)
            X_s = self.X[idx]
            y_s = self.y[idx]
        else:
            X_s = self.X
            y_s = self.y

        # Baseline: max individual correlation
        individual_corrs = {}
        for i in range(n_features):
            if np.std(X_s[:, i]) > 1e-10:
                individual_corrs[i] = abs(np.corrcoef(X_s[:, i], y_s)[0, 1])
            else:
                individual_corrs[i] = 0.0

        results = []
        pairs = list(combinations(range(n_features), 2))

        # Limit for large feature sets
        if len(pairs) > 1000:
            top_feats = sorted(individual_corrs, key=lambda k: individual_corrs[k], reverse=True)[:15]
            pairs = list(combinations(top_feats, 2))

        for i, j in pairs:
            a = X_s[:, i]
            b = X_s[:, j]
            baseline = max(individual_corrs.get(i, 0), individual_corrs.get(j, 0))

            combos = {}
            if "multiply" in operations:
                combos[f"{self.feature_names[i]} * {self.feature_names[j]}"] = a * b
            if "divide" in operations:
                safe_b = np.where(np.abs(b) > 1e-10, b, 1e-10)
                combos[f"{self.feature_names[i]} / {self.feature_names[j]}"] = a / safe_b
            if "subtract" in operations:
                combos[f"{self.feature_names[i]} - {self.feature_names[j]}"] = a - b
            if "add" in operations:
                combos[f"{self.feature_names[i]} + {self.feature_names[j]}"] = a + b

            for expr, values in combos.items():
                if np.std(values) < 1e-10:
                    continue
                # Handle NaN/inf
                mask = np.isfinite(values)
                if mask.sum() < 30: # Need enough degrees of freedom
                    continue

                from scipy import stats
                v_mask = values[mask]
                y_mask = y_s[mask]
                
                # Check absolute correlation significance
                r, p_val = stats.pearsonr(v_mask, y_mask)
                corr = abs(r)
                if np.isnan(corr) or p_val > 0.01: # Must be strongly significant
                    continue

                lift = corr - baseline
                
                # To guarantee the F-test passes later, the partial correlation 
                # (controlling for original features) must be significant.
                # We do a fast t-test on the new coefficient.
                try:
                    X_base = np.column_stack([np.ones_like(y_mask), a[mask], b[mask], v_mask])
                    beta, res, rank, s = np.linalg.lstsq(X_base, y_mask, rcond=None)
                    
                    # Compute standard errors for beta
                    mse = res[0] / (len(y_mask) - 4) if len(res) > 0 else np.var(y_mask)
                    cov_matrix = np.linalg.inv(X_base.T @ X_base) * mse
                    se = np.sqrt(np.diag(cov_matrix))
                    
                    # t-statistic for the last coefficient (the new feature)
                    t_stat = beta[-1] / (se[-1] + 1e-10)
                    p_val_partial = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(y_mask)-4))
                    
                    if p_val_partial > 0.05: # The engineered feature doesn't provide significant unique variance
                        continue
                except Exception:
                    continue # Skip if matrix is singular

                # VIF check: does this new feature add collinearity?
                vif_score = None
                collinearity_warning = False
                if collinearity_check and lift > 0:
                    X_augmented = np.column_stack([X_s[mask], v_mask.reshape(-1, 1)])
                    new_col_idx = X_augmented.shape[1] - 1
                    vif_score = self._compute_vif(X_augmented, new_col_idx)
                    collinearity_warning = vif_score > vif_threshold

                op_name = expr.split(" ")[1]  # *, /, -, +
                entry = {
                    "expression": expr,
                    "operation": op_name,
                    "feat_a": self.feature_names[i],
                    "feat_b": self.feature_names[j],
                    "target_correlation": round(float(corr), 4),
                    "baseline_correlation": round(float(baseline), 4),
                    "lift": round(float(lift), 4),
                    "p_value": float(p_val_partial),
                }
                if collinearity_check:
                    entry["vif_score"] = round(float(vif_score), 2) if vif_score is not None else None
                    entry["collinearity_warning"] = collinearity_warning

                results.append(entry)

        # Sort by lift (how much better than individual features)
        results.sort(key=lambda x: x["lift"], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # 3. Redundant Feature Detection
    # ------------------------------------------------------------------
    def redundant_features(
        self,
        threshold: float = 0.95,
    ) -> List[Dict[str, Any]]:
        """
        Find pairs of features that are nearly identical (redundant).

        Uses Pearson correlation to detect near-duplicates.

        Args:
            threshold: Correlation threshold above which features are considered redundant.

        Returns:
            List of {feat_a, feat_b, correlation, recommendation}
        """
        n_features = len(self.feature_names)
        results = []

        for i, j in combinations(range(n_features), 2):
            if np.std(self.X[:, i]) < 1e-10 or np.std(self.X[:, j]) < 1e-10:
                continue

            corr = np.corrcoef(self.X[:, i], self.X[:, j])[0, 1]
            if abs(corr) >= threshold:
                # Decide which to keep: the one with higher target correlation
                corr_i = abs(np.corrcoef(self.X[:, i], self.y)[0, 1])
                corr_j = abs(np.corrcoef(self.X[:, j], self.y)[0, 1])

                keep = self.feature_names[i] if corr_i >= corr_j else self.feature_names[j]
                drop = self.feature_names[j] if corr_i >= corr_j else self.feature_names[i]

                results.append({
                    "feat_a": self.feature_names[i],
                    "feat_b": self.feature_names[j],
                    "correlation": round(float(corr), 4),
                    "recommendation": f"Drop '{drop}', keep '{keep}'",
                    "keep": keep,
                    "drop": drop,
                })

        results.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return results

    # ------------------------------------------------------------------
    # 4. Feature Clusters
    # ------------------------------------------------------------------
    def feature_clusters(
        self,
        n_clusters: Optional[int] = None,
        method: str = "correlation",
    ) -> List[Dict[str, Any]]:
        """
        Group features into clusters based on similarity.

        Helps identify feature groups for dimensionality reduction or feature selection.

        Args:
            n_clusters: Number of clusters. Auto-determined if None.
            method: 'correlation' (default).

        Returns:
            List of {cluster_id, features, cohesion_score}
        """
        from sklearn.cluster import AgglomerativeClustering

        n_features = len(self.feature_names)
        if n_features < 3:
            return [{"cluster_id": 0, "features": self.feature_names, "cohesion_score": 1.0}]

        # Correlation-based distance matrix
        corr_matrix = np.corrcoef(self.X.T)
        np.fill_diagonal(corr_matrix, 1.0)
        # Handle NaN
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        distance_matrix = 1 - np.abs(corr_matrix)
        distance_matrix = np.clip(distance_matrix, 0, 2)

        if n_clusters is None:
            n_clusters = max(2, min(n_features // 3, 8))

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(distance_matrix)

        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.feature_names[idx])

        results = []
        for cluster_id, features in sorted(clusters.items()):
            # Cohesion: average within-cluster correlation
            if len(features) > 1:
                feat_indices = [self.feature_names.index(f) for f in features]
                sub_corr = corr_matrix[np.ix_(feat_indices, feat_indices)]
                mask = np.triu(np.ones_like(sub_corr, dtype=bool), k=1)
                cohesion = float(np.abs(sub_corr[mask]).mean())
            else:
                cohesion = 1.0

            results.append({
                "cluster_id": int(cluster_id),
                "features": features,
                "cohesion_score": round(cohesion, 4),
                "size": len(features),
            })

        results.sort(key=lambda x: x["cluster_id"])
        return results

    # ------------------------------------------------------------------
    # 5. Full Report
    # ------------------------------------------------------------------
    def report(
        self,
        top_interactions: int = 10,
        top_suggestions: int = 10,
        redundancy_threshold: float = 0.95,
    ) -> None:
        """Generate a comprehensive feature engineering report."""
        lines = []
        lines.append("=" * 60)
        lines.append("  FlowGrad — Feature Engineering Report")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"📊 Features analyzed: {len(self.feature_names)}")
        lines.append(f"📐 Samples: {self.X.shape[0]}")
        lines.append(f"🎯 Task: {self.task}")
        lines.append("")

        # Feature Interactions
        lines.append("─" * 60)
        lines.append("🔗 Feature Interactions (Top Synergies)")
        lines.append("─" * 60)
        try:
            interactions = self.interactions(top_k=top_interactions, method="correlation")
            if interactions:
                for i, item in enumerate(interactions, 1):
                    emoji = "🔥" if item["synergy_score"] > 0.05 else "📊"
                    lines.append(
                        f"  {emoji} {i}. {item['feat_a']} × {item['feat_b']}  "
                        f"synergy={item['synergy_score']:+.4f}"
                    )
            else:
                lines.append("  No significant interactions detected.")
        except Exception as e:
            lines.append(f"  ⚠️ Could not compute interactions: {e}")
        lines.append("")

        # Feature Suggestions
        lines.append("─" * 60)
        lines.append("💡 Suggested Feature Combinations")
        lines.append("─" * 60)
        try:
            suggestions = self.suggest_features(top_k=top_suggestions)
            positive = [s for s in suggestions if s["lift"] > 0.01]
            if positive:
                for i, item in enumerate(positive, 1):
                    lines.append(
                        f"  {i}. CREATE: {item['expression']}"
                    )
                    lines.append(
                        f"     target_corr={item['target_correlation']:.4f}  "
                        f"lift=+{item['lift']:.4f}  (p={item['p_value']:.2e})"
                    )
            else:
                lines.append("  No significant improvements found via simple combinations.")
                lines.append("  Consider polynomial features or domain-specific transformations.")
        except Exception as e:
            lines.append(f"  ⚠️ Could not generate suggestions: {e}")
        lines.append("")

        # Redundant features
        lines.append("─" * 60)
        lines.append("🗑️ Redundant Features")
        lines.append("─" * 60)
        try:
            redundant = self.redundant_features(threshold=redundancy_threshold)
            if redundant:
                for item in redundant:
                    lines.append(
                        f"  ⚠️ {item['feat_a']} ↔ {item['feat_b']}  "
                        f"corr={item['correlation']:.4f}"
                    )
                    lines.append(f"     💊 {item['recommendation']}")
            else:
                lines.append(f"  ✅ No redundant feature pairs (threshold={redundancy_threshold}).")
        except Exception as e:
            lines.append(f"  ⚠️ Could not check redundancy: {e}")
        lines.append("")

        # Feature clusters
        lines.append("─" * 60)
        lines.append("📦 Feature Clusters")
        lines.append("─" * 60)
        try:
            clusters = self.feature_clusters()
            for cluster in clusters:
                lines.append(
                    f"  Cluster {cluster['cluster_id']}: "
                    f"({cluster['size']} features, cohesion={cluster['cohesion_score']:.2f})"
                )
                for f in cluster["features"]:
                    lines.append(f"    • {f}")
        except Exception as e:
            lines.append(f"  ⚠️ Could not cluster features: {e}")
        lines.append("")

        lines.append("=" * 60)
        report = "\n".join(lines)
        print(report)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    @property
    def plot(self):
        return FeaturePlotAPI(self)


class FeaturePlotAPI:
    """Visualization for feature analysis."""

    def __init__(self, analyzer: FeatureAnalyzer):
        self.analyzer = analyzer

    def interaction_heatmap(self, top_k: int = 15, figsize=(10, 8)):
        """Heatmap of feature interaction strengths."""
        import matplotlib.pyplot as plt

        from flowgrad.viz.plots import PALETTE, _apply_style

        interactions = self.analyzer.interactions(top_k=top_k * 3, method="correlation")

        # Build matrix
        feats = list(set(
            [it["feat_a"] for it in interactions] + [it["feat_b"] for it in interactions]
        ))[:top_k]
        n = len(feats)
        matrix = np.zeros((n, n))
        feat_idx = {f: i for i, f in enumerate(feats)}

        for it in interactions:
            a, b = it["feat_a"], it["feat_b"]
            if a in feat_idx and b in feat_idx:
                matrix[feat_idx[a], feat_idx[b]] = it["synergy_score"]
                matrix[feat_idx[b], feat_idx[a]] = it["synergy_score"]

        fig, ax = plt.subplots(figsize=figsize)
        _apply_style(fig, ax)

        im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto",
                       vmin=-max(abs(matrix.min()), abs(matrix.max())),
                       vmax=max(abs(matrix.min()), abs(matrix.max())))
        cbar = fig.colorbar(im, ax=ax, label="Synergy Score")
        cbar.ax.yaxis.label.set_color(PALETTE["text"])
        cbar.ax.tick_params(colors=PALETTE["text"])

        ax.set_xticks(range(n))
        ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(feats, fontsize=7)
        ax.set_title("Feature Interaction Heatmap")

        fig.tight_layout()
        return fig

    def suggestion_chart(self, top_k: int = 10, figsize=(10, 5)):
        """Bar chart of top feature combination suggestions by lift."""
        import matplotlib.pyplot as plt
        from flowgrad.viz.plots import PALETTE, _apply_style

        suggestions = self.analyzer.suggest_features(top_k=top_k)
        positive = [s for s in suggestions if s["lift"] > 0]

        if not positive:
            fig, ax = plt.subplots(figsize=figsize)
            _apply_style(fig, ax)
            ax.text(0.5, 0.5, "No positive-lift combinations found",
                    ha="center", va="center", color=PALETTE["text"])
            return fig

        labels = [s["expression"][:30] for s in positive]
        lifts = [s["lift"] for s in positive]
        corrs = [s["target_correlation"] for s in positive]

        fig, ax = plt.subplots(figsize=figsize)
        _apply_style(fig, ax)

        bars = ax.barh(range(len(labels)), lifts, color=PALETTE["success"], alpha=0.85)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Lift (improvement over individual features)")
        ax.set_title("Top Feature Combination Suggestions")

        for i, (l, c) in enumerate(zip(lifts, corrs)):
            ax.text(l + 0.001, i, f"corr={c:.3f}", va="center",
                    fontsize=7, color=PALETTE["text"])

        fig.tight_layout()
        return fig

    def redundancy_graph(self, threshold: float = 0.95, figsize=(10, 8)):
        """Network-style visualization of redundant feature pairs."""
        import matplotlib.pyplot as plt
        from flowgrad.viz.plots import PALETTE, _apply_style

        redundant = self.analyzer.redundant_features(threshold=threshold)

        if not redundant:
            fig, ax = plt.subplots(figsize=figsize)
            _apply_style(fig, ax)
            ax.text(0.5, 0.5, "No redundant features detected",
                    ha="center", va="center", color=PALETTE["text"])
            return fig

        # Simple scatter-based network
        all_feats = list(set(
            [r["feat_a"] for r in redundant] + [r["feat_b"] for r in redundant]
        ))
        n = len(all_feats)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        positions = {f: (np.cos(a), np.sin(a)) for f, a in zip(all_feats, angles)}

        fig, ax = plt.subplots(figsize=figsize)
        _apply_style(fig, ax)

        # Draw edges
        for r in redundant:
            p1 = positions[r["feat_a"]]
            p2 = positions[r["feat_b"]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=PALETTE["danger"], alpha=0.6, linewidth=2)
            mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            ax.annotate(f"{r['correlation']:.2f}", mid, fontsize=6,
                        color=PALETTE["warning"], ha="center")

        # Draw nodes
        for feat, (x, y) in positions.items():
            ax.scatter(x, y, s=200, color=PALETTE["primary"], zorder=5)
            ax.annotate(feat, (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=7,
                        color=PALETTE["text"])

        ax.set_title("Feature Redundancy Network")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

        fig.tight_layout()
        return fig

    def cluster_map(self, figsize=(12, 5)):
        """Visualization of feature clusters."""
        import matplotlib.pyplot as plt
        from flowgrad.viz.plots import PALETTE, _apply_style

        clusters = self.analyzer.feature_clusters()

        fig, ax = plt.subplots(figsize=figsize)
        _apply_style(fig, ax)

        colors_list = [
            PALETTE["primary"], PALETTE["secondary"], PALETTE["success"],
            PALETTE["warning"], PALETTE["info"], PALETTE["danger"],
            "#A855F7", "#06B6D4",
        ]

        y_pos = 0
        labels = []
        y_positions = []
        bar_colors = []

        for cluster in clusters:
            color = colors_list[cluster["cluster_id"] % len(colors_list)]
            for feat in cluster["features"]:
                labels.append(feat)
                y_positions.append(y_pos)
                bar_colors.append(color)
                y_pos += 1

        ax.barh(y_positions, [1] * len(labels), color=bar_colors, alpha=0.7)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("")
        ax.set_title("Feature Clusters")

        # Add cluster labels
        y_start = 0
        for cluster in clusters:
            y_end = y_start + len(cluster["features"])
            mid = (y_start + y_end - 1) / 2
            ax.text(1.05, mid, f"Cluster {cluster['cluster_id']}\ncohesion={cluster['cohesion_score']:.2f}",
                    va="center", fontsize=7, color=PALETTE["text"])
            y_start = y_end

        fig.tight_layout()
        return fig
