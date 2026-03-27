"""
TreeTracker — GBDT (XGBoost, LightGBM, CatBoost) Dynamics Analyzer.

Analyzes internal tree structures, leaf-node influence, and feature split 
concentration to diagnose model health and recommend improvements.
"""
from __future__ import annotations

import json
import numpy as np
import os
import tempfile
from typing import Any, Dict, List, Optional
from collections import Counter

class TreeTracker:
    """
    Expert GBDT analyzer for tree-based models.
    Supports XGBoost, LightGBM, and CatBoost.
    """
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names
        self.model_type = self._detect_model_type(model)
        self.booster = self._get_booster(model)
        
        if self.feature_names is None:
            self.feature_names = self._detect_feature_names()

        # Cached analysis
        self._dynamics: Optional[Dict[str, Any]] = None
        self._interactions: Optional[List[tuple]] = None

    def _detect_model_type(self, model: Any) -> str:
        name = type(model).__name__
        if name == 'Booster' and hasattr(model, 'get_dump'):
            return 'xgboost'
        elif hasattr(model, 'get_booster'):
            return 'xgboost'
        elif name in ['LGBMRegressor', 'LGBMClassifier', 'Booster'] and hasattr(model, 'dump_model'):
            return 'lightgbm'
        elif name in ['CatBoost', 'CatBoostRegressor', 'CatBoostClassifier']:
            return 'catboost'
        raise ValueError(f"Model type {name} is not supported. Use XGBoost, LightGBM, or CatBoost.")

    def _get_booster(self, model: Any) -> Any:
        if self.model_type == 'xgboost' and hasattr(model, 'get_booster'):
            return model.get_booster()
        return model

    def _detect_feature_names(self) -> Optional[List[str]]:
        if self.model_type == 'xgboost':
            if hasattr(self.booster, 'feature_names'):
                return self.booster.feature_names
        elif self.model_type == 'lightgbm':
            if hasattr(self.booster, 'feature_name'):
                return self.booster.feature_name()
        return None

    def analyze(self) -> Dict[str, Any]:
        """Runs the full structural analysis of the forest."""
        if self._dynamics is not None:
            return self._dynamics

        leaf_values = []
        split_features = []
        depths = []
        interactions = [] # (parent_feat, child_feat)
        num_trees = 0
        
        if self.model_type == 'xgboost':
            dump = self.booster.get_dump(dump_format='json')
            trees = [json.loads(d) for d in dump]
            num_trees = len(trees)
            
            def traverse_xgb(node, current_depth, parent_feat=None):
                depths.append(current_depth)
                if 'leaf' in node:
                    leaf_values.append(node['leaf'])
                else:
                    feat = node.get('split')
                    if feat:
                        split_features.append(feat)
                        if parent_feat:
                            interactions.append((parent_feat, feat))
                    for child in node.get('children', []):
                        traverse_xgb(child, current_depth + 1, feat)
                        
            for tree in trees:
                traverse_xgb(tree, 0)
                
        elif self.model_type == 'lightgbm':
            dump = self.booster.dump_model()
            trees = dump.get('tree_info', [])
            num_trees = len(trees)
            
            def traverse_lgb(node, current_depth, parent_feat=None):
                depths.append(current_depth)
                if 'leaf_value' in node:
                    leaf_values.append(node['leaf_value'])
                else:
                    if 'split_feature' in node:
                        feat_idx = node['split_feature']
                        feat = str(feat_idx)
                        if self.feature_names and feat_idx < len(self.feature_names):
                            feat = self.feature_names[feat_idx]
                        split_features.append(feat)
                        if parent_feat:
                            interactions.append((parent_feat, feat))
                        
                        if 'left_child' in node:
                            traverse_lgb(node['left_child'], current_depth + 1, feat)
                        if 'right_child' in node:
                            traverse_lgb(node['right_child'], current_depth + 1, feat)
                        
            for tree in trees:
                traverse_lgb(tree.get('tree_structure', {}), 0)
                
        elif self.model_type == 'catboost':
            # Simplified CatBoost handling
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
                tmp_name = tmp.name
            try:
                self.booster.save_model(tmp_name, format="json")
                with open(tmp_name, 'r') as f:
                    dump = json.load(f)
                
                if 'oblivious_trees' in dump:
                    trees = dump['oblivious_trees']
                    num_trees = len(trees)
                    for tree in trees:
                        if 'leaf_values' in tree:
                            vals = tree['leaf_values']
                            if isinstance(vals[0], list):
                                for v in vals: leaf_values.extend(v)
                            else:
                                leaf_values.extend(vals)
                        if 'splits' in tree:
                            depths.append(len(tree['splits']))
                            for split in tree['splits']:
                                if 'float_feature_index' in split:
                                    idx = split['float_feature_index']
                                    feat = self.feature_names[idx] if self.feature_names and idx < len(self.feature_names) else str(idx)
                                    split_features.append(feat)
            finally:
                if os.path.exists(tmp_name):
                    os.remove(tmp_name)
        
        leaf_variance = np.var(leaf_values) if leaf_values else 0.0
        max_depth = max(depths) if depths else 0
        feat_counts = Counter(split_features)
        
        # Saliency of interactions
        inter_counts = Counter(interactions)
        sorted_inter = sorted(inter_counts.items(), key=lambda x: x[1], reverse=True)
        self._interactions = [(f1, f2, count/num_trees if num_trees > 0 else 0) for (f1, f2), count in sorted_inter]

        self._dynamics = {
            "num_trees": num_trees,
            "total_leaves": len(leaf_values),
            "max_depth": max_depth,
            "leaf_variance": float(leaf_variance),
            "top_splits": dict(feat_counts.most_common(5)),
            "leaf_weights": {
                "mean": float(np.mean(leaf_values)) if leaf_values else 0,
                "max": float(np.max(np.abs(leaf_values))) if leaf_values else 0,
            }
        }
        return self._dynamics

    def leaf_influence(self) -> Dict[str, float]:
        """Calculates leaf-based complexity metrics."""
        stats = self.analyze()
        return {
            "mean_weight": stats["leaf_weights"]["mean"],
            "max_weight": stats["leaf_weights"]["max"],
            "variance": stats["leaf_variance"],
            "overfit_risk": min(1.0, stats["leaf_variance"] * 2.0) # Heuristic
        }

    def interaction_saliency(self) -> List[tuple]:
        """Identifies strong feature interactions based on parent-child splits."""
        self.analyze()
        return self._interactions or []

    def report(self) -> None:
        """Prints a comprehensive GBDT health and interpretability report."""
        stats = self.analyze()
        influ = self.leaf_influence()
        inter = self.interaction_saliency()
        
        # Use simple survivors in case of encoding issues
        try:
            tree_icon = "🌳"
            alert_icon = "🚨"
            check_icon = "✅"
            bullet = "•"
        except UnicodeEncodeError:
            tree_icon = "#"
            alert_icon = "!"
            check_icon = "OK"
            bullet = "-"

        def safe_print(line):
            try:
                print(line)
            except UnicodeEncodeError:
                # Fallback: remove emojis
                print(line.replace("🌳", "#").replace("🚨", "!").replace("✅", "OK").replace("•", "-"))

        header = f"\n{tree_icon * 20}"
        safe_print(header)
        safe_print(f" GradTracer: {self.model_type.upper()} Structural Intelligence")
        safe_print(header)
        safe_print(f" [1] Topology: {stats['num_trees']} trees, {stats['total_leaves']} leaves, Max Depth {stats['max_depth']}")
        safe_print(f" [2] Leaf Dynamics: Influence={influ['mean_weight']:.3f}, Var={stats['leaf_variance']:.5f}")
        safe_print(f" [3] Overfit Risk: {influ['overfit_risk']*100:.1f}%")
        
        if stats['leaf_variance'] < 1e-4:
             safe_print("     🐌 ALERT: Stagnant leaves detected. Consider increasing learning rate.")
        elif stats['leaf_variance'] > 1.0:
             safe_print("     🧨 ALERT: Exploding leaves detected. Check regularization (lambda/alpha).")
        
        safe_print("\n [4] Top Feature Splits (Concentration):")
        for f, c in stats['top_splits'].items():
            safe_print(f"  {bullet} {f}: {c} times")
            
        if inter:
            safe_print("\n [5] Top Recommended Cross-Features (Parent-Child pairs):")
            for f1, f2, score in inter[:3]:
                safe_print(f"  {bullet} {f1} x {f2} (Interaction Strength: {score:.2f})")
        
        safe_print("="*40 + "\n")

class FeatureAdvisor:
    """Scientific Feature Engineering Advisor using TreeTracker insights."""
    
    @staticmethod
    def recommend_interactions(model: Any, feature_names: List[str] = None, top_k: int = 5) -> List[tuple]:
        """Suggests feature interactions based on tree routing patterns."""
        tracker = TreeTracker(model, feature_names)
        raw_inter = tracker.interaction_saliency()
        return raw_inter[:top_k]
        
    @staticmethod
    def detect_leakage(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[str]:
        """Identifies features that might be leaking target information."""
        leaks = []
        for i, name in enumerate(feature_names):
            # Simple correlation check for now
            try:
                corr = np.corrcoef(X[:, i].astype(float), y.astype(float))[0, 1]
                if abs(corr) > 0.98:
                    leaks.append(f"{name} (Correlation: {corr:.4f})")
            except Exception:
                pass
        return leaks
