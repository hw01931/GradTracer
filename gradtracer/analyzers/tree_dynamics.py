"""
TreeDynamicsTracker — Deep GBDT Node-Level Diagnostic Tracking for XGBoost.

Instead of just looking at standard feature importance, it unpacks the raw tree 
structures to mathematically compute:
  - Leaf Velocity (Variance)
  - Feature Split Concentration
"""
from __future__ import annotations

import json
import numpy as np
from typing import Dict, Any, List, Optional
from collections import Counter


class TreeDynamicsTracker:
    """
    Analyzes internal tree structures of Gradient Boosting models.
    Supports XGBoost, LightGBM, and CatBoost (via their JSON/dict dumps).
    """

    def __init__(self, model, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names
        
        # Determine Model Type
        if type(model).__name__ == 'Booster' and hasattr(model, 'get_dump'):
            # XGBoost raw Booster
            self.booster = model
            self.model_type = 'xgboost'
        elif hasattr(model, 'get_booster'):
            # XGBoost Scikit-learn API
            self.booster = model.get_booster()
            self.model_type = 'xgboost'
        elif type(model).__name__ in ['LGBMRegressor', 'LGBMClassifier', 'Booster'] and hasattr(model, 'dump_model'):
            # LightGBM
            self.booster = model
            self.model_type = 'lightgbm'
        elif type(model).__name__ in ['CatBoost', 'CatBoostRegressor', 'CatBoostClassifier']:
            # CatBoost
            self.booster = model
            self.model_type = 'catboost'
        else:
            raise ValueError(f"Model type {type(model).__name__} is not supported. Use XGBoost, LightGBM, or CatBoost.")
            
        if self.feature_names is None:
            if hasattr(self.booster, 'feature_names') and self.booster.feature_names is not None:
                self.feature_names = self.booster.feature_names
            elif hasattr(self.booster, 'feature_name'):
                self.feature_names = self.booster.feature_name()

    def extract_dynamics(self) -> Dict[str, Any]:
        """
        Extract core mathematical dynamics from the tree structures.
        """
        leaf_values = []
        split_features = []
        depths = []
        num_trees = 0
        
        if self.model_type == 'xgboost':
            dump = self.booster.get_dump(dump_format='json')
            trees = [json.loads(d) for d in dump]
            num_trees = len(trees)
            
            def traverse_xgb(node, current_depth):
                depths.append(current_depth)
                if 'leaf' in node:
                    leaf_values.append(node['leaf'])
                else:
                    if 'split' in node:
                        split_features.append(node['split'])
                    for child in node.get('children', []):
                        traverse_xgb(child, current_depth + 1)
                        
            for tree in trees:
                traverse_xgb(tree, 0)
                
        elif self.model_type == 'lightgbm':
            dump = self.booster.dump_model()
            trees = dump.get('tree_info', [])
            num_trees = len(trees)
            
            def traverse_lgb(node, current_depth):
                depths.append(current_depth)
                if 'leaf_value' in node:
                    leaf_values.append(node['leaf_value'])
                else:
                    if 'split_feature' in node:
                        # Convert feature index to name if possible, or string index
                        feat = str(node['split_feature'])
                        if self.feature_names and node['split_feature'] < len(self.feature_names):
                            feat = self.feature_names[node['split_feature']]
                        split_features.append(feat)
                    if 'left_child' in node:
                        traverse_lgb(node['left_child'], current_depth + 1)
                    if 'right_child' in node:
                        traverse_lgb(node['right_child'], current_depth + 1)
                        
            for tree in trees:
                traverse_lgb(tree.get('tree_structure', {}), 0)
                
        elif self.model_type == 'catboost':
            import tempfile
            import os
            # CatBoost requires saving to a temporary file to get JSON dump
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
                tmp_name = tmp.name
            try:
                self.booster.save_model(tmp_name, format="json")
                with open(tmp_name, 'r') as f:
                    dump = json.load(f)
                
                # A very simplified CatBoost traverse (CatBoost uses oblivious trees)
                # This approximates leaf variances from the leaf values matrix
                if 'oblivious_trees' in dump:
                    trees = dump['oblivious_trees']
                    num_trees = len(trees)
                    for tree in trees:
                        # Extract leaves
                        if 'leaf_values' in tree:
                            vals = tree['leaf_values']
                            # Flatten if multi-class
                            if isinstance(vals[0], list):
                                for v in vals: leaf_values.extend(v)
                            else:
                                leaf_values.extend(vals)
                        # Extract splits (which are shared across the depth layer)
                        if 'splits' in tree:
                            depth = len(tree['splits'])
                            depths.append(depth)
                            # CatBoost JSON splits are complex, we just fetch feature indices
                            for split in tree['splits']:
                                if 'float_feature_index' in split:
                                    feat = str(split['float_feature_index'])
                                    if self.feature_names and split['float_feature_index'] < len(self.feature_names):
                                        feat = self.feature_names[split['float_feature_index']]
                                    split_features.append(feat)
            finally:
                if os.path.exists(tmp_name):
                    os.remove(tmp_name)
        
        leaf_variance = np.var(leaf_values) if leaf_values else 0.0
        max_depth_reached = max(depths) if depths else 0
        feat_counts = Counter(split_features)
        
        return {
            "num_trees": num_trees,
            "total_leaves": len(leaf_values),
            "max_depth_reached": max_depth_reached,
            "leaf_variance": float(leaf_variance),
            "top_splits": dict(feat_counts.most_common(5))
        }

    def report(self) -> None:
        """
        Generate a human-readable diagnostic report for tree dynamics.
        """
        stats = self.extract_dynamics()
        lines = []
        lines.append("=" * 60)
        lines.append("  GradTracer — Tree Dynamics (Node-Level Analysis)")
        lines.append("=" * 60)
        lines.append(f"🌳 Trees analyzed: {stats['num_trees']}")
        lines.append(f"🍃 Total leaves: {stats['total_leaves']} (Max Depth: {stats['max_depth_reached']})")
        lines.append(f"📉 Leaf Variance (Velocity): {stats['leaf_variance']:.5f}")
        lines.append("")
        lines.append("🪓 Top Split Features (Concentration):")
        for f, c in stats['top_splits'].items():
            lines.append(f"   - {f}: {c} splits")
            
        lines.append("")
        lines.append("⚠️  Alerts & Prescriptions")
        if stats['leaf_variance'] < 1e-4:
             lines.append("  🐌 STAGNANT LEAVES (Low Variance)")
             lines.append("     💊 Recommendation: Increase learning rate (eta) or max_depth. The model nodes are under-contributing.")
        elif stats['leaf_variance'] > 1.0:
             lines.append("  🧨 EXPLODING LEAVES (High Variance)")
             lines.append("     💊 Recommendation: Decrease learning rate (eta) or apply L1/L2 regularization to leaf weights (lambda/alpha).")
        else:
             lines.append("  ✅ Tree dynamics are healthy. Leaves are actively contributing.")
             
        lines.append("=" * 60)
        print("\n".join(lines))
