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
    Supports XGBoost out of the box.
    """

    def __init__(self, model, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names
        
        if hasattr(model, 'get_booster'):
            self.booster = model.get_booster()
            self.model_type = 'xgboost'
        else:
            raise ValueError("Only XGBoost is supported for TreeDynamicsTracker right now.")
            
        if self.feature_names is None and hasattr(self.booster, 'feature_names'):
            self.feature_names = self.booster.feature_names

    def extract_dynamics(self) -> Dict[str, Any]:
        """
        Extract core mathematical dynamics from the tree structures.
        """
        dump = self.booster.get_dump(dump_format='json')
        trees = [json.loads(d) for d in dump]
        
        leaf_values = []
        split_features = []
        depths = []
        
        def traverse(node, current_depth):
            depths.append(current_depth)
            if 'leaf' in node:
                leaf_values.append(node['leaf'])
            else:
                if 'split' in node:
                    split_features.append(node['split'])
                for child in node.get('children', []):
                    traverse(child, current_depth + 1)
                    
        for tree in trees:
            traverse(tree, 0)
            
        leaf_variance = np.var(leaf_values) if leaf_values else 0.0
        max_depth_reached = max(depths) if depths else 0
        feat_counts = Counter(split_features)
        
        return {
            "num_trees": len(trees),
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
        lines.append("  FlowGrad — Tree Dynamics (Node-Level Analysis)")
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
