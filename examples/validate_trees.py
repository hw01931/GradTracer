"""
Validation Script: GradTracer Tree Analysis (Structural Intelligence)
Verifies the merged TreeTracker and FeatureAdvisor modules for XGBoost and LightGBM.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from gradtracer.analyzers.trees import TreeTracker, FeatureAdvisor

def run_xgboost_validation():
    print("\n[Trial 1] XGBoost Structural Analysis")
    X, y = make_classification(n_samples=5000, n_features=10, n_informative=5, random_state=42)
    feature_names = [f"feat_{i}" for i in range(10)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    params = {'max_depth': 4, 'eta': 0.3, 'objective': 'binary:logistic'}
    model = xgb.train(params, dtrain, num_boost_round=10)
    
    # Track
    tracker = TreeTracker(model)
    tracker.report()
    
    # Feature Advisor
    recomms = FeatureAdvisor.recommend_interactions(model, feature_names)
    print("--- Recommended Interactions (XGBoost) ---")
    for f1, f2, score in recomms[:3]:
        print(f"  - {f1} x {f2} (Strength: {score:.3f})")

def run_lightgbm_validation():
    print("\n[Trial 2] LightGBM Structural Analysis")
    X, y = make_classification(n_samples=5000, n_features=10, n_informative=5, random_state=42)
    feature_names = [f"feat_{i}" for i in range(10)]
    
    # Train model
    train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
    params = {'objective': 'binary', 'num_leaves': 16, 'learning_rate': 0.1}
    model = lgb.train(params, train_data, num_boost_round=10)
    
    # Track
    tracker = TreeTracker(model)
    tracker.report()
    
    # Feature Advisor
    recomms = FeatureAdvisor.recommend_interactions(model, feature_names)
    print("--- Recommended Interactions (LightGBM) ---")
    for f1, f2, score in recomms[:3]:
        print(f"  - {f1} x {f2} (Strength: {score:.3f})")

def run_leakage_detection():
    print("\n[Trial 3] Feature Leakage Detection")
    X, y = make_classification(n_samples=1000, n_features=5, random_state=42)
    feature_names = [f"feat_{i}" for i in range(5)]
    
    # Inject leakage
    X = np.append(X, (y + np.random.normal(0, 0.001, y.shape)).reshape(-1, 1), axis=1)
    feature_names.append("leaky_target_copy")
    
    leaks = FeatureAdvisor.detect_leakage(X, y, feature_names)
    print("!!! LEAKAGE ALERT !!!")
    for l in leaks:
        print(f"  - {l}")

if __name__ == "__main__":
    run_xgboost_validation()
    run_lightgbm_validation()
    run_leakage_detection()
    print("\nValidation Complete.")
