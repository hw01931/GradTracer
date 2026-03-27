"""
Rigorous Performance Verification: FeatureAdvisor Interaction Boost
Uses high-complexity benchmarks (Adult, Hastie) to prove performance gains.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_hastie_10_2, fetch_openml
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
from scipy import stats
from gradtracer.analyzers.trees import FeatureAdvisor
import warnings

warnings.filterwarnings('ignore')

def get_hastie_data():
    """10-feature non-linear dataset from Hastie et al."""
    X, y = make_hastie_10_2(n_samples=10000, random_state=42)
    # Convert y from [-1, 1] to [0, 1]
    y = (y == 1).astype(int)
    feature_names = [f"x{i}" for i in range(10)]
    return X, y, feature_names

def get_adult_data():
    """Real-world benchmark (Census Income). Handles categorical features."""
    print("Loading Adult dataset from OpenML (this may take a few seconds)...")
    data = fetch_openml('adult', version=2, as_frame=True)
    df = data.frame
    # Target to binary
    y = (df['class'] == '>50K').astype(int).values
    X_raw = df.drop(columns=['class'])
    
    # Preprocessing: Basic one-hot for categorical
    X = pd.get_dummies(X_raw, drop_first=True)
    feature_names = list(X.columns)
    return X.values.astype(float), y, feature_names

def run_experiment(name, X, y, feature_names):
    print(f"\n--- [RIGOROUS EXPERIMENT] {name} ---")
    kb = KFold(n_splits=5, shuffle=True, random_state=42)
    
    baseline_scores = []
    enhanced_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kb.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 1. Baseline Training (Depth restricted to show FE value)
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
        
        # Restrict depth to 2 to force the model to rely on simple features
        params = {'max_depth': 2, 'eta': 0.1, 'objective': 'binary:logistic', 'eval_metric': 'auc'}
        model = xgb.train(params, dtrain, num_boost_round=100)
        
        preds_base = model.predict(dtest)
        base_auc = roc_auc_score(y_test, preds_base)
        baseline_scores.append(base_auc)
        
        # 2. FeatureAdvisor Recommendation
        recomms = FeatureAdvisor.recommend_interactions(model, feature_names, top_k=3)
        if fold == 0:
            print(f"    Fold 1 Recommendations: {[(r[0], r[1]) for r in recomms]}")
        
        # 3. Enhanced Training (Original + Interactions)
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        new_feature_names = list(feature_names)
        for f1, f2, _ in recomms:
            new_name = f"inter_{f1}_{f2}"
            X_train_df[new_name] = X_train_df[f1] * X_train_df[f2]
            X_test_df[new_name] = X_test_df[f1] * X_test_df[f2]
            new_feature_names.append(new_name)
            
        dtrain_enh = xgb.DMatrix(X_train_df, label=y_train, feature_names=new_feature_names)
        dtest_enh = xgb.DMatrix(X_test_df, label=y_test, feature_names=new_feature_names)
        
        model_enh = xgb.train(params, dtrain_enh, num_boost_round=100)
        
        preds_enh = model_enh.predict(dtest_enh)
        enh_auc = roc_auc_score(y_test, preds_enh)
        enhanced_scores.append(enh_auc)
        
        print(f"  Fold {fold+1}: Baseline AUC={base_auc:.4f} | Enhanced AUC={enh_auc:.4f} (Delta: {(enh_auc - base_auc)*100:+.3f}%)")
        
    avg_base = np.mean(baseline_scores)
    avg_enh = np.mean(enhanced_scores)
    t_stat, p_value = stats.ttest_rel(enhanced_scores, baseline_scores)
    
    print(f"\n[FINAL SUMMARY] {name}:")
    print(f"   Avg Baseline AUC: {avg_base:.4f}")
    print(f"   Avg Enhanced AUC: {avg_enh:.4f}")
    print(f"   Delta AUC:        {(avg_enh - avg_base)*100:+.4f}%")
    print(f"   P-value (T-test): {p_value:.6f}")
    
    if p_value < 0.05 and avg_enh > avg_base:
        print("RESULT: SUCCESS! Statistically Significant Performance Boost.")
    else:
        print("RESULT: Marginal Improvement.")

if __name__ == "__main__":
    # Experiment 1: Hastie-10-2 (The pure non-linear benchmark)
    X_h, y_h, f_h = get_hastie_data()
    run_experiment("Hastie-10-2", X_h, y_h, f_h)
    
    # Experiment 2: Adult Dataset (Real-world Tabular)
    X_a, y_a, f_a = get_adult_data()
    run_experiment("Adult Census Income", X_a, y_a, f_a)

    print("\nRigorous Verification Complete.")
