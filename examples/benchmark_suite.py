"""
Extreme Performance Benchmark Suite: FeatureAdvisor Interaction Boost
Tests multiple real-world datasets and varying tree depths (2, 4, 6)
to verify the robustness of TreeTracker's recommendations.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_hastie_10_2, fetch_openml
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from scipy import stats
from gradtracer.analyzers.trees import FeatureAdvisor
import warnings

warnings.filterwarnings('ignore')

def load_all_datasets():
    """Fetches and cleans multiple benchmark datasets."""
    print("--- Loading and Preprocessing Extreme Benchmarks ---")
    datasets = {}
    
    # 1. Hastie-10-2
    X_h, y_h = make_hastie_10_2(n_samples=10000, random_state=42)
    datasets['hastie'] = (X_h.astype(float), (y_h == 1).astype(int), [f"x{i}" for i in range(10)])
    
    # helper for openml
    def fetch_clean_by_id(data_id):
        print(f"  Fetching Dataset ID {data_id}...")
        data = fetch_openml(data_id=data_id, version=1, as_frame=True)
        df = data.frame
        target = data.target_names[0]
        # Binarize if necessary
        if df[target].dtype == 'object' or df[target].dtype.name == 'category':
            y = (df[target] == df[target].unique()[0]).astype(int).values
        else:
            y = (df[target] > df[target].median()).astype(int).values
        X_raw = df.drop(columns=[target])
        X = pd.get_dummies(X_raw, drop_first=True)
        return X.values.astype(float), y, list(X.columns)

    # 2. Adult Census (ID 1590)
    try:
        datasets['adult'] = fetch_clean_by_id(1590)
    except Exception as e:
        print(f"  Error loading Adult: {e}")
    
    # 3. Credit-G (ID 31)
    try:
        datasets['credit-g'] = fetch_clean_by_id(31)
    except Exception as e:
        print(f"  Error loading Credit-G: {e}")
    
    # 4. Bank Marketing (ID 1461)
    try:
        datasets['bank'] = fetch_clean_by_id(1461)
    except Exception as e:
        print(f"  Error loading Bank: {e}")
    
    # 5. Steel Plates Faults (ID 1504)
    try:
        datasets['steel'] = fetch_clean_by_id(1504)
    except Exception as e:
        print(f"  Error loading Steel: {e}")

    return datasets

def run_dataset_benchmark(ds_name, X, y, feature_names, depths):
    print(f"\n--- [BENCHMARK] Dataset: {ds_name} ---")
    results = []
    
    for depth in depths:
        print(f"  Testing max_depth={depth}...")
        kb = KFold(n_splits=5, shuffle=True, random_state=42)
        base_scores = []
        enh_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(kb.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
            dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
            
            params = {'max_depth': depth, 'eta': 0.1, 'objective': 'binary:logistic', 'eval_metric': 'auc'}
            model = xgb.train(params, dtrain, num_boost_round=50)
            
            base_auc = roc_auc_score(y_test, model.predict(dtest))
            base_scores.append(base_auc)
            
            recomms = FeatureAdvisor.recommend_interactions(model, feature_names, top_k=2)
            
            X_train_df = pd.DataFrame(X_train, columns=feature_names)
            X_test_df = pd.DataFrame(X_test, columns=feature_names)
            new_feats = list(feature_names)
            for f1, f2, _ in recomms:
                new_name = f"inter_{f1}_{f2}"
                X_train_df[new_name] = X_train_df[f1] * X_train_df[f2]
                X_test_df[new_name] = X_test_df[f1] * X_test_df[f2]
                new_feats.append(new_name)
            
            dtrain_enh = xgb.DMatrix(X_train_df, label=y_train, feature_names=new_feats)
            dtest_enh = xgb.DMatrix(X_test_df, label=y_test, feature_names=new_feats)
            
            model_enh = xgb.train(params, dtrain_enh, num_boost_round=50)
            enh_auc = roc_auc_score(y_test, model_enh.predict(dtest_enh))
            enh_scores.append(enh_auc)
            
        avg_base = np.mean(base_scores)
        avg_enh = np.mean(enh_scores)
        t_stat, p_val = stats.ttest_rel(enh_scores, base_scores)
        delta = (avg_enh - avg_base) * 100
        
        results.append({
            "Depth": depth,
            "Baseline": avg_base,
            "Enhanced": avg_enh,
            "Delta%": delta,
            "P-value": p_val
        })
        print(f"    Depth {depth}: Base={avg_base:.4f} | Enh={avg_enh:.4f} | Delta={delta:+.3f}% | P={p_val:.4f}")

    return results

if __name__ == "__main__":
    datasets = load_all_datasets()
    depths = [2, 4, 6]
    
    report_lines = ["\n" + "="*85]
    report_lines.append(" EXTREME TREE PERFORMANCE BENCHMARK SUMMARY")
    report_lines.append("="*85)
    report_lines.append(f"{'Dataset':<15s} {'Depth':^7s} {'Base AUC':^10s} {'Enh AUC':^10s} {'Delta %':^10s} {'P-value':^10s}")
    report_lines.append("-" * 85)
    
    for name, (X, y, f) in datasets.items():
        res = run_dataset_benchmark(name, X, y, f, depths)
        for row in res:
            p_mark = "WIN" if row['P-value'] < 0.05 else "   "
            line = f"{name:<15s} {row['Depth']:^7d} {row['Baseline']:^10.4f} {row['Enhanced']:^10.4f} {row['Delta%']:^+10.3f}% {row['P-value']:^10.4f} {p_mark}"
            report_lines.append(line)
            
    report_lines.append("="*85)
    print("\n".join(report_lines))
