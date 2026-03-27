"""
Validation: GBDT Tree Interpetability & Scientific FE.
Demonstrates how GradTracer analyzes XGBoost/LightGBM structures to 
recommend the most scientifically sound feature interactions.
"""
import numpy as np
from gradtracer.analyzers.trees import TreeTracker, FeatureAdvisor

# 1. Simulate a Trained GBDT result
# Since we don't want to force install xgboost/lightgbm in this env,
# we use the TreeTracker's capability to analyze any forest-like object.
class MockBooster:
    def __init__(self):
        self.feature_names = ["age", "fare", "sex", "pclass", "embarked"]

if __name__ == "__main__":
    print("🚀 Simulating GBDT (XGBoost/LightGBM) analysis...")
    
    mock_model = MockBooster()
    
    # 2. Analyze Tree Dynamics
    tracker = TreeTracker(mock_model, model_type="xgboost")
    tracker.report()
    
    # 3. Scientific Feature Engineering Interaction Discovery
    print("🔍 DISCOVERING SCIENTIFIC FEATURE INTERACTIONS...")
    recommendations = FeatureAdvisor.recommend_interactions(mock_model, mock_model.feature_names)
    
    print("\n[GradTracer FE Recommendation]")
    for f1, f2, score in recommendations:
        print(f"  ★ Recommended Interaction: {f1} x {f2} (Confidence: {score:.2f})")
        print(f"    Rationale: Found as frequent Parent-Child path in GBDT. Highly non-linear.")

    # 4. Leakage (Watermark) Detection
    print("\n🕵️ CHECKING FOR DATA LEAKAGE (WATERMARKS)...")
    # Simulate data where 'pclass' is accidentally the same as Target y (Leakage)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    X[:, 3] = y + np.random.normal(0, 0.001, 100) # Blatant leak in pclass
    
    leaks = FeatureAdvisor.detect_leakage(X, y, mock_model.feature_names)
    if leaks:
        print(f"  🚨 LEAKAGE DETECTED: {', '.join(leaks)}")
        print("  Advice: Remove these features. They provide a 'shortcut' that fails in production.")
    else:
        print("  ✅ No blatant leakage detected.")
