"""
FlowGrad — ML Training Dynamics Tracker

One-line layer-by-layer visualization for PyTorch & boosting models.
Advanced compression diagnostics, feature engineering, and AI-native agent output.

Usage (PyTorch):
    from flowgrad import FlowTracker
    tracker = FlowTracker(model)
    for epoch in range(100):
        loss = train(model)
        tracker.step(loss=loss)
    tracker.report()

Usage (XGBoost / LightGBM / CatBoost):
    from flowgrad import BoostingTracker
    tracker = BoostingTracker()
    model = xgb.train(params, dtrain, callbacks=[tracker.as_xgb_callback()])
    tracker.report()

Usage (Compression):
    from flowgrad import CompressionTracker
    tracker = CompressionTracker(model, eval_fn=accuracy_fn)
    result = tracker.auto_compress(performance_floor=0.95)

Usage (AI Agent Mode):
    xml = tracker.export_for_agent()
    print(xml)  # AI reads this and auto-applies fixes
"""

__version__ = "0.5.0"

from flowgrad.tracker import FlowTracker
from flowgrad.analyzers.boosting import BoostingTracker
from flowgrad.analyzers.sklearn_tracker import SklearnTracker
from flowgrad.analyzers.features import FeatureAnalyzer
from flowgrad.analyzers.compression import CompressionTracker
from flowgrad.analyzers.saliency import SaliencyAnalyzer
from flowgrad.analyzers.quantization import QuantizationAdvisor
from flowgrad.analyzers.distillation import DistillationTracker
from flowgrad.analyzers.peft import PEFTTracker
from flowgrad.analyzers.embedding import EmbeddingTracker
from flowgrad.agent import AgentExporter
from flowgrad.history import HistoryTracker


def info():
    """
    Print information about FlowGrad and its optional dependencies.
    Useful for debugging installation or reporting issues.
    """
    import sys
    print("=" * 45)
    print(f"🌊 FlowGrad version: {__version__}")
    print(f"🐍 Python version:   {sys.version.split(' ')[0]}")
    print("=" * 45)
    print("📦 Optional Dependencies:")

    deps = [
        ("PyTorch", "torch"),
        ("scikit-learn", "sklearn"),
        ("XGBoost", "xgboost"),
        ("LightGBM", "lightgbm"),
        ("CatBoost", "catboost"),
    ]
    for label, module in deps:
        try:
            mod = __import__(module)
            ver = getattr(mod, "__version__", "?")
            print(f"  {label:<14s} {ver} ✅")
        except ImportError:
            print(f"  {label:<14s} Not installed ❌")

    print("=" * 45)
    print("🧩 Available Modules:")
    print(f"  FlowTracker          DL training dynamics")
    print(f"  BoostingTracker      XGBoost/LightGBM/CatBoost")
    print(f"  SklearnTracker       scikit-learn models")
    print(f"  FeatureAnalyzer      Feature engineering + VIF")
    print(f"  CompressionTracker   Auto pruning search")
    print(f"  SaliencyAnalyzer     Dynamic pruning priority")
    print(f"  QuantizationAdvisor  Mixed-precision guidance")
    print(f"  DistillationTracker  Knowledge distillation")
    print(f"  PEFTTracker          LoRA / adapter optimization")
    print(f"  EmbeddingTracker     RecSys embedding dynamics")
    print(f"  AgentExporter        AI-native XML output")
    print("=" * 45)


__all__ = [
    "FlowTracker",
    "BoostingTracker",
    "SklearnTracker",
    "FeatureAnalyzer",
    "CompressionTracker",
    "SaliencyAnalyzer",
    "QuantizationAdvisor",
    "DistillationTracker",
    "PEFTTracker",
    "EmbeddingTracker",
    "AgentExporter",
    "HistoryTracker",
    "info",
    "__version__",
]
