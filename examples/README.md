# 🚀 GradTracer Examples & Validation Suite

This directory contains a collection of scripts used for benchmarking, validation, and demonstrating core library features.

---

## 🏗️ Benchmarking & Performance
| File | Description | Focus |
| :--- | :--- | :--- |
| [`benchmark.py`](benchmark.py) | Measures CPU/GPU overhead on 1M+ embedding tables. | Efficiency |
| [`benchmark_models.py`](benchmark_models.py) | Comparative performance across XGBoost, LightGBM, and CatBoost. | Scaling |
| [`verify_performance_boost.py`](verify_performance_boost.py) | **Primary Validation Script** for AUC gains on Adult-Census data. | Utility |

## 🌊 Embedding & Auto-Fix
| File | Description | Focus |
| :--- | :--- | :--- |
| [`validate_autofix.py`](validate_autofix.py) | **Primary Validation Script** for Bayesian Auto-Fix generalization boost. | Accuracy |
| [`validate_recsys_real.py`](validate_recsys_real.py) | Validates Popularity Bias and Gini index tracking on real RecSys datasets. | Stability |

## 🌲 Tree Analysis (GBDT)
| File | Description | Focus |
| :--- | :--- | :--- |
| [`validate_trees.py`](validate_trees.py) | Basic structure analysis and split concentration metrics. | Interpretation |
| [`validate_trees_fe.py`](validate_trees_fe.py) | Non-linear interaction discovery and scientific FE recommendations. | Discovery |

## ✂️ Compression & Pruning
| File | Description | Focus |
| :--- | :--- | :--- |
| [`validate_compression.py`](validate_compression.py) | Dynamic-guided structural pruning vs. static-magnitude pruning. | Optimization |
| [`validate_joint_compression.py`](validate_joint_compression.py) | Mixed-precision quantization (FP16 $\to$ INT8/4) fused with pruning. | Edge Deployment |
| [`validate_quantization.py`](validate_quantization.py) | Standalone quantization drift detection. | Integrity |

---

## 🏃 How to Run
To run any example from the repository root:
```bash
$env:PYTHONPATH="."  # Windows
export PYTHONPATH="." # Linux/MacOS
python examples/validate_autofix.py
```

> [!TIP]
> Use `PYTHONUTF8=1` on Windows if you experience encoding issues with emojis in the console output.
