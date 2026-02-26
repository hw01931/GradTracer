<p align="center">
  <h1 align="center">🌊 FlowGrad</h1>
  <p align="center">
    <strong>Flow-based Diagnostics for Embedding Systems & Compression</strong>
  </p>
  <p align="center">
    <a href="#installation">Installation</a> •
    <a href="#core-focus">Core Focus</a> •
    <a href="#ai-agent-xml">AI Agent Integration</a>
  </p>
</p>

---

FlowGrad is a specialized diagnostic library designed for **Embedding-heavy Recommendation Systems (RecSys)** and **Model Compression**. 

Rather than competing with general-purpose loggers like TensorBoard or Weights & Biases, FlowGrad tracks step-by-step training dynamics (`dG/dt`)—such as embedding drift, gradient oscillation, and exposure frequency—to diagnose silent failures (e.g., representation collapse) that traditional scalar metrics miss.

Crucially, every FlowGrad module exports its findings as **structured causal XML**, allowing AI coding assistants (Cursor, Copilot, Antigravity) to automatically parse the diagnosis and apply statistically backed prescriptions.

## 🎯 Core Focus: RecSys & Compression

### 1. Embedding Dynamics Tracker
General DL diagnostics often fail for RecSys because embedding tables are highly sparse and suffer from popularity bias. `EmbeddingTracker` identifies:
*   **Zombie Embeddings:** Items with high update velocity but oscillating gradient directions (failing to generalize).
*   **Dead Embeddings:** Items suffering from severe cold-start or broken negative sampling.
*   **Popularity Bias:** Exposure distribution skew (Gini/Entropy) that hurts long-tail coverage.
*   **Frequency-Aware Saliency:** Normalizes update velocity by exposure frequency to identify truly important embeddings for pruning.

### 2. Dynamics-Aware Compression Suite
Instead of blindly pruning based on weight magnitude, FlowGrad uses training dynamics to guide compression:
*   **Dynamic Saliency (`SaliencyAnalyzer`):** Ranks layers by how actively they are learning (velocity + momentum).
*   **Quantization Guidance (`QuantizationAdvisor`):** Recommends mixed-precision (4/8/16-bit) based on layer-specific gradient SNR and weight variance.
*   **Knowledge Distillation (`DistillationTracker`):** Identifies exactly which layers the student model is struggling to mimic from the teacher.

## 🤖 AI Agent XML Export
FlowGrad serves as a "Decision Layer" for AI coding assistants. By calling `.to_agent_xml()`, models receive exact logic and prescriptions.

```python
from flowgrad import EmbeddingTracker

tracker = EmbeddingTracker(model.item_emb)
# ... training loop (tracker.step()) ...

print(tracker.to_agent_xml())
```

```xml
<flowgrad_embedding_report layer="item_emb">
  <causal_model type="ZOMBIE_EMBEDDINGS">
    <premise>8.5% of embeddings have high update velocity but strictly negative cosine similarity between steps.</premise>
    <implies>Optimizer is oscillating. Conflicting gradients from different users are pulling these embeddings back and forth.</implies>
    <action>DECREASE_LR_BETA_OR_USE_SPARSE_ADAM</action>
    <expected_effect>Smooth out the trajectory of rare items and prevent representation collapse.</expected_effect>
    <confidence>0.92</confidence>
  </causal_model>
</flowgrad_embedding_report>
```

## 📊 Statistical Validation
FlowGrad's recommendations are backed by formal statistical tests. As demonstrated in our [Validation Notebooks](examples/), our feature engineering and embedding prescriptions yield **Statistically Significant Improvements** measured via:
*   **F-Test & Adjusted R²** for Feature Synergy recommendations.
*   **Paired t-tests** demonstrating Variance Reduction and MSE improvements on MovieLens-100K MF baselines.

## 🚀 Quick Start
```bash
pip install git+https://github.com/hw01931/FlowGrad.git
```

## 🧩 Experimental Modules
While optimized for RecSys and Compression, FlowGrad still includes its original v0.2/v0.3 modules for general ML tasks:
*   `FlowTracker`: General PyTorch training stability (SNR, Stagnation).
*   `BoostingTracker`: XGBoost / LightGBM tree dynamics.
*   `FeatureAnalyzer`: VIF-filtered interaction suggestions.

## License
[MIT](LICENSE)
