<p align="center">
  <h1 align="center">🌊 GradTracer</h1>
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

GradTracer is a specialized diagnostic library designed for **Embedding-heavy Recommendation Systems (RecSys)** and **Model Compression**. 

Rather than competing with general-purpose loggers like TensorBoard or Weights & Biases, GradTracer tracks step-by-step training dynamics (`dG/dt`)—such as embedding drift, gradient oscillation, and exposure frequency—to diagnose silent failures (e.g., representation collapse) that traditional scalar metrics miss.

Crucially, every GradTracer module exports its findings as **structured causal XML**, allowing AI coding assistants (Cursor, Copilot, Antigravity) to automatically parse the diagnosis and apply statistically backed prescriptions.

## 🎯 Core Focus: RecSys & Compression

### 1. Embedding Dynamics Tracker & Bayesian Auto-Fix (v0.6+)
General DL diagnostics often fail for RecSys because embedding tables are highly sparse and suffer from popularity bias. `EmbeddingTracker` identifies:
*   **Zombie Embeddings:** Items with high update velocity but oscillating gradient directions (failing to generalize).
*   **Dead Embeddings:** Items suffering from severe cold-start or broken negative sampling.
*   **Popularity Bias:** Exposure distribution skew (Gini/Entropy) that hurts long-tail coverage.
*   **Frequency-Aware Saliency:** Normalizes update velocity by exposure frequency to identify truly important embeddings for pruning.
*   **⚡ [NEW] Bayesian Auto-Fix:** Pass `auto_fix=True` to let GradTracer actively intercept and dynamically scale gradients for Zombie embeddings during the backward pass based on SNR and global loss posteriors.
*   **🌍 [NEW] DDP Support:** Automatically handles `torch.distributed.all_reduce` to aggregate embedding stats across multiple GPUs.

### 2. Dynamics-Aware Compression Suite
Instead of blindly pruning based on weight magnitude, GradTracer uses training dynamics to guide compression:
*   **Dynamic Saliency (`SaliencyAnalyzer`):** Ranks layers by how actively they are learning (velocity + momentum).
*   **Quantization Guidance (`QuantizationAdvisor`):** Recommends mixed-precision (4/8/16-bit) based on layer-specific gradient SNR and weight variance.

### 3. Deep Tree Dynamics (v0.6)
*   **Node-Level GBDT Tracking (`TreeDynamicsTracker`):** Unlike basic feature importance, GradTracer unpacks the raw tree structure to track Leaf Velocity (Variance) and Feature Split Concentration. Evaluates mathematically if trees are stagnating or exploding.
*   **[NEW] Broad Support:** Now fully supports `XGBoost`, `LightGBM`, and `CatBoost`.

### 4. Complex Architectures (v0.6)
*   **`FlowManager`:** Centralized multi-tracker hub designed for Two-Tower, GNN, or Sequential architectures to calculate cross-layer correlations (e.g., User Tower vs. Item Tower).

## 🤖 AI Agent XML Export
GradTracer serves as a "Decision Layer" for AI coding assistants. By calling `.to_agent_xml()`, models receive exact logic and prescriptions.

```python
from gradtracer.analyzers.embedding import EmbeddingTracker

tracker = EmbeddingTracker(model.item_emb, auto_fix=True, track_interval=100)
# ... training loop (tracker.step()) ...

print(tracker.to_agent_xml())
```

```xml
<gradtracer_embedding_report layer="item_emb">
  <causal_model type="ZOMBIE_EMBEDDINGS">
    <premise>8.5% of embeddings have high update velocity but strictly negative cosine similarity between steps.</premise>
    <implies>Optimizer is oscillating. Conflicting gradients from different users are pulling these embeddings back and forth.</implies>
    <action>AUTO_FIX_ENGAGED</action>
    <expected_effect>Scaled down gradients for 1,204 oscillatory indices based on Bayesian variance.</expected_effect>
    <confidence>0.92</confidence>
  </causal_model>
</gradtracer_embedding_report>
```

## 📊 Mathematical & Statistical Validation
GradTracer's recommendations are backed by formal statistical tests. As demonstrated in our [Validation Notebooks](examples/), our Auto-Fix logic and embedding prescriptions yield **Statistically Significant Improvements** mathematically guaranteed via:
*   **NDCG@10 & Hit Rate@10 Paired t-tests** demonstrating statistically rigorous ranking improvements when Auto-Fix intercepts oscillatory parameters in MovieLens-100K MF baselines.
*   **Cosine Similarity Tracking** guaranteeing true oscillatory embeddings rather than in-sample noise.

## 🚀 Quick Start
```bash
pip install git+https://github.com/hw01931/GradTracer.git
```

## 🧩 Experimental Modules
While optimized for RecSys and Compression, GradTracer still includes its original Python dynamics modules:
*   `FlowTracker`: General PyTorch training stability (SNR, Stagnation).
*   `FeatureAnalyzer`: VIF-filtered interaction suggestions.

## License
[MIT](LICENSE)
