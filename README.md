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

Crucially, every GradTracer module exports its findings as **standardized causal JSON**, allowing AI coding assistants (Cursor, Copilot, Antigravity) and custom IDE Extensions to automatically parse the diagnosis and apply statistically backed prescriptions.

## 🎯 Core Focus: RecSys & Compression

### 1. Embedding Dynamics Tracker & Bayesian Auto-Fix (v0.6+)
General DL diagnostics often fail for RecSys because embedding tables are highly sparse and suffer from popularity bias. `EmbeddingTracker` identifies:
*   **Zombie Embeddings:** Items with high update velocity but oscillating gradient directions (failing to generalize).
*   **Dead Embeddings:** Items suffering from severe cold-start or broken negative sampling.
*   **Popularity Bias:** Exposure distribution skew (Gini/Entropy) that hurts long-tail coverage.
*   **Frequency-Aware Saliency:** Normalizes update velocity by exposure frequency to identify truly important embeddings for pruning.
*   **⚡ [NEW] Bayesian Auto-Fix:** Pass `auto_fix=True` to let GradTracer actively intercept and dynamically scale gradients for Zombie embeddings during the backward pass based on SNR and global loss posteriors.
*   **🌍 [NEW] DDP Support:** Automatically handles `torch.distributed.all_reduce` to aggregate embedding stats across multiple GPUs.
*   **🛡️ [NEW] White-Glass Audit:** Complete transparency. The `AutoFixAuditLogger` saves every causal intervention locally to `.gradtracer/audit.jsonl` to ensure perfect reproducibility.

### 2. Production Hardening & Ecosystem (v0.7)
*   **🏎️ Zero-Overhead Tracking:** Lazy evaluation with `track_interval` and strictly GPU-bound tensor operations guarantee `< 5%` performance overhead even on massive 1M+ parameter embedding tables (Verified in `examples/benchmark.py`).
*   **🌉 Weights & Biases Bridge:** Use `log_to_wandb(tracker)` to seamlessly stream GradTracer's diagnostics (zombie ratios, health scores, Popularity Gini) directly into your existing W&B dashboards.
*   **🤖 Universal JSON Export:** Replaced legacy XML with standard causal JSON output for flawless integration with modern LLM coding agents.

### 3. Auto-Compression Suite & IDE Extension (v0.7)
Instead of blindly pruning based on weight magnitude, GradTracer uses training dynamics to guide compression:
*   **[NEW] 🥇 Auto-Compression Recipe (`RecipeGenerator`):** The Holy Grail of compression. Analyzes dynamic health and SNR to automatically output an optimal mixed-precision (FP16/INT8/INT4) and joint structural pruning recipe (e.g., "INT4 + 80% Prune").
*   **[NEW] 💻 VS Code Extension:** Integrates directly into your IDE. Hover over Python layer definitions to see real-time GradTracer diagnostic popups and 1-click apply compression recipes.

### 4. Deep Tree Dynamics (v0.6)
*   **Node-Level GBDT Tracking (`TreeDynamicsTracker`):** Unlike basic feature importance, GradTracer unpacks the raw tree structure to track Leaf Velocity (Variance) and Feature Split Concentration. Evaluates mathematically if trees are stagnating or exploding.
*   **[NEW] Broad Support:** Now fully supports `XGBoost`, `LightGBM`, and `CatBoost`.

### 5. Complex Architectures (v0.6)
*   **`FlowManager`:** Centralized multi-tracker hub designed for Two-Tower, GNN, or Sequential architectures to calculate cross-layer correlations (e.g., User Tower vs. Item Tower).

## 🤖 AI Agent JSON Export
GradTracer serves as a "Decision Layer" for AI coding assistants. By calling `AgentExporter.export_embedding()`, models receive exact logic and prescriptions.

```python
from gradtracer.agent import AgentExporter
from gradtracer.analyzers.embedding import EmbeddingTracker

tracker = EmbeddingTracker(model.item_emb, auto_fix=True, track_interval=100)
# ... training loop (tracker.step()) ...

print(AgentExporter.export_embedding(tracker))
```

```json
{
  "gradtracer_embedding_report": {
    "layer": "item_emb",
    "embedding_stats": {
      "num_embeddings": 1000000,
      "active_coverage_pct": 20.4,
      "popularity_gini": 0.85
    },
    "diagnostics": [
      {
        "type": "POPULARITY_BIAS",
        "premise": "Exposure distribution is highly skewed (Gini coefficient: 0.85 > 0.8).",
        "implies": "Model is predominantly optimizing for top-popular items. Long-tail embeddings will suffer from inadequate learning.",
        "action": "APPLY_LOG_Q_CORRECTION_OR_INVERSE_FREQUENCY_SAMPLING",
        "expected_effect": "Debias the softmax logits and improve long-tail recommendation coverage.",
        "confidence": 0.95
      }
    ]
  }
}
```

## 📊 Mathematical & Statistical Validation
GradTracer's recommendations are backed by formal statistical tests. As demonstrated in our [Validation Notebooks](examples/), our Auto-Fix logic and embedding prescriptions yield **Statistically Significant Improvements** mathematically guaranteed via:
*   **NDCG@10 & Hit Rate@10 Paired t-tests** demonstrating statistically rigorous ranking improvements when Auto-Fix intercepts oscillatory parameters in MovieLens-100K MF baselines.
*   **Cosine Similarity Tracking** guaranteeing true oscillatory embeddings rather than in-sample noise.

## 🚀 Quick Start
```bash
pip install gradtracer
```

## 🧩 Experimental Modules
While optimized for RecSys and Compression, GradTracer still includes its original Python dynamics modules:
*   `FlowTracker`: General PyTorch training stability (SNR, Stagnation).
*   `FeatureAnalyzer`: VIF-filtered interaction suggestions.

## License
[MIT](LICENSE)
