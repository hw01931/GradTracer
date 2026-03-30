<p align="center">
  <h1 align="center">🌊 GradTracer</h1>
  <p align="center">
    <strong>Flow-based Diagnostics for Embedding Systems & Model Compression</strong>
  </p>
  <p align="center">
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-empirical-utility--scientific-validation">Empirical Utility</a> •
    <a href="#-core-focus">Core Focus</a> •
    <a href="#-ai-agent-integration">AI Agent Integration</a>
  </p>
</p>

---

GradTracer is a specialized diagnostic library for **Embedding-heavy Recommendation Systems (RecSys)** and **Model Performance Optimization**. 

While traditional loggers track scalars, GradTracer monitors training dynamics (`dG/dt`)—such as embedding drift, gradient oscillation, and exposure frequency—to diagnose silent failures like representation collapse. **All diagnostics are exported as causal JSON**, designed specifically for automated intervention by AI coding assistants.

---

## 📊 Empirical Utility & Scientific Validation

GradTracer isn't just about visualization; it provides **scientifically backed performance improvements**. We rigorously validate our prescriptions across synthetic and real-world benchmarks.

### 1. Bayesian Auto-Fix: Generalization Boost
**Scenario**: Identifying and intercepting "Zombie" embeddings (popular items with oscillating, conflicting gradients) during the backward pass.

| Metric | Baseline | **GradTracer (Auto-Fix)** | Delta |
| :--- | :--- | :--- | :--- |
| **Test Loss (BCELoss)** | 0.9434 | **0.9131** | **-3.22%** |
| **Representation Drift** | High | **Low (Normalized)** | -45% |

> [!NOTE]
> By dynamically scaling gradients based on SNR and global loss posteriors, GradTracer prevented representation collapse in 1M+ parameter embedding tables. (Verified in [validate_autofix.py](examples/validate_autofix.py))

### 2. Joint Compression: Accuracy Retention at 80% Sparsity
**Scenario**: Compressing a high-capacity model by 80% using structural pruning and INT8 quantization.

| Metric | Uniform Pruning + INT8 | **GradTracer (Heterogeneous)** | Delta |
| :--- | :--- | :--- | :--- |
| **Accuracy (80% Sparsity)** | 58.35% | **78.52%** | **+20.17%** |
| **Model Size Reduction** | 10.5x | **10.5x** | **Same Footprint** |

> [!TIP]
> GradTracer's `CompressionTracker` uses Fisher-informed sensitivity analysis to protect "Signal-rich bottleneck layers," preventing the collapse typically observed in uniform pruning. (Verified in [validate_joint_compression.py](examples/validate_joint_compression.py))

### 3. GBDT Feature Interaction: Statistical Significance
**Scenario**: Using `FeatureAdvisor` to identify non-linear interactions in XGBoost/LightGBM structures on the **Adult Census Income** dataset.
...

## 🎯 Core Focus: RecSys & Compression

### 1. Embedding Dynamics Tracker & Bayesian Auto-Fix
General DL diagnostics often fail for RecSys because embedding tables are highly sparse and suffer from popularity bias. `EmbeddingTracker` identifies:
*   **Zombie Embeddings:** Items with high update velocity but oscillating gradient directions (failing to generalize).
*   **Dead Embeddings:** Items suffering from severe cold-start or broken negative sampling.
*   **⚡ [NEW] Bayesian Auto-Fix:** Pass `auto_fix=True` to let GradTracer actively intercept and dynamically scale gradients for Zombie embeddings during the backward pass.
*   **🛡️ [NEW] White-Glass Audit:** Complete transparency. The `AutoFixAuditLogger` saves every causal intervention locally to `.gradtracer/audit.jsonl`.

### 2. Auto-Compression Suite & IDE Extension
Instead of blindly pruning based on weight magnitude, GradTracer uses training dynamics to guide compression:
*   **🥇 Auto-Compression Recipe (`RecipeGenerator`):** Analyzes dynamic health and SNR to automatically output an optimal mixed-precision (FP16/INT8/INT4) and joint structural pruning recipe.
*   **💻 VS Code Extension:** Hover over Python layer definitions to see real-time GradTracer diagnostic popups and 1-click apply compression recipes.

### 3. Deep Tree Dynamics (v0.6)
*   **Node-Level GBDT Tracking (`TreeDynamicsTracker`):** Tracks Leaf Velocity (Variance) and Feature Split Concentration for XGBoost, LightGBM, and CatBoost.

---

## 🤖 AI Agent Integration
GradTracer serves as a "Decision Layer" for AI coding assistants. Models receive exact logic and prescriptions via `AgentExporter`.

```python
from gradtracer.agent import AgentExporter
from gradtracer.analyzers.embedding import EmbeddingTracker

tracker = EmbeddingTracker(model.item_emb, auto_fix=True, track_interval=100)
# ... training loop (tracker.step()) ...
print(AgentExporter.export_embedding(tracker))
```

---

## 🚀 Quick Start
```bash
pip install gradtracer
```

Detailed benchmarks and mathematical methodology can be found in our [Benchmarks Documentation](docs/BENCHMARKS.md).

## License
[MIT](LICENSE)
