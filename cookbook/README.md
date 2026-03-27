# 📖 GradTracer Master Cookbook
**"The Path from Black-Box to Transparent Model"**

This cookbook isn't just a set of scripts; it's a **flight manual** for modern Deep Learning. It demonstrates the full engineering lifecycle: **Observe -> Interpret -> Fix -> Compress.**

---

### [Phase 1] 🌋 Observability & Diagnosis
Every ML project starts with **Hidden Costs.** 
- **Zombies:** We identify *oscillating* parameters that are updated frequently but move nowhere because gradients are in a "tug-of-war." 
- **Starvation:** We use **Pezeshki et al. (2020)** theory to detect if high-level features are being "starved" of gradient signal by cheap shortcuts.

**How to use:**
```python
tracker = FlowTracker(model)
emb_tracker = EmbeddingTracker(model.embedding)
# ... train ...
```

---

### [Phase 2] 🧠 Mechanistic Interpretability (2025 SOTA)
Instead of guessing why accuracy dropped, GradTracer looks into the **representational health** during training.
- **Grokking Progress:** Tracks when the model shifts from memorizing noise to forming robust circuits.
- **Epistemic Uncertainty:** High gradient variance during training flags input regions where the model is fundamentally an "uncertain guesser," regardless of the softmax confidence.

**How to use:**
```python
advisor = InterpretationAdvisor(tracker)
advisor.report()
tracker.plot.mechanistic("xai_dashboard.png")
```

---

### [Phase 3] 🛠️ Strategic Intervention (Auto-Fix)
GradTracer is **Adaptive.** It doesn't just watch; it intervenes. 
- **Auto-Fix:** When an embedding is identified as a "Zombie," GradTracer dynamically dampens its gradient update to stop the oscillation, allowing other features to catch up.

**How to use:**
```python
emb_tracker.auto_fix = True  # Activates gradient-scaling hooks
```

---

### [Phase 4] 📉 Joint Compression (Efficiency without Sacrifice)
Shrink your model intelligently using **Fisher Information Analysis.**
- Unlike magnitude pruning (which is naive), GradTracer uses **Fisher Saliency** to find which parameters *actually* contribute to the loss optimization.
- **Mixed-Precision:** Automatically leaves sensitive early layers in FP32 while crushing redundant ones to INT8.

**How to use:**
```python
ct = CompressionTracker(model, tracker=tracker)
ct.apply_joint_compression(target_sparsity=0.7)
```

---

### [Phase 5] 🤖 Agent-Ready Reports
GradTracer exports its findings as **Causal JSON.** 
This allows AI Agents (like Cursor or Antigravity) to read the report and say: *"I see layers 2 and 3 are starved. I suggest adding Dropout or stronger Augmentation to break the shortcut."*

---
### 🚀 Quick Start
```bash
python cookbook/full_cycle_demo.py
```
*Look at the generated `.png` and `.json` files to see the results!*
