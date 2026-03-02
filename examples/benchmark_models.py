""\"
GradTracer Model Profiling Benchmark
Benchmarks the overhead of FlowTracker across various famous models.
""\"
import time, torch, torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig
from gradtracer import FlowTracker

models = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "distilroberta-base",
    "albert-base-v2"
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16
seq_len = 128
steps = 30

def benchmark_model(model_name, use_tracker=False):
    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    model = AutoModelForSequenceClassification.from_config(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    if use_tracker:
        tracker = FlowTracker(model, track_gradients=True)
    
    # Dummy data
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), device=device)
    labels = torch.ones((batch_size,), dtype=torch.long, device=device)
    
    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        if use_tracker: tracker.step(outputs.loss.item())
        optimizer.step()
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    for _ in range(steps):
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        if use_tracker: tracker.step(outputs.loss.item())
        optimizer.step()
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    return (time.time() - start) / steps

print("="*60)
print(f"{'Model Name':<30} | {'Native (s)':<12} | {'GradTracer (s)':<12} | {'Overhead'}")
print("-" * 60)

for m in models:
    try:
        t_native = benchmark_model(m, use_tracker=False)
        t_gt = benchmark_model(m, use_tracker=True)
        overhead = (t_gt - t_native) / t_native * 100
        print(f"{m:<30} | {t_native:12.4f} | {t_gt:14.4f} | {overhead:8.2f}%")
    except Exception as e:
        print(f"{m:<30} | FAILED: {str(e)}")

print("="*60)
