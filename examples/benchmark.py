"""
GradTracer Performance Benchmark (EmbeddingTracker)

This script proves that using GradTracer on massive embedding tables
(e.g., 1 Million items, standard for RecSys and large language models)
adds minimal overhead (<5%) when `track_interval` is configured correctly.
"""
import time
import torch
import torch.nn as nn
from gradtracer.analyzers.embedding import EmbeddingTracker

print("====================================================")
print("🚀 GradTracer Performance Overhead Benchmark")
print("====================================================\n")

NUM_ITEMS = 1_000_000
DIM = 64
BATCH_SIZE = 1024
STEPS = 500

print(f"Dataset Size: {NUM_ITEMS:,} items")
print(f"Embedding Dim: {DIM}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Total Steps: {STEPS}\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

def run_benchmark(tracker=None, auto_fix=False):
    model = nn.Embedding(NUM_ITEMS, DIM, sparse=True).to(device)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
    
    if tracker is not None:
        tracker_instance = EmbeddingTracker(model, auto_fix=auto_fix, track_interval=100)
    
    # Warmup
    for _ in range(5):
        indices = torch.randint(0, NUM_ITEMS, (BATCH_SIZE,), device=device)
        optimizer.zero_grad()
        loss = model(indices).sum()
        loss.backward()
        if tracker is not None:
            tracker_instance.step()
        optimizer.step()
        
    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    for _ in range(STEPS):
        indices = torch.randint(0, NUM_ITEMS, (BATCH_SIZE,), device=device)
        optimizer.zero_grad()
        loss = model(indices).sum()
        loss.backward()
        
        if tracker is not None:
            tracker_instance.step()
            
        optimizer.step()
        
    torch.cuda.synchronize() if device.type == 'cuda' else None
    duration = time.time() - start_time
    steps_per_sec = STEPS / duration
    
    if tracker is not None:
        tracker_instance.detach()
        
    return duration, steps_per_sec

# 1. Baseline
base_dur, base_sps = run_benchmark(tracker=None)
print(f"[1] Baseline (Native PyTorch):")
print(f"    - Duration: {base_dur:.3f} s")
print(f"    - Throughput: {base_sps:.1f} steps/s\n")

# 2. GradTracer (Diagnostics Only, track_interval=100)
diag_dur, diag_sps = run_benchmark(tracker=True, auto_fix=False)
diag_overhead = (base_sps - diag_sps) / base_sps * 100
print(f"[2] GradTracer (Diagnostics Only, track_interval=100):")
print(f"    - Duration: {diag_dur:.3f} s")
print(f"    - Throughput: {diag_sps:.1f} steps/s")
print(f"    - Overhead: {diag_overhead:.2f}%\n")

# 3. GradTracer (AutoFix Enabled, track_interval=100)
fix_dur, fix_sps = run_benchmark(tracker=True, auto_fix=True)
fix_overhead = (base_sps - fix_sps) / base_sps * 100
print(f"[3] GradTracer (Auto-Fix Active, track_interval=100):")
print(f"    - Duration: {fix_dur:.3f} s")
print(f"    - Throughput: {fix_sps:.1f} steps/s")
print(f"    - Overhead: {fix_overhead:.2f}%\n")

print("====================================================")
if fix_overhead < 5.0 and diag_overhead < 5.0:
    print("✅ SUCCESS: Tracking overhead is well within acceptable <5% limits.")
else:
    print("⚠️ WARNING: Overhead exceeded typical thresholds. Optimization needed.")
print("====================================================\n")
