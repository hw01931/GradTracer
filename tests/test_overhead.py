"""
Verification script for GradTracer overhead and async hooks.
"""
import time
import torch
import torch.nn as nn
from gradtracer.tracker import FlowTracker

def benchmark_overhead():
    print("--- GradTracer Benchmarking (Efficiency Mode) ---")
    
    # Large model to simulate overhead
    model = nn.Sequential(
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10)
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    data = torch.randn(128, 2048)
    target = torch.randn(128, 10)
    
    def run_trial(tracker=None, steps=20):
        start = time.time()
        for i in range(steps):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if tracker:
                tracker.step(loss=loss.item())
        return (time.time() - start) / steps

    # 1. Native Baseline
    t_native = run_trial(steps=100)
    print(f"Native Throughput: {1/t_native:.2f} steps/s")
    
    # 2. GradTracer (Full Mode, Hook Interval 1)
    tracker_full = FlowTracker(model, hook_interval=1)
    t_full = run_trial(tracker_full, steps=100)
    print(f"GradTracer (Full): {1/t_full:.2f} steps/s (Overhead: {(t_full/t_native-1)*100:.2f}%)")
    
    # 3. GradTracer (Efficient Mode, Hook Interval 10)
    tracker_eff = FlowTracker(model, hook_interval=10, track_interval=10)
    t_eff = run_trial(tracker_eff, steps=100)
    print(f"GradTracer (Production, Int=10): {1/t_eff:.2f} steps/s (Overhead: {(t_eff/t_native-1)*100:.2f}%)")
    
    # 4. GradTracer (Light Mode)
    tracker_light = FlowTracker(model, mode='light')
    t_light = run_trial(tracker_light, steps=100)
    print(f"GradTracer (Light): {1/t_light:.2f} steps/s (Overhead: {(t_light/t_native-1)*100:.2f}%)")

if __name__ == "__main__":
    benchmark_overhead()
