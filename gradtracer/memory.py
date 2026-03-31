"""
Memory tracker utility — monitors GPU and CPU memory consumption.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

try:
    import psutil
except ImportError:
    psutil = None


def get_cpu_memory_mb() -> float:
    """Returns the current CPU memory usage of the process in MB."""
    if psutil is None:
        return 0.0
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_gpu_memory_info() -> Dict[int, Dict[str, float]]:
    """
    Returns a dictionary of GPU memory stats for all available devices.
    Returns: {device_id: {'allocated_mb': ..., 'reserved_mb': ..., 'peak_mb': ...}}
    """
    stats = {}
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                stats[i] = {
                    "allocated_mb": torch.cuda.memory_allocated(i) / (1024 * 1024),
                    "reserved_mb": torch.cuda.memory_reserved(i) / (1024 * 1024),
                    "peak_mb": torch.cuda.max_memory_allocated(i) / (1024 * 1024),
                }
    except Exception:
        pass
    return stats


class MemoryTracker:
    def __init__(self):
        self.initial_cpu = get_cpu_memory_mb()
        self.peak_cpu = self.initial_cpu
        self.peak_gpu: Dict[int, float] = {}
        
    def update(self):
        """Update peak memory statistics."""
        current_cpu = get_cpu_memory_mb()
        self.peak_cpu = max(self.peak_cpu, current_cpu)
        
        gpu_info = get_gpu_memory_info()
        for dev_id, info in gpu_info.items():
            self.peak_gpu[dev_id] = max(self.peak_gpu.get(dev_id, 0), info['peak_mb'])
            
    def get_summary(self) -> Dict[str, float]:
        """Returns a summary of memory usage."""
        summary = {
            "peak_cpu_mb": self.peak_cpu,
            "delta_cpu_mb": self.peak_cpu - self.initial_cpu,
        }
        for dev_id, peak in self.peak_gpu.items():
            summary[f"peak_gpu_{dev_id}_mb"] = peak
            
        return summary
