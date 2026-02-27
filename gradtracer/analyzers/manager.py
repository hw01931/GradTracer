"""
FlowManager - Centralized tracking logic for complex architectures.

Supports registering multiple embeddings or layers (e.g., Two-Tower, GNNs)
to calculate cross-layer correlation and global statistics.
"""
from typing import Dict, List, Any
import numpy as np

class FlowManager:
    """
    Manages multiple trackers (e.g., User Tower, Item Tower).
    Combines stats to understand cross-layer dynamics and global shifts.
    """
    def __init__(self, name: str = "flow_manager"):
        self.name = name
        self.trackers = {}
        
    def add_tracker(self, key: str, tracker: Any):
        self.trackers[key] = tracker
        
    def step(self):
        """Steps all active trackers."""
        for name, tracker in self.trackers.items():
            tracker.step()
            
    def summary(self) -> Dict[str, Any]:
        """Summarizes stats across all registered trackers."""
        combined_stats = {}
        total_zombies = 0
        total_dead = 0
        
        for name, tracker in self.trackers.items():
            if hasattr(tracker, "summary"):
                s = tracker.summary()
                combined_stats[name] = s
                # E.g. for EmbeddingTrackers
                if "zombie_count" in s:
                    total_zombies += s["zombie_count"]
                if "dead_count" in s:
                    total_dead += s["dead_count"]
                    
        combined_stats["global_zombies"] = total_zombies
        combined_stats["global_dead"] = total_dead
        return combined_stats

    def report(self):
        print(f"=== FlowManager Report: {self.name} ===")
        s = self.summary()
        for name, stats in s.items():
            if name not in ["global_zombies", "global_dead"]:
                print(f"[{name}] Coverage: {stats.get('coverage_pct', 0):.1f}% | "
                      f"Zombies: {stats.get('zombie_pct', 0):.1f}% | "
                      f"Dead: {stats.get('dead_pct', 0):.1f}%")
        print("=========================================")
