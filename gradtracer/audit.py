import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

AUDIT_DIR = ".gradtracer"
AUDIT_FILE = "audit.jsonl"


class AutoFixAuditLogger:
    """
    Transparently logs every gradient intervention made by AutoFix.
    Solves the 'Reproducibility Nightmare' by ensuring no magic happens in a black box.
    """
    
    def __init__(self, run_id: str = "default_run"):
        self.run_id = run_id
        self._ensure_dir()
        self.path = os.path.join(AUDIT_DIR, AUDIT_FILE)
        
        # Memory buffer for fast reporting
        self.interventions: List[Dict[str, Any]] = []

    def _ensure_dir(self):
        if not os.path.exists(AUDIT_DIR):
            try:
                os.makedirs(AUDIT_DIR)
            except OSError:
                pass

    def log_intervention(self, step: int, layer_name: str, intervention_type: str, indices: List[int], scale: float):
        """
        Record an intervention event.
        """
        if not indices:
            return
            
        event = {
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "step": step,
            "layer": layer_name,
            "type": intervention_type,
            "num_affected": len(indices),
            "scale_applied": scale,
            "indices_sample": indices[:10]  # Only log up to 10 for brevity in file
        }
        
        self.interventions.append(event)
        
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass
            
    def summary(self) -> Dict[str, Any]:
        """Return a summary of all interventions in this run."""
        stats = {}
        for ev in self.interventions:
            key = ev["type"]
            stats[key] = stats.get(key, 0) + ev["num_affected"]
        
        return {
            "total_events": len(self.interventions),
            "affected_by_type": stats
        }
