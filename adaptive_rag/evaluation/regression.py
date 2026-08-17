from typing import Dict, Any
import json
import os

class RegressionTester:
    """Tracks token efficiency and quality metrics across different versions of the system."""
    
    def __init__(self, history_file: str = "./data/regression_history.json"):
        self.history_file = history_file
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)

    def save_run(self, version_name: str, metrics: Dict[str, Any]):
        history = self.load_history()
        history[version_name] = metrics
        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=4)

    def load_history(self) -> Dict[str, Any]:
        if os.path.exists(self.history_file):
            with open(self.history_file, "r") as f:
                return json.load(f)
        return {}

    def compare_versions(self, base_version: str, new_version: str) -> Dict[str, Any]:
        """Ensures token efficiency is never optimized at the expense of answer quality."""
        history = self.load_history()
        if base_version not in history or new_version not in history:
            raise ValueError("Requested versions not found in regression history.")
        
        base = history[base_version]
        new = history[new_version]

        token_savings = base.get("avg_tokens_per_query", 0) - new.get("avg_tokens_per_query", 0)
        recall_change = new.get("recall_at_5", 0) - base.get("recall_at_5", 0)

        return {
            "token_efficiency_improvement": token_savings,
            "recall_change": round(recall_change, 3),
            "quality_maintained": recall_change >= -0.05,
            "regression_detected": recall_change < -0.05
        }
