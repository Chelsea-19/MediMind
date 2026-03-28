from typing import List

class EvaluationMetrics:
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
        """Calculate Recall@K where K is len(retrieved_ids)."""
        if not ground_truth_ids:
            return 0.0
        hits = set(retrieved_ids).intersection(set(ground_truth_ids))
        return len(hits) / len(ground_truth_ids)
    
    @staticmethod
    def exact_match(prediction: str, ground_truth: str) -> float:
        """Placeholder for exact string match on specific attributes."""
        return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0

    @staticmethod
    def evaluate_triage_accuracy(predicted_triage: str, true_triage: str) -> float:
        """Binary accuracy for triage classification."""
        # Simple exact match on label names
        return 1.0 if predicted_triage.strip().lower() == true_triage.strip().lower() else 0.0

# Experiment logging abstraction
class ExperimentLogger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        import os
        os.makedirs(log_dir, exist_ok=True)
        
    def log_result(self, experiment_name: str, results: dict, detailed_records: List[dict] = None):
        import json
        from pathlib import Path
        import time
        
        timestamp = int(time.time())
        file_path = Path(self.log_dir) / f"{experiment_name}_{timestamp}.json"
        
        data = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "aggregate_metrics": results,
            "details": detailed_records or []
        }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
