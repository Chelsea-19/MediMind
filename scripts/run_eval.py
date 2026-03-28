import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.metrics import EvaluationMetrics, ExperimentLogger

def main():
    print("Running evaluation skeleton...")
    
    # 1. Mock benchmark dataset (Placeholder for external benchmark data)
    ground_truth_data = [
        {"query": "胸痛，向左臂放射，呼吸急促", "true_triage": "Emergency", "relevant_docs": ["doc_mi_01", "doc_mi_02"]},
        {"query": "持续两天干咳，无发热", "true_triage": "Routine", "relevant_docs": ["doc_cough_1"]},
    ]
    
    # 2. Mock model & retrieval logic results (Assume we ran our full pipeline on the benchmark)
    predictions = [
        {"predicted_triage": "Emergency", "retrieved_docs": ["doc_mi_01", "doc_other"]},
        {"predicted_triage": "Routine", "retrieved_docs": ["doc_cough_1"]}
    ]
    
    logger = ExperimentLogger(log_dir=os.path.join(os.path.dirname(__file__), '..', 'results'))
    
    total_acc = 0.0
    total_recall = 0.0
    detailed_results = []
    
    for gt, pred in zip(ground_truth_data, predictions):
        triage_acc = EvaluationMetrics.evaluate_triage_accuracy(pred["predicted_triage"], gt["true_triage"])
        recall = EvaluationMetrics.recall_at_k(pred["retrieved_docs"], gt["relevant_docs"])
        
        total_acc += triage_acc
        total_recall += recall
        
        detailed_results.append({
            "query": gt["query"],
            "triage_acc": triage_acc,
            "recall@k": recall
        })
    
    avg_acc = total_acc / len(ground_truth_data)
    avg_recall = total_recall / len(ground_truth_data)
    
    print(f"Eval Summary - Triage Acc: {avg_acc:.2f}, Recall@K: {avg_recall:.2f}")
    
    logger.log_result("baseline_eval", {
        "triage_accuracy": avg_acc,
        "recall_at_k": avg_recall,
        "n_samples": len(ground_truth_data)
    }, detailed_results)

if __name__ == "__main__":
    main()
