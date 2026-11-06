"""CLI script to execute the Logistic Regression training and calibration pipeline."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.logging_config import setup_logger
from src.training.train_logistic_regression import run_logistic_regression_pipeline

logger = setup_logger("train_logistic_regression_cli")


def main():
    """Run pipeline and display formatted summary."""
    results = run_logistic_regression_pipeline()
    
    exp_meta = results["experiment_metadata"]
    eval_res = results["evaluation_results"]
    
    print("\n" + "=" * 65)
    print("LOGISTIC REGRESSION BASELINE: TRAINING & EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Experiment ID:     {exp_meta['experiment_id']}")
    print(f"Timestamp:         {exp_meta['timestamp']}")
    print(f"Train Samples:     {exp_meta['dataset']['train_samples']}")
    print(f"Val Samples:       {exp_meta['dataset']['val_samples']}")
    print(f"Eval Samples:      {exp_meta['dataset']['eval_subset_samples']}")
    print(f"Total Duration:    {exp_meta['timings_seconds']['total_pipeline_duration']} seconds")
    print("-" * 65)
    
    uncal = eval_res["uncalibrated"]
    cal = eval_res["calibrated"]
    comp = eval_res["comparison"]
    
    print(f"{'Metric':<25} | {'Uncalibrated':<15} | {'Calibrated':<15} | {'Delta':<10}")
    print("-" * 65)
    print(f"{'Accuracy':<25} | {uncal['accuracy']:<15.4f} | {cal['accuracy']:<15.4f} | {comp['accuracy_delta']:+.4f}")
    print(f"{'Macro F1':<25} | {uncal['macro_f1']:<15.4f} | {cal['macro_f1']:<15.4f} | {comp['macro_f1_delta']:+.4f}")
    print(f"{'Weighted F1':<25} | {uncal['weighted_f1']:<15.4f} | {cal['weighted_f1']:<15.4f} | {cal['weighted_f1'] - uncal['weighted_f1']:+.4f}")
    print(f"{'Brier Score (lower better)':<25} | {uncal['brier_score']:<15.4f} | {cal['brier_score']:<15.4f} | {-comp['brier_improvement']:+.4f}")
    print(f"{'Log Loss (lower better)':<25} | {uncal['log_loss']:<15.4f} | {cal['log_loss']:<15.4f} | {-comp['log_loss_improvement']:+.4f}")
    print(f"{'ECE (lower better)':<25} | {uncal['ece']:<15.4f} | {cal['ece']:<15.4f} | {-comp['ece_improvement']:+.4f}")
    print("-" * 65)
    print("\nPER-CLASS EVALUATION (CALIBRATED MODEL):")
    for cls_name, cls_stats in cal["per_class"].items():
        print(f"  {cls_name:<10} -> Precision: {cls_stats['precision']:.4f} | Recall: {cls_stats['recall']:.4f} | F1: {cls_stats['f1_score']:.4f} | Support: {cls_stats['support']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
