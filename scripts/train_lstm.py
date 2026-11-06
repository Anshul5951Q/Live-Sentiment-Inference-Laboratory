"""CLI script to run the Bidirectional LSTM training and evaluation pipeline."""
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.logging_config import setup_logger
from src.training.train_lstm import run_lstm_pipeline

logger = setup_logger("train_lstm_cli")


def main():
    """Run Keras LSTM pipeline and display summary."""
    results = run_lstm_pipeline()
    
    meta = results["model_metadata"]
    eval_m = results["eval_metrics"]
    history = results["history"]
    
    print("\n" + "=" * 65)
    print("KERAS BIDIRECTIONAL LSTM: TRAINING & EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Device Placement:  {meta['device_info']}")
    print(f"Vocab Size:        {meta['architecture']['vocab_size']} words")
    print(f"Sequence Length:   {meta['architecture']['max_length']} tokens")
    print(f"Epochs Completed:  {meta['training_config']['epochs_completed']} (EarlyStopping)")
    print(f"Training Duration: {meta['timings_seconds']['training_duration']} seconds")
    print(f"Inference Latency: {eval_m.get('inference_latency_per_sample_ms', 0.0):.2f} ms/sample")
    print("-" * 65)
    print(f"{'Metric':<25} | {'Validation Score':<15} | {'Evaluation Subset':<15}")
    print("-" * 65)
    print(f"{'Accuracy':<25} | {results['val_metrics']['accuracy']:<15.4f} | {eval_m['accuracy']:<15.4f}")
    print(f"{'Macro Precision':<25} | {results['val_metrics']['macro_precision']:<15.4f} | {eval_m['macro_precision']:<15.4f}")
    print(f"{'Macro Recall':<25} | {results['val_metrics']['macro_recall']:<15.4f} | {eval_m['macro_recall']:<15.4f}")
    print(f"{'Macro F1':<25} | {results['val_metrics']['macro_f1']:<15.4f} | {eval_m['macro_f1']:<15.4f}")
    print(f"{'Brier Score (lower better)':<25} | {results['val_metrics']['brier_score']:<15.4f} | {eval_m['brier_score']:<15.4f}")
    print(f"{'Log Loss (lower better)':<25} | {results['val_metrics']['log_loss']:<15.4f} | {eval_m['log_loss']:<15.4f}")
    print(f"{'ECE (lower better)':<25} | {results['val_metrics']['ece']:<15.4f} | {eval_m['ece']:<15.4f}")
    print("-" * 65)
    print("\nPER-CLASS EVALUATION (3,000 EVALUATION SUBSET):")
    for cls_name, cls_stats in eval_m["per_class"].items():
        print(f"  {cls_name:<10} -> Precision: {cls_stats['precision']:.4f} | Recall: {cls_stats['recall']:.4f} | F1: {cls_stats['f1_score']:.4f} | Support: {cls_stats['support']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

