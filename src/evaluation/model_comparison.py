"""Comparative evaluation pipeline benchmarking Logistic Regression, BiLSTM, and Gemini."""
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from config.settings import settings
from config.logging_config import setup_logger
from src.evaluation.classification_metrics import evaluate_sentiment_model
from src.utils.file_io import save_json, save_dataframe, load_dataframe

logger = setup_logger(__name__)


def generate_comparison_plots(
    comparison_results: Dict[str, Any],
    output_dir: Path
) -> List[Path]:
    """Generate publication-quality charts for the four-way model benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots = []
    
    models = list(comparison_results.keys())
    model_labels = [m.replace("_", " ") for m in models]
    colors = ["#3498db", "#9b59b6", "#e67e22", "#2ecc71"]
    
    # 1. Accuracy & Macro-F1 Comparison Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    accuracies = [comparison_results[m]["accuracy"] for m in models]
    macro_f1s = [comparison_results[m]["macro_f1"] for m in models]
    
    bars1 = ax1.bar(model_labels, accuracies, color=colors, width=0.55)
    ax1.set_title("Overall Accuracy on 300-Sample Benchmark", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Accuracy", fontsize=10)
    ax1.set_ylim([0.0, 1.05])
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
                     
    bars2 = ax2.bar(model_labels, macro_f1s, color=colors, width=0.55)
    ax2.set_title("Macro F1-Score on 300-Sample Benchmark", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Macro F1", fontsize=10)
    ax2.set_ylim([0.0, 1.05])
    ax2.grid(axis="y", linestyle="--", alpha=0.6)
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
                     
    fig.autofmt_xdate(rotation=15)
    plt.tight_layout()
    acc_f1_path = output_dir / "accuracy_and_f1_comparison.png"
    plt.savefig(acc_f1_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_plots.append(acc_f1_path)
    
    # 2. Per-Class F1 Comparison (Grouped Bar Chart)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    class_names = ["Negative", "Neutral", "Positive"]
    x = np.arange(len(class_names))
    width = 0.20
    
    for i, m in enumerate(models):
        per_class = comparison_results[m]["per_class"]
        f1_vals = [per_class[c]["f1_score"] for c in class_names]
        rects = ax.bar(x + (i - 1.5) * width, f1_vals, width, label=model_labels[i], color=colors[i % len(colors)])
        for r in rects:
            h = r.get_height()
            ax.annotate(f"{h:.2f}", xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
                        
    ax.set_title("Per-Class F1-Score Comparison Across Sentiment Classes", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=10, fontweight="bold")
    ax.set_ylabel("F1 Score", fontsize=10)
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    per_class_path = output_dir / "per_class_f1_comparison.png"
    plt.savefig(per_class_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_plots.append(per_class_path)
    
    # 3. 2x2 Grid of Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), dpi=300)
    axes_flat = axes.flatten()
    
    for i, m in enumerate(models):
        ax = axes_flat[i]
        cm = np.asarray(comparison_results[m]["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar=False
        )
        acc = comparison_results[m]["accuracy"]
        f1 = comparison_results[m]["macro_f1"]
        ax.set_title(f"{model_labels[i]}\nAccuracy: {acc:.3f} | Macro F1: {f1:.3f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        
    for j in range(len(models), len(axes_flat)):
        axes_flat[j].axis("off")
        
    plt.suptitle("Confusion Matrices on 300-Sample Benchmark (100 per Class)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrices_all.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_plots.append(cm_path)
    
    # 4. Latency Comparison Chart
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    latencies = [comparison_results[m].get("latency_per_sample_ms", 0.0) for m in models]
    bars = ax.bar(model_labels, latencies, color=colors, width=0.5)
    ax.set_title("Average Inference Latency per Sample (Milliseconds)", fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel("Latency (ms / sample)", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.1f} ms", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
                     
    fig.autofmt_xdate(rotation=15)
    plt.tight_layout()
    lat_path = output_dir / "latency_comparison.png"
    plt.savefig(lat_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_plots.append(lat_path)
    
    logger.info("Saved 4 comparative benchmark charts to %s", output_dir)
    return generated_plots


def run_four_way_comparison(
    lr_predictor: Any,
    lstm_classifier: Any,
    lstm_tokenizer: Any,
    gemini_zero_shot_df: Optional[pd.DataFrame] = None,
    gemini_few_shot_df: Optional[pd.DataFrame] = None,
    benchmark_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Execute four-way comparison pipeline across LR, BiLSTM, and Gemini."""
    results_dir = settings.results_dir / "model_comparison"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if benchmark_df is None:
        benchmark_df = load_dataframe(settings.data_dir / "gemini_benchmark_300.parquet")
        
    texts = benchmark_df["review"].tolist()
    y_true = benchmark_df["label"].values.tolist()
    n_samples = len(texts)
    
    logger.info("Running direct four-way benchmark on %d identical samples...", n_samples)
    
    comparison_summary: Dict[str, Any] = {}
    
    # 1. Logistic Regression Evaluation on 300 benchmark
    lr_start = time.time()
    lr_preds = lr_predictor.predict(texts, use_calibrated=True)
    lr_duration = time.time() - lr_start
    
    lr_y_pred = [p["predicted_label"] for p in lr_preds]
    lr_probs = np.array([
        [p["calibrated_probabilities"]["Negative"],
         p["calibrated_probabilities"]["Neutral"],
         p["calibrated_probabilities"]["Positive"]]
        for p in lr_preds
    ])
    lr_metrics = evaluate_sentiment_model(y_true, lr_y_pred, y_prob=lr_probs)
    lr_metrics["latency_per_sample_ms"] = float((lr_duration / n_samples) * 1000.0)
    comparison_summary["Logistic_Regression"] = lr_metrics
    
    # 2. BiLSTM Evaluation on 300 benchmark
    lstm_start = time.time()
    X_lstm = lstm_tokenizer.transform(texts)
    lstm_probs = lstm_classifier.predict_proba(X_lstm)
    lstm_y_pred = np.argmax(lstm_probs, axis=1)
    lstm_duration = time.time() - lstm_start
    
    lstm_metrics = evaluate_sentiment_model(y_true, lstm_y_pred, y_prob=lstm_probs)
    lstm_metrics["latency_per_sample_ms"] = float((lstm_duration / n_samples) * 1000.0)
    comparison_summary["Bidirectional_LSTM"] = lstm_metrics
    
    # 3. Gemini Zero-Shot
    if gemini_zero_shot_df is not None and len(gemini_zero_shot_df) > 0:
        zs_pred = gemini_zero_shot_df["predicted_label"].values.tolist()
        zs_metrics = evaluate_sentiment_model(y_true, zs_pred, y_prob=None)
        zs_metrics["latency_per_sample_ms"] = float(gemini_zero_shot_df["inference_duration_ms"].mean())
        comparison_summary["Gemini_Zero_Shot"] = zs_metrics
        
    # 4. Gemini Few-Shot
    if gemini_few_shot_df is not None and len(gemini_few_shot_df) > 0:
        fs_pred = gemini_few_shot_df["predicted_label"].values.tolist()
        fs_metrics = evaluate_sentiment_model(y_true, fs_pred, y_prob=None)
        fs_metrics["latency_per_sample_ms"] = float(gemini_few_shot_df["inference_duration_ms"].mean())
        comparison_summary["Gemini_Few_Shot"] = fs_metrics
        
    # Save comparison metrics JSON
    save_json(comparison_summary, results_dir / "comparison_metrics.json")
    
    # Generate publication charts
    generate_comparison_plots(comparison_summary, results_dir)
    
    logger.info("Four-way comparative benchmark completed successfully.")
    return comparison_summary
