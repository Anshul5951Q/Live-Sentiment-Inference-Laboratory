"""Visualization utilities for publication-quality sentiment analysis charts."""
from pathlib import Path
from typing import Any, Dict, List, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger(__name__)

# Configure clean aesthetic
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
CLASS_LABELS = ["Negative", "Neutral", "Positive"]
CLASS_COLORS = {"Negative": "#e74c3c", "Neutral": "#f39c12", "Positive": "#2ecc71"}


def plot_confusion_matrix(
    cm: Union[List[List[int]], np.ndarray],
    title: str,
    output_path: Union[str, Path],
    normalize: bool = False
) -> Path:
    """Plot and save a high-resolution confusion matrix heatmap."""
    cm_arr = np.asarray(cm)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    
    display_data = cm_arr.astype(float) / cm_arr.sum(axis=1)[:, np.newaxis] if normalize else cm_arr
    fmt = ".2f" if normalize else "d"
    
    sns.heatmap(
        display_data,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=True,
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Sentiment Label", fontsize=10, labelpad=8)
    ax.set_ylabel("True Sentiment Label", fontsize=10, labelpad=8)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", path)
    return path


def plot_reliability_diagram(
    reliability_data: Dict[str, Any],
    title: str,
    output_path: Union[str, Path]
) -> Path:
    """Plot multiclass reliability diagram (top-label confidence vs accuracy and per-class)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    # Left subplot: Top-label confidence calibration
    top = reliability_data.get("top_label", {})
    conf = top.get("mean_confidence", [])
    acc = top.get("accuracy", [])
    counts = top.get("counts", [])
    
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.5)
    ax1.plot(conf, acc, "s-", color="#2980b9", label="Empirical Calibration", linewidth=2, markersize=6)
    
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("Mean Predicted Confidence", fontsize=10)
    ax1.set_ylabel("Empirical Accuracy", fontsize=10)
    ax1.set_title("Top-Label Reliability Curve", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right")
    
    # Right subplot: Per-class One-vs-Rest calibration curves
    per_class = reliability_data.get("per_class", {})
    ax2.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.5)
    
    for cls_name, curve_data in per_class.items():
        pred_p = curve_data.get("mean_predicted_prob", [])
        frac_pos = curve_data.get("fraction_positives", [])
        color = CLASS_COLORS.get(cls_name, "#333333")
        ax2.plot(pred_p, frac_pos, "o-", label=f"Class: {cls_name}", color=color, linewidth=1.8, markersize=5)
        
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax2.set_ylabel("True Fraction of Positives", fontsize=10)
    ax2.set_title("Per-Class One-vs-Rest Calibration", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower right")
    
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved reliability diagram to %s", path)
    return path


def plot_metrics_comparison(
    uncal_metrics: Dict[str, Any],
    cal_metrics: Dict[str, Any],
    output_path: Union[str, Path]
) -> Path:
    """Plot bar chart comparing uncalibrated vs calibrated model performance."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    metric_keys = ["accuracy", "macro_f1", "brier_score", "log_loss", "ece"]
    labels = ["Accuracy\n(higher better)", "Macro F1\n(higher better)", "Brier Score\n(lower better)", "Log Loss\n(lower better)", "ECE\n(lower better)"]
    
    uncal_vals = [uncal_metrics.get(k, 0.0) for k in metric_keys]
    cal_vals = [cal_metrics.get(k, 0.0) for k in metric_keys]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    rects1 = ax.bar(x - width/2, uncal_vals, width, label="Uncalibrated Logistic Regression", color="#3498db")
    rects2 = ax.bar(x + width/2, cal_vals, width, label="Calibrated Logistic Regression (Sigmoid)", color="#2ecc71")
    
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Logistic Regression: Uncalibrated vs Calibrated Benchmark", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add value annotations on bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved metrics comparison plot to %s", path)
    return path


def plot_confidence_distribution(
    uncal_stats: Dict[str, Any],
    cal_stats: Dict[str, Any],
    output_path: Union[str, Path]
) -> Path:
    """Plot confidence distribution histogram comparison."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    uncal_hist = uncal_stats.get("histogram", {})
    cal_hist = cal_stats.get("histogram", {})
    
    bin_edges = np.array(uncal_hist.get("bin_edges", np.linspace(0, 1, 11)))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = (bin_edges[1] - bin_edges[0]) * 0.4
    
    ax.bar(bin_centers - width/2, uncal_hist.get("counts", []), width=width, label="Uncalibrated Confidence", color="#3498db", alpha=0.85)
    ax.bar(bin_centers + width/2, cal_hist.get("counts", []), width=width, label="Calibrated Confidence", color="#2ecc71", alpha=0.85)
    
    ax.set_xlabel("Predicted Confidence (Max Probability)", fontsize=10)
    ax.set_ylabel("Number of Samples", fontsize=10)
    ax.set_title("Probability Confidence Distribution on 3,000 Evaluation Subset", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper left")
    ax.set_xlim([0.3, 1.02])
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confidence distribution plot to %s", path)
    return path
