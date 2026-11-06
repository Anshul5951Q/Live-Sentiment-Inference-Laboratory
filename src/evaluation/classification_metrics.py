"""Comprehensive classification and probability calibration metrics."""
from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from config.settings import settings


def compute_multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the standard multi-class Brier score:
    
    BS = (1/N) * sum_{i=1}^N sum_{k=1}^K (p_{ik} - y_{ik})^2
    where y_{ik} is 1 if instance i has true class k, and 0 otherwise.
    Range: [0, 2] for 3-class; lower is better (0 = perfect calibration).
    """
    n_samples, n_classes = y_prob.shape
    y_true_one_hot = np.zeros((n_samples, n_classes), dtype=float)
    y_true_one_hot[np.arange(n_samples), y_true] = 1.0
    
    squared_errors = np.sum((y_prob - y_true_one_hot) ** 2, axis=1)
    return float(np.mean(squared_errors))


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) on top-1 confidence predictions."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        count = np.sum(in_bin)
        
        if count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (count / n_samples) * np.abs(bin_acc - bin_conf)
            
    return float(ece)


def compute_reliability_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """Compute reliability diagram data for top-label confidence and per-class One-vs-Rest."""
    y_t = np.asarray(y_true, dtype=int)
    y_pr = np.asarray(y_prob, dtype=float)
    n_samples, n_classes = y_pr.shape
    
    # 1. Top-label confidence reliability
    confidences = np.max(y_pr, axis=1)
    predictions = np.argmax(y_pr, axis=1)
    accuracies = (predictions == y_t).astype(float)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        in_bin = (confidences > lower) & (confidences <= upper) if i > 0 else (confidences >= lower) & (confidences <= upper)
        count = int(np.sum(in_bin))
        bin_counts.append(count)
        if count > 0:
            bin_confidences.append(float(np.mean(confidences[in_bin])))
            bin_accuracies.append(float(np.mean(accuracies[in_bin])))
        else:
            bin_confidences.append(float((lower + upper) / 2))
            bin_accuracies.append(0.0)
            
    # 2. Per-class One-vs-Rest curves
    per_class_curves = {}
    class_names = ["Negative", "Neutral", "Positive"]
    for k in range(n_classes):
        binary_true = (y_t == k).astype(int)
        class_prob = y_pr[:, k]
        prob_true, prob_pred = calibration_curve(binary_true, class_prob, n_bins=n_bins, strategy="uniform")
        per_class_curves[class_names[k]] = {
            "mean_predicted_prob": [float(p) for p in prob_pred],
            "fraction_positives": [float(p) for p in prob_true],
        }
        
    return {
        "top_label": {
            "mean_confidence": bin_confidences,
            "accuracy": bin_accuracies,
            "counts": bin_counts,
            "bin_edges": [float(b) for b in bin_boundaries],
        },
        "per_class": per_class_curves,
    }


def compute_confidence_statistics(y_prob: np.ndarray) -> Dict[str, Any]:
    """Calculate descriptive statistics and distribution for prediction confidences."""
    confidences = np.max(y_prob, axis=1)
    
    # Compute histogram
    hist_counts, bin_edges = np.histogram(confidences, bins=10, range=(0.0, 1.0))
    
    return {
        "mean_confidence": float(np.mean(confidences)),
        "median_confidence": float(np.median(confidences)),
        "std_confidence": float(np.std(confidences)),
        "min_confidence": float(np.min(confidences)),
        "max_confidence": float(np.max(confidences)),
        "percentiles": {
            "p10": float(np.percentile(confidences, 10)),
            "p25": float(np.percentile(confidences, 25)),
            "p50": float(np.percentile(confidences, 50)),
            "p75": float(np.percentile(confidences, 75)),
            "p90": float(np.percentile(confidences, 90)),
        },
        "histogram": {
            "counts": [int(c) for c in hist_counts],
            "bin_edges": [float(b) for b in bin_edges],
        }
    }


def evaluate_sentiment_model(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_prob: Optional[Union[List[List[float]], np.ndarray]] = None,
    n_bins: int = 10
) -> Dict[str, Any]:
    """Execute complete multi-class sentiment evaluation."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    
    acc = float(accuracy_score(y_t, y_p))
    
    # Macro metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_t, y_p, average="macro", zero_division=0
    )
    # Weighted metrics
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
        y_t, y_p, average="weighted", zero_division=0
    )
    
    # Per-class metrics
    p_class, r_class, f1_class, support = precision_recall_fscore_support(
        y_t, y_p, average=None, labels=[0, 1, 2], zero_division=0
    )
    
    cm = confusion_matrix(y_t, y_p, labels=[0, 1, 2]).tolist()
    report_text = classification_report(
        y_t, y_p, labels=[0, 1, 2],
        target_names=settings.class_names,
        zero_division=0
    )
    
    per_class_summary = {}
    for idx, name in settings.label_to_sentiment.items():
        per_class_summary[name] = {
            "label": idx,
            "precision": float(p_class[idx]),
            "recall": float(r_class[idx]),
            "f1_score": float(f1_class[idx]),
            "support": int(support[idx]),
        }
        
    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_wt),
        "per_class": per_class_summary,
        "confusion_matrix": cm,
        "classification_report_str": report_text,
    }
    
    if y_prob is not None:
        y_pr = np.asarray(y_prob, dtype=float)
        metrics["brier_score"] = compute_multiclass_brier_score(y_t, y_pr)
        metrics["log_loss"] = float(log_loss(y_t, y_pr, labels=[0, 1, 2]))
        metrics["ece"] = compute_expected_calibration_error(y_t, y_pr, n_bins=n_bins)
        metrics["confidence_stats"] = compute_confidence_statistics(y_pr)
        metrics["reliability_curves"] = compute_reliability_data(y_t, y_pr, n_bins=n_bins)
        
    return metrics
