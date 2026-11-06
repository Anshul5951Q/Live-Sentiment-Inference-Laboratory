"""Calibration evaluation and reliability diagram computation."""
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.calibration import calibration_curve
from src.evaluation.metrics import compute_multiclass_brier_score, compute_expected_calibration_error


def compute_reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """Compute reliability diagram data for top-label and per-class one-vs-rest.
    
    Returns structured data suitable for plotting in Plotly or Matplotlib.
    """
    y_t = np.asarray(y_true)
    y_pr = np.asarray(y_prob)
    
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
            
    # 2. Per-class One-vs-Rest calibration curves
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
        },
        "per_class": per_class_curves,
        "brier_score": compute_multiclass_brier_score(y_t, y_pr),
        "ece": compute_expected_calibration_error(y_t, y_pr, n_bins=n_bins),
    }


def compare_calibration_performance(
    y_true: np.ndarray,
    uncalibrated_prob: np.ndarray,
    calibrated_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """Compare uncalibrated vs calibrated probabilities."""
    uncal_data = compute_reliability_curve(y_true, uncalibrated_prob, n_bins=n_bins)
    cal_data = compute_reliability_curve(y_true, calibrated_prob, n_bins=n_bins)
    
    brier_delta = cal_data["brier_score"] - uncal_data["brier_score"]
    ece_delta = cal_data["ece"] - uncal_data["ece"]
    
    return {
        "uncalibrated": uncal_data,
        "calibrated": cal_data,
        "comparison": {
            "uncalibrated_brier": uncal_data["brier_score"],
            "calibrated_brier": cal_data["brier_score"],
            "brier_improvement": -brier_delta,  # Positive means calibration lowered Brier score (good)
            "uncalibrated_ece": uncal_data["ece"],
            "calibrated_ece": cal_data["ece"],
            "ece_improvement": -ece_delta,      # Positive means calibration lowered ECE (good)
        }
    }
