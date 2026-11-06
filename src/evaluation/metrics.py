"""Evaluation metrics calculation for multi-class sentiment analysis."""
from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from config.settings import settings


def compute_multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the multi-class Brier score:
    
    BS = (1/N) * sum_{i=1}^N sum_{k=1}^K (p_{ik} - y_{ik})^2
    Lower is better (0 = perfect calibration).
    """
    n_samples, n_classes = y_prob.shape
    # One-hot encode y_true
    y_true_one_hot = np.zeros((n_samples, n_classes))
    y_true_one_hot[np.arange(n_samples), y_true] = 1.0
    
    squared_diffs = np.sum((y_prob - y_true_one_hot) ** 2, axis=1)
    return float(np.mean(squared_diffs))


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) using top-1 confidence."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Select samples within this confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
            
    return float(ece)


def compute_classification_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_prob: Optional[Union[List[List[float]], np.ndarray]] = None
) -> Dict[str, Any]:
    """Compute comprehensive classification and calibration metrics.
    
    Returns:
        accuracy (float)
        macro_precision (float)
        macro_recall (float)
        macro_f1 (float)
        per_class (Dict): Per-class precision, recall, f1, support
        confusion_matrix (List[List[int]])
        brier_score (Optional[float])
        ece (Optional[float])
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    
    acc = float(accuracy_score(y_t, y_p))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_t, y_p, average="macro", zero_division=0
    )
    
    p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_t, y_p, average=None, labels=[0, 1, 2], zero_division=0
    )
    
    cm = confusion_matrix(y_t, y_p, labels=[0, 1, 2]).tolist()
    
    per_class_dict = {}
    for label_idx, class_name in settings.label_to_sentiment.items():
        per_class_dict[class_name] = {
            "label": label_idx,
            "precision": float(p_per_class[label_idx]),
            "recall": float(r_per_class[label_idx]),
            "f1_score": float(f1_per_class[label_idx]),
            "support": int(support[label_idx]),
        }
        
    metrics = {
        "accuracy": acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "per_class": per_class_dict,
        "confusion_matrix": cm,
    }
    
    if y_prob is not None:
        y_pr = np.asarray(y_prob)
        metrics["brier_score"] = compute_multiclass_brier_score(y_t, y_pr)
        metrics["ece"] = compute_expected_calibration_error(y_t, y_pr)
        
    return metrics
