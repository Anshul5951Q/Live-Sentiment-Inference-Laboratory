"""Unit tests for classification and calibration metrics calculation."""
import pytest
import numpy as np
from src.evaluation.classification_metrics import (
    compute_multiclass_brier_score,
    compute_expected_calibration_error,
    evaluate_sentiment_model,
)


class TestClassificationMetrics:
    """Test suite for classification metrics."""

    def test_brier_score_perfect_prediction(self):
        # 3 samples, perfect 1.0 confidence for each true label [0, 1, 2]
        y_true = np.array([0, 1, 2])
        y_prob = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        # Perfect predictions should have Brier score 0.0
        bs = compute_multiclass_brier_score(y_true, y_prob)
        assert bs == 0.0

    def test_brier_score_worst_prediction(self):
        # 3 samples, completely inverted wrong predictions
        y_true = np.array([0, 1, 2])
        y_prob = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        # (1-0)^2 + (0-1)^2 = 2 for each sample -> mean is 2.0
        bs = compute_multiclass_brier_score(y_true, y_prob)
        assert bs == 2.0

    def test_expected_calibration_error(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_prob = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7]
        ])
        ece = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
        assert 0.0 <= ece <= 1.0

    def test_evaluate_sentiment_model_complete(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 2, 2]  # 5 correct, 1 wrong (sample 4 predicted 2 instead of 1)
        y_prob = [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.1, 0.8],
            [0.9, 0.05, 0.05],
            [0.1, 0.4, 0.5],
            [0.1, 0.2, 0.7]
        ]
        
        metrics = evaluate_sentiment_model(y_true, y_pred, y_prob)
        
        assert "accuracy" in metrics
        assert metrics["accuracy"] == pytest.approx(5 / 6)
        assert "macro_f1" in metrics
        assert "confusion_matrix" in metrics
        assert len(metrics["confusion_matrix"]) == 3
        assert len(metrics["confusion_matrix"][0]) == 3
        assert "brier_score" in metrics
        assert "log_loss" in metrics
        assert "ece" in metrics
        assert "confidence_stats" in metrics
        assert "reliability_curves" in metrics

