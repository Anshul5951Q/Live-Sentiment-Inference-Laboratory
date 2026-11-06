"""Unit tests for the four-way model comparison pipeline."""
import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.evaluation.model_comparison import generate_comparison_plots, run_four_way_comparison


class TestModelComparison:
    """Test suite for comparative evaluation."""

    @pytest.fixture
    def mock_comparison_data(self):
        return {
            "Logistic_Regression": {
                "accuracy": 0.74,
                "macro_f1": 0.73,
                "per_class": {
                    "Negative": {"f1_score": 0.80},
                    "Neutral": {"f1_score": 0.61},
                    "Positive": {"f1_score": 0.78},
                },
                "confusion_matrix": [[80, 15, 5], [18, 60, 22], [8, 12, 80]],
                "latency_per_sample_ms": 0.5,
            },
            "Bidirectional_LSTM": {
                "accuracy": 0.76,
                "macro_f1": 0.75,
                "per_class": {
                    "Negative": {"f1_score": 0.82},
                    "Neutral": {"f1_score": 0.65},
                    "Positive": {"f1_score": 0.80},
                },
                "confusion_matrix": [[82, 12, 6], [15, 65, 20], [6, 10, 84]],
                "latency_per_sample_ms": 2.1,
            },
            "Gemini_Zero_Shot": {
                "accuracy": 0.78,
                "macro_f1": 0.77,
                "per_class": {
                    "Negative": {"f1_score": 0.85},
                    "Neutral": {"f1_score": 0.68},
                    "Positive": {"f1_score": 0.82},
                },
                "confusion_matrix": [[85, 10, 5], [12, 68, 20], [5, 8, 87]],
                "latency_per_sample_ms": 350.0,
            },
            "Gemini_Few_Shot": {
                "accuracy": 0.81,
                "macro_f1": 0.80,
                "per_class": {
                    "Negative": {"f1_score": 0.87},
                    "Neutral": {"f1_score": 0.73},
                    "Positive": {"f1_score": 0.85},
                },
                "confusion_matrix": [[87, 8, 5], [10, 73, 17], [4, 6, 90]],
                "latency_per_sample_ms": 420.0,
            },
        }

    def test_generate_comparison_plots(self, mock_comparison_data, tmp_path):
        plots = generate_comparison_plots(mock_comparison_data, tmp_path)
        
        # Ensure 4 publication plots are generated
        assert len(plots) == 4
        assert (tmp_path / "accuracy_and_f1_comparison.png").exists()
        assert (tmp_path / "per_class_f1_comparison.png").exists()
        assert (tmp_path / "confusion_matrices_all.png").exists()
        assert (tmp_path / "latency_comparison.png").exists()

