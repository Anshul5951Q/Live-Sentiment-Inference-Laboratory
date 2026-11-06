"""Integration test for Milestone 3: LSTM, Gemini, and 4-way comparative evaluation."""
import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from src.models.text_tokenizer import TextTokenizer
from src.llm.gemini_provider import GeminiProvider
from src.llm.sentiment_classifier import extract_gemini_benchmark_subset
from src.evaluation.model_comparison import run_four_way_comparison
from src.models.common_interface import ModelPredictionResult


class TestMilestone3Integration:
    """End-to-end integration test with mocked components."""

    @pytest.fixture
    def mock_eval_df(self):
        # 60 samples: 20 per class 0, 1, 2
        rows = []
        for l in (0, 1, 2):
            sentiment = {0: "Negative", 1: "Neutral", 2: "Positive"}[l]
            for i in range(20):
                rows.append({
                    "review": f"Review {sentiment} sample #{i}",
                    "label": l,
                    "sentiment": sentiment,
                    "source": "real"
                })
        return pd.DataFrame(rows)

    def test_benchmark_subset_extraction(self, mock_eval_df, tmp_path):
        benchmark_df, meta = extract_gemini_benchmark_subset(
            mock_eval_df,
            samples_per_class=10,
            seed=42,
            output_dir=tmp_path
        )
        assert len(benchmark_df) == 30
        assert meta["samples_per_class"] == 10
        assert (tmp_path / "gemini_benchmark_300.parquet").exists()
        assert (tmp_path / "gemini_benchmark_300_metadata.json").exists()

    def test_mocked_four_way_comparison_pipeline(self, mock_eval_df, tmp_path, monkeypatch):
        from config.settings import settings
        monkeypatch.setattr(settings, "results_dir", tmp_path / "results")
        
        # 1. Mock LR Predictor
        mock_lr = MagicMock()
        mock_lr.predict.return_value = [
            {
                "predicted_label": row["label"],
                "calibrated_probabilities": {
                    "Negative": 0.8 if row["label"] == 0 else 0.1,
                    "Neutral": 0.8 if row["label"] == 1 else 0.1,
                    "Positive": 0.8 if row["label"] == 2 else 0.1,
                }
            }
            for _, row in mock_eval_df.iterrows()
        ]
        
        # 2. Mock LSTM
        mock_lstm = MagicMock()
        mock_lstm.predict_proba.return_value = np.array([
            [0.8 if row["label"] == 0 else 0.1,
             0.8 if row["label"] == 1 else 0.1,
             0.8 if row["label"] == 2 else 0.1]
            for _, row in mock_eval_df.iterrows()
        ])
        mock_tok = MagicMock()
        mock_tok.transform.return_value = np.zeros((len(mock_eval_df), 10))
        
        # 3. Mock Gemini Zero-Shot and Few-Shot DFs
        zs_df = pd.DataFrame({
            "predicted_label": mock_eval_df["label"].values,
            "inference_duration_ms": [150.0] * len(mock_eval_df)
        })
        fs_df = pd.DataFrame({
            "predicted_label": mock_eval_df["label"].values,
            "inference_duration_ms": [180.0] * len(mock_eval_df)
        })
        
        # Run comparison
        summary = run_four_way_comparison(
            lr_predictor=mock_lr,
            lstm_classifier=mock_lstm,
            lstm_tokenizer=mock_tok,
            gemini_zero_shot_df=zs_df,
            gemini_few_shot_df=fs_df,
            benchmark_df=mock_eval_df
        )
        
        assert "Logistic_Regression" in summary
        assert "Bidirectional_LSTM" in summary
        assert "Gemini_Zero_Shot" in summary
        assert "Gemini_Few_Shot" in summary
        
        # Verify plots were saved
        res_dir = tmp_path / "results" / "model_comparison"
        assert (res_dir / "accuracy_and_f1_comparison.png").exists()
        assert (res_dir / "per_class_f1_comparison.png").exists()
        assert (res_dir / "confusion_matrices_all.png").exists()
        assert (res_dir / "latency_comparison.png").exists()
        assert (res_dir / "comparison_metrics.json").exists()

