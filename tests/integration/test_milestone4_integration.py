"""Integration tests for Milestone 4: Synthetic generation, quality audit, and 5-way model comparison."""
import pytest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
from pathlib import Path

from config.settings import settings
from src.llm.synthetic_review_generator import (
    BaseSyntheticGeneratorProvider,
    SyntheticReviewGenerator,
)
from src.llm.synthetic_review_validator import SyntheticReviewValidator
from src.evaluation.synthetic_data_analyzer import analyze_synthetic_dataset
from src.evaluation.model_comparison import run_four_way_comparison, generate_comparison_plots
from src.models.common_interface import ModelPredictionResult
from src.utils.file_io import save_dataframe, load_dataframe


class FastDeterministicMockProvider(BaseSyntheticGeneratorProvider):
    """Generates unique deterministic reviews for 500-sample validation testing."""

    def __init__(self):
        self.call_idx = 0

    @property
    def provider_name(self) -> str:
        return "Google Gemini (Academic Substitution for GPT)"

    @property
    def model_name(self) -> str:
        return "gemini-1.5-flash"

    def generate_batch(self, batch_size, target_sentiment, target_label, aspect_topic):
        self.call_idx += 1
        return [
            {
                "review": f"Detailed {target_sentiment} customer experience {self.call_idx}_{i}: The cuisine and service in this establishment were notable.",
                "sentiment": target_sentiment,
                "label": target_label,
                "source": "synthetic_gemini",
                "generation_model": self.model_name,
                "generation_timestamp": "2026-11-06T22:30:00Z",
            }
            for i in range(batch_size)
        ]


class TestMilestone4Integration:
    """Integration suite for Milestone 4 workflows."""

    def test_full_500_synthetic_generation_and_quality_audit(self, tmp_path):
        """Test full synthetic review generation pipeline produces exactly 500 validated records."""
        provider = FastDeterministicMockProvider()
        validator = SyntheticReviewValidator(min_char_length=20)
        
        generator = SyntheticReviewGenerator(
            provider=provider,
            validator=validator,
            output_dir=tmp_path,
            batch_size=50,
            rate_limit_delay=0.0,
        )
        # Standard target: 167 Negative, 167 Neutral, 166 Positive = 500 Total
        meta = generator.generate_all(max_batches_per_class=10)
        
        assert meta["is_complete"] is True
        assert meta["total_accepted"] == 500
        assert meta["counts"]["Negative"] == 167
        assert meta["counts"]["Neutral"] == 167
        assert meta["counts"]["Positive"] == 166
        
        parquet_path = tmp_path / "synthetic_reviews.parquet"
        assert parquet_path.exists()
        df = load_dataframe(parquet_path)
        assert len(df) == 500
        assert set(df["label"].unique()) == {0, 1, 2}
        
        # Run quality audit on generated dataset
        quality_report = analyze_synthetic_dataset(
            synthetic_parquet_path=parquet_path,
            output_dir=tmp_path / "analysis"
        )
        assert quality_report["status"] == "COMPLETED"
        assert quality_report["total_records"] == 500
        assert quality_report["exact_duplicates_within_synthetic"] == 0
        assert (tmp_path / "analysis" / "synthetic_quality_report.json").exists()

    def test_five_way_model_comparison_pipeline(self, tmp_path):
        """Test 5-model comparison pipeline handles LR, BiLSTM, BERT, and Gemini."""
        benchmark_df = pd.DataFrame({
            "review": [f"Review sample number {i}" for i in range(30)],
            "label": [0] * 10 + [1] * 10 + [2] * 10,
        })
        
        # 1. Mock LR Predictor
        mock_lr = MagicMock()
        mock_lr.predict.return_value = [
            {
                "predicted_label": 0 if i < 10 else (1 if i < 20 else 2),
                "calibrated_probabilities": {"Negative": 0.8, "Neutral": 0.1, "Positive": 0.1},
            }
            for i in range(30)
        ]
        
        # 2. Mock BiLSTM
        mock_lstm_clf = MagicMock()
        mock_lstm_clf.predict_proba.return_value = np.array([
            [0.7, 0.2, 0.1] if i < 10 else ([0.1, 0.8, 0.1] if i < 20 else [0.1, 0.2, 0.7])
            for i in range(30)
        ])
        mock_lstm_tok = MagicMock()
        mock_lstm_tok.transform.return_value = np.zeros((30, 128))
        
        # 3. Mock BERT Predictor
        mock_bert = MagicMock()
        mock_bert.predict_batch.return_value = [
            ModelPredictionResult(
                model_name="BERT_base_uncased",
                model_version="1.0.0",
                review=f"Review sample number {i}",
                predicted_label=0 if i < 10 else (1 if i < 20 else 2),
                predicted_sentiment="Negative" if i < 10 else ("Neutral" if i < 20 else "Positive"),
                confidence=0.85,
                class_probabilities={"Negative": 0.85, "Neutral": 0.10, "Positive": 0.05},
            )
            for i in range(30)
        ]
        
        # 4. Mock Gemini Zero-Shot and Few-Shot dataframes
        zs_df = pd.DataFrame({
            "predicted_label": [0] * 10 + [1] * 10 + [2] * 10,
            "inference_duration_ms": [1200.0] * 30,
        })
        fs_df = pd.DataFrame({
            "predicted_label": [0] * 10 + [1] * 10 + [2] * 10,
            "inference_duration_ms": [1400.0] * 30,
        })
        
        # Run comparison
        summary = run_four_way_comparison(
            lr_predictor=mock_lr,
            lstm_classifier=mock_lstm_clf,
            lstm_tokenizer=mock_lstm_tok,
            bert_predictor=mock_bert,
            gemini_zero_shot_df=zs_df,
            gemini_few_shot_df=fs_df,
            benchmark_df=benchmark_df,
        )
        
        assert "Logistic_Regression" in summary
        assert "Bidirectional_LSTM" in summary
        assert "HuggingFace_BERT" in summary
        assert "Gemini_Zero_Shot" in summary
        assert "Gemini_Few_Shot" in summary
        assert len(summary) == 5
        
        # Verify plots generated for 5 models
        plot_files = generate_comparison_plots(summary, tmp_path, benchmark_size=30)
        assert len(plot_files) == 4
        for pf in plot_files:
            assert pf.exists()
