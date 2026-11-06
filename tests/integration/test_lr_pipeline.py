"""Integration test for Logistic Regression training and evaluation pipeline."""
import pytest
import pandas as pd
from unittest.mock import patch
from src.training.train_logistic_regression import run_logistic_regression_pipeline
from src.models.predictor import SentimentPredictor


class TestLogisticRegressionPipelineIntegration:
    """Test full training, calibration, and prediction pipeline with mock datasets."""

    @pytest.fixture
    def mock_splits(self, tmp_path):
        train_rows = []
        for l in (0, 1, 2):
            for i in range(25):
                train_rows.append({
                    "review": f"Train review #{i} for sentiment class {l}",
                    "label": l,
                    "sentiment": {0: "Negative", 1: "Neutral", 2: "Positive"}[l],
                    "source": "real"
                })
                
        val_rows = []
        for l in (0, 1, 2):
            for i in range(10):
                val_rows.append({
                    "review": f"Val review #{i} for sentiment class {l}",
                    "label": l,
                    "sentiment": {0: "Negative", 1: "Neutral", 2: "Positive"}[l],
                    "source": "real"
                })
                
        eval_rows = []
        for l in (0, 1, 2):
            for i in range(10):
                eval_rows.append({
                    "review": f"Evaluation review #{i} for sentiment class {l}",
                    "label": l,
                    "sentiment": {0: "Negative", 1: "Neutral", 2: "Positive"}[l],
                    "source": "real"
                })
                
        data_dir = tmp_path / "data"
        models_dir = tmp_path / "models"
        results_dir = tmp_path / "results"
        data_dir.mkdir(parents=True)
        models_dir.mkdir(parents=True)
        results_dir.mkdir(parents=True)
        
        pd.DataFrame(train_rows).to_parquet(data_dir / "train_real.parquet")
        pd.DataFrame(val_rows).to_parquet(data_dir / "val_real.parquet")
        pd.DataFrame(eval_rows).to_parquet(data_dir / "eval_subset_real.parquet")
        
        return data_dir, models_dir, results_dir

    def test_pipeline_execution_and_predictor_roundtrip(self, mock_splits, monkeypatch):
        data_dir, models_dir, results_dir = mock_splits
        
        from config.settings import settings
        monkeypatch.setattr(settings, "data_dir", data_dir)
        monkeypatch.setattr(settings, "models_dir", models_dir)
        monkeypatch.setattr(settings, "results_dir", results_dir)
        monkeypatch.setattr(settings, "tfidf_max_features", 50)
        
        # Execute pipeline
        results = run_logistic_regression_pipeline()
        
        # Assert artifacts on disk
        lr_models_dir = models_dir / "logistic_regression"
        lr_results_dir = results_dir / "logistic_regression"
        
        assert (lr_models_dir / "tfidf_vectorizer.joblib").exists()
        assert (lr_models_dir / "logistic_regression.joblib").exists()
        assert (lr_models_dir / "calibrated_logistic_regression.joblib").exists()
        assert (lr_models_dir / "model_metadata.json").exists()
        
        assert (lr_results_dir / "validation_metrics.json").exists()
        assert (lr_results_dir / "evaluation_metrics.json").exists()
        assert (lr_results_dir / "predictions.parquet").exists()
        assert (lr_results_dir / "experiment_metadata.json").exists()
        assert (lr_results_dir / "confusion_matrix_uncalibrated.png").exists()
        assert (lr_results_dir / "confusion_matrix_calibrated.png").exists()
        assert (lr_results_dir / "calibration_uncalibrated.png").exists()
        assert (lr_results_dir / "calibration_calibrated.png").exists()
        assert (lr_results_dir / "metrics_comparison.png").exists()
        assert (lr_results_dir / "confidence_distribution.png").exists()
        
        # Load through predictor and verify inference roundtrip
        predictor = SentimentPredictor.from_artifacts(lr_models_dir)
        pred = predictor.predict("A completely novel test sentence.")
        assert pred["predicted_label"] in (0, 1, 2)
        assert pred["confidence"] > 0
