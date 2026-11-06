"""Unit tests for configuration validation and directories."""
from pathlib import Path
from config.settings import settings


class TestConfig:
    """Test suite for application settings."""

    def test_settings_initialization(self):
        assert settings.random_seed == 42
        assert settings.total_real_samples == 24000
        assert settings.samples_per_class == 8000
        assert settings.eval_subset_samples_per_class == 1000
        assert settings.synthetic_total_target == 500

    def test_label_mappings_consistency(self):
        for label, sentiment in settings.label_to_sentiment.items():
            assert settings.sentiment_to_label[sentiment] == label
            
        assert settings.label_to_sentiment[0] == "Negative"
        assert settings.label_to_sentiment[1] == "Neutral"
        assert settings.label_to_sentiment[2] == "Positive"

    def test_ensure_directories(self, tmp_path):
        custom_data_dir = tmp_path / "data"
        custom_models_dir = tmp_path / "models"
        custom_results_dir = tmp_path / "results"
        
        settings.data_dir = custom_data_dir
        settings.models_dir = custom_models_dir
        settings.results_dir = custom_results_dir
        
        settings.ensure_directories()
        assert custom_data_dir.exists()
        assert custom_models_dir.exists()
        assert custom_results_dir.exists()

