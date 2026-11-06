"""Integration tests for YelpDataLoader with mock Hugging Face datasets."""
import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.data.loader import YelpDataLoader
from src.utils.file_io import load_json, load_dataframe


class TestYelpDataLoaderIntegration:
    """Test full dataset pipeline with mocked HF streaming/download."""

    @pytest.fixture
    def mock_hf_train_df(self):
        """Create mock train pool with 90 samples (30 per 0-indexed rating 0..4)."""
        data = []
        for rating in range(5):
            for i in range(30):
                data.append({
                    "text": f"This is train review #{i} with rating {rating} stars.",
                    "label": rating
                })
        return pd.DataFrame(data)

    @pytest.fixture
    def mock_hf_test_df(self):
        """Create mock test pool with 50 samples (10 per rating 0..4)."""
        data = []
        for rating in range(5):
            for i in range(10):
                data.append({
                    "text": f"This is test review #{i} with rating {rating} stars.",
                    "label": rating
                })
        return pd.DataFrame(data)

    def test_loader_end_to_end_mocked(self, monkeypatch, tmp_path, mock_hf_train_df, mock_hf_test_df):
        loader = YelpDataLoader(data_dir=tmp_path)
        
        # Monkeypatch fetch_hf_dataset so no network is consumed
        def mock_fetch(split: str):
            if split == "train":
                return mock_hf_train_df
            elif split == "test":
                return mock_hf_test_df
            raise ValueError(f"Unknown split: {split}")
            
        monkeypatch.setattr(loader, "fetch_hf_dataset", mock_fetch)
        
        # Override settings for small test scale
        from config.settings import settings
        monkeypatch.setattr(settings, "samples_per_class", 15)  # 15 * 3 = 45 total subsample
        monkeypatch.setattr(settings, "train_split_ratio", 0.8) # 36 train, 9 val
        monkeypatch.setattr(settings, "eval_subset_samples_per_class", 5) # 15 eval subset
        
        splits, stats = loader.prepare_dataset(force_reload=True)
        
        # Assert returned splits structure
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        assert "eval_subset" in splits
        
        # Check lengths
        assert len(splits["train"]) == 36
        assert len(splits["val"]) == 9
        assert len(splits["eval_subset"]) == 15
        
        # Verify persistence on disk
        assert (tmp_path / "train_real.parquet").exists()
        assert (tmp_path / "val_real.parquet").exists()
        assert (tmp_path / "test_real.parquet").exists()
        assert (tmp_path / "eval_subset_real.parquet").exists()
        assert (tmp_path / "eval_subset_metadata.json").exists()
        assert (tmp_path / "dataset_stats.json").exists()
        
        # Verify cached load
        cached_splits = loader.load_cached_splits()
        assert len(cached_splits["train"]) == 36
        
        # Verify metadata contents
        meta = load_json(tmp_path / "eval_subset_metadata.json")
        assert meta["samples_per_class"] == 5
        assert meta["total_eval_samples"] == 15
        assert len(meta["selected_original_indices"]) == 15
