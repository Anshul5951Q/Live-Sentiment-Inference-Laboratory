"""Unit tests for dataset stratification, splitting, evaluation subsetting, and leakage checks."""
import pytest
import pandas as pd
from src.data.splitter import (
    create_stratified_subsample,
    split_train_validation,
    create_evaluation_subset,
    verify_no_leakage,
)


@pytest.fixture
def mock_dataset_pool() -> pd.DataFrame:
    """Create a mock pool of 300 samples (100 per class)."""
    rows = []
    for label in (0, 1, 2):
        sentiment = {0: "Negative", 1: "Neutral", 2: "Positive"}[label]
        for i in range(100):
            rows.append({
                "review": f"Review {sentiment} sample number {i}",
                "label": label,
                "sentiment": sentiment,
                "source": "real"
            })
    return pd.DataFrame(rows)


class TestSplitter:
    """Test suite for dataset splitting and stratification."""

    def test_create_stratified_subsample(self, mock_dataset_pool):
        subsample = create_stratified_subsample(mock_dataset_pool, samples_per_class=30, seed=42)
        assert len(subsample) == 90
        counts = subsample["label"].value_counts().to_dict()
        assert counts == {0: 30, 1: 30, 2: 30}

    def test_create_stratified_subsample_insufficient_samples(self, mock_dataset_pool):
        with pytest.raises(ValueError, match="only has 100 samples"):
            create_stratified_subsample(mock_dataset_pool, samples_per_class=150, seed=42)

    def test_split_train_validation(self, mock_dataset_pool):
        # 100 per class -> 300 total
        train_df, val_df = split_train_validation(mock_dataset_pool, train_ratio=0.8, seed=42)
        
        # 80 train per class = 240 train, 20 val per class = 60 val
        assert len(train_df) == 240
        assert len(val_df) == 60
        
        train_counts = train_df["label"].value_counts().to_dict()
        val_counts = val_df["label"].value_counts().to_dict()
        assert train_counts == {0: 80, 1: 80, 2: 80}
        assert val_counts == {0: 20, 1: 20, 2: 20}

    def test_create_evaluation_subset(self, mock_dataset_pool):
        eval_subset, meta = create_evaluation_subset(mock_dataset_pool, samples_per_class=20, seed=42)
        
        assert len(eval_subset) == 60
        assert meta["samples_per_class"] == 20
        assert meta["total_eval_samples"] == 60
        assert meta["class_distribution"] == {0: 20, 1: 20, 2: 20}
        assert len(meta["selected_original_indices"]) == 60

    def test_verify_no_leakage_clean(self, mock_dataset_pool):
        train_df, val_df = split_train_validation(mock_dataset_pool, train_ratio=0.8, seed=42)
        
        # Create a mock separate test set
        test_rows = [
            {"review": f"Completely unique test review {i}", "label": i % 3, "sentiment": "Test", "source": "real"}
            for i in range(30)
        ]
        test_df = pd.DataFrame(test_rows)
        eval_subset = test_df.iloc[:15].copy()
        
        # Should execute cleanly without error
        stats = verify_no_leakage(train_df, val_df, test_df, eval_subset)
        assert stats["train_val_overlap"] == 0
        assert stats["train_test_overlap"] == 0
        assert stats["val_test_overlap"] == 0

    def test_verify_no_leakage_detects_leak(self, mock_dataset_pool):
        train_df, val_df = split_train_validation(mock_dataset_pool, train_ratio=0.8, seed=42)
        
        # Deliberately inject a leaked sample from train into test
        leaked_review = train_df.iloc[0]["review"]
        test_df = pd.DataFrame([{"review": leaked_review, "label": 0, "sentiment": "Negative", "source": "real"}])
        
        with pytest.raises(ValueError, match="Data leakage detected"):
            verify_no_leakage(train_df, val_df, test_df)

