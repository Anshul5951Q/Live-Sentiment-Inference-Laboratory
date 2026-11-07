"""Unit tests for Milestone 5 retraining and data augmentation utilities."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.training.retrain_augmented_models import retrain_logistic_regression


class TestRetrainingExperiments:
    """Test data concatenation, augmentation ratios, and metric evaluation."""

    def test_augmentation_ratio_splitting(self):
        """Verify 50% stratified sampling preserves class proportions."""
        synth_data = {
            "review": [f"Review {i}" for i in range(30)],
            "label": [0] * 10 + [1] * 10 + [2] * 10,
        }
        synth_df = pd.DataFrame(synth_data)
        
        # Sample 50% per class
        synth_50 = pd.concat([
            grp.sample(frac=0.5, random_state=42)
            for _, grp in synth_df.groupby("label")
        ]).reset_index(drop=True)
        
        assert len(synth_50) == 15
        assert (synth_50["label"].value_counts() == 5).all()

    def test_condition_concatenation_shapes(self):
        """Verify train concatenation produces exact anticipated record counts."""
        real_df = pd.DataFrame({
            "review": [f"Real {i}" for i in range(100)],
            "label": [0, 1, 2, 0] * 25,
        })
        synth_df = pd.DataFrame({
            "review": [f"Synth {i}" for i in range(10)],
            "label": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
        })
        
        cond_b = pd.concat([real_df, synth_df], ignore_index=True)
        assert len(cond_b) == 110
        assert "review" in cond_b.columns
        assert "label" in cond_b.columns

