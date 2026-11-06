"""Unit tests for synthetic review quality analysis and reporting."""
import pytest
import pandas as pd
from pathlib import Path

from src.evaluation.synthetic_data_analyzer import (
    compute_text_length_stats,
    compute_near_duplicates,
    analyze_synthetic_dataset,
)
from src.utils.file_io import save_dataframe


class TestSyntheticDataAnalyzer:
    """Test suite for synthetic review data analyzer."""

    def test_text_length_stats(self):
        """Test calculation of character and word length statistics."""
        texts = [
            "Short review here.",
            "This is a somewhat longer review with more detailed feedback about food quality.",
        ]
        stats = compute_text_length_stats(texts)
        assert "character_length" in stats
        assert "word_length" in stats
        assert stats["character_length"]["min"] == len(texts[0])
        assert stats["character_length"]["max"] == len(texts[1])
        assert stats["word_length"]["min"] == 3
        assert stats["word_length"]["max"] == 13

    def test_compute_near_duplicates_detection(self):
        """Test near-duplicate estimation on overlapping texts."""
        texts = [
            "The pizza was wonderfully hot and the crust was perfectly crisp and cheesy.",
            "The pizza was wonderfully hot and the crust was very crisp and cheesy.",  # Near duplicate
            "An utterly dreadful experience with cold food and rude waiters.",  # Dissimilar
        ]
        near_dups = compute_near_duplicates(texts, threshold=0.70)
        assert near_dups >= 1

    def test_analyze_synthetic_dataset_end_to_end(self, tmp_path):
        """Test analyze_synthetic_dataset generates reports and plots."""
        # Create a mock synthetic dataset
        data = {
            "review": [
                f"Generated customer review number {i} for dining experience."
                for i in range(30)
            ],
            "sentiment": ["Negative"] * 10 + ["Neutral"] * 10 + ["Positive"] * 10,
            "label": [0] * 10 + [1] * 10 + [2] * 10,
            "source": ["synthetic_gemini"] * 30,
        }
        df = pd.DataFrame(data)
        synth_file = tmp_path / "synthetic_reviews.parquet"
        save_dataframe(df, synth_file)
        
        report = analyze_synthetic_dataset(synthetic_parquet_path=synth_file, output_dir=tmp_path)
        assert report["status"] == "COMPLETED"
        assert report["total_records"] == 30
        assert report["class_distribution"]["Negative"] == 10
        assert (tmp_path / "synthetic_quality_report.json").exists()
        assert (tmp_path / "synthetic_class_distribution.png").exists()
        assert (tmp_path / "synthetic_length_distribution.png").exists()

