"""Unit tests for text preprocessing, HTML cleaning, and label mapping."""
import pytest
import pandas as pd
from src.data.preprocessor import clean_text, map_star_to_three_class, preprocess_dataframe


class TestPreprocessor:
    """Test suite for preprocessing utilities."""

    def test_clean_text_html_unescape(self):
        raw = "Fish &amp; Chips were &lt;tasty&gt;!"
        cleaned = clean_text(raw)
        assert cleaned == "Fish & Chips were <tasty>!"

    def test_clean_text_break_tags(self):
        raw = "First line.<br>Second line.<br />Third line.<BR/>Fourth line."
        cleaned = clean_text(raw)
        assert cleaned == "First line. Second line. Third line. Fourth line."

    def test_clean_text_whitespace(self):
        raw = "   Too   many    spaces   and\nnewlines\t\t "
        cleaned = clean_text(raw)
        assert cleaned == "Too many spaces and newlines"

    def test_clean_text_non_string(self):
        assert clean_text(None) == ""
        assert clean_text(12345) == ""

    @pytest.mark.parametrize("star,expected", [
        (0, 0),  # 0-indexed 1-star -> Negative
        (1, 0),  # 0-indexed 2-star -> Negative
        (2, 1),  # 0-indexed 3-star -> Neutral
        (3, 2),  # 0-indexed 4-star -> Positive
        (4, 2),  # 0-indexed 5-star -> Positive
        (1.0, 0), # 1-star float -> Negative
        (5.0, 2), # 5-star float -> Positive
    ])
    def test_map_star_to_three_class(self, star, expected):
        assert map_star_to_three_class(star) == expected

    def test_map_star_invalid_raises_error(self):
        with pytest.raises(ValueError):
            map_star_to_three_class(7)
        with pytest.raises(ValueError):
            map_star_to_three_class(-1)
        with pytest.raises(ValueError):
            map_star_to_three_class("invalid_string")

    def test_preprocess_dataframe_standardization(self, sample_raw_yelp_df):
        processed = preprocess_dataframe(sample_raw_yelp_df)
        
        # Verify schema
        assert list(processed.columns) == ["review", "label", "sentiment", "source"]
        
        # Check label mapping consistency
        for _, row in processed.iterrows():
            if row["label"] == 0:
                assert row["sentiment"] == "Negative"
            elif row["label"] == 1:
                assert row["sentiment"] == "Neutral"
            elif row["label"] == 2:
                assert row["sentiment"] == "Positive"
            assert row["source"] == "real"

    def test_preprocess_dataframe_dedup_and_empty(self, sample_duplicates_df):
        processed = preprocess_dataframe(sample_duplicates_df)
        
        # Original had 4 rows: 1 duplicate, 1 empty whitespace, 2 unique valid
        assert len(processed) == 2
        assert "Great! Best food ever! & drinks." in processed["review"].values
        assert "Terrible place." in processed["review"].values

