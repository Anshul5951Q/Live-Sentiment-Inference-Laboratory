"""Unit tests for synthetic review generation, validation, and checkpointing."""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from config.settings import settings
from src.llm.synthetic_review_validator import SyntheticReviewValidator
from src.llm.synthetic_review_generator import (
    BaseSyntheticGeneratorProvider,
    GeminiSyntheticProvider,
    SyntheticReviewGenerator,
)


class TestSyntheticReviewValidator:
    """Test suite for SyntheticReviewValidator."""

    @pytest.fixture
    def validator(self):
        real_corpus = {"delicious pasta and friendly staff.", "worst diner in the city."}
        return SyntheticReviewValidator(real_review_texts=real_corpus, min_char_length=20, max_char_length=500)

    def test_valid_review_passes(self, validator):
        """Test a clean valid review record passes validation."""
        rec = {
            "review": "The grilled salmon was fresh, seasoned properly, and served quickly.",
            "sentiment": "Positive",
            "label": 2,
        }
        is_valid, reason = validator.validate_record(rec)
        assert is_valid is True
        assert reason is None

    def test_empty_or_short_review_rejected(self, validator):
        """Test empty or too short review text is rejected."""
        rec = {"review": "Too short", "sentiment": "Negative", "label": 0}
        is_valid, reason = validator.validate_record(rec)
        assert is_valid is False
        assert "too short" in reason.lower()

    def test_unfilled_template_rejected(self, validator):
        """Test unfilled template placeholders like [Restaurant] are rejected."""
        rec = {
            "review": "I visited [Restaurant Name] and ordered the [Dish], but it was subpar.",
            "sentiment": "Negative",
            "label": 0,
        }
        is_valid, reason = validator.validate_record(rec)
        assert is_valid is False
        assert "placeholder" in reason.lower()

    def test_sentiment_label_mismatch_rejected(self, validator):
        """Test mismatched sentiment string and numeric label are rejected."""
        rec = {
            "review": "The ambiance was calm and the service was reasonably attentive.",
            "sentiment": "Neutral",
            "label": 2,  # 2 is Positive, expected 1
        }
        is_valid, reason = validator.validate_record(rec)
        assert is_valid is False
        assert "does not match" in reason.lower()

    def test_duplicate_within_synthetic_batch_rejected(self, validator):
        """Test identical synthetic reviews in the same run are rejected."""
        rec1 = {
            "review": "Authentic homemade gelato with rich pistachio flavor.",
            "sentiment": "Positive",
            "label": 2,
        }
        rec2 = {
            "review": "authentic homemade gelato with rich pistachio flavor.",
            "sentiment": "Positive",
            "label": 2,
        }
        val1, _ = validator.validate_record(rec1)
        val2, reason = validator.validate_record(rec2)
        assert val1 is True
        assert val2 is False
        assert "duplicate" in reason.lower()

    def test_exact_overlap_with_real_yelp_rejected(self, validator):
        """Test collision with real Yelp training dataset is rejected."""
        rec = {
            "review": "Worst diner in the city.",
            "sentiment": "Negative",
            "label": 0,
        }
        is_valid, reason = validator.validate_record(rec)
        assert is_valid is False
        assert "existing real yelp review" in reason.lower()

    def test_secondary_validation_catches_semantic_contradiction(self, validator):
        """Test secondary validation flags contradiction (e.g. Negative review with extreme superlatives)."""
        rec = {
            "review": "An absolute superb culinary masterpiece of perfection.",
            "sentiment": "Negative",
            "label": 0,
        }
        is_valid, reason = validator.validate_record(rec)
        assert is_valid is False
        assert "secondary validation failed" in reason.lower()


class MockSyntheticProvider(BaseSyntheticGeneratorProvider):
    """Deterministic mock synthetic provider for testing."""

    def __init__(self):
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "MockProvider"

    @property
    def model_name(self) -> str:
        return "mock-llm-v1"

    def generate_batch(self, batch_size, target_sentiment, target_label, aspect_topic):
        self.call_count += 1
        return [
            {
                "review": f"Unique {target_sentiment} customer review number {self.call_count}_{i} about {aspect_topic[:15]} with genuine feedback.",
                "sentiment": target_sentiment,
                "label": target_label,
                "source": "synthetic_mock",
                "generation_model": self.model_name,
                "generation_timestamp": "2026-11-06T22:00:00Z",
            }
            for i in range(batch_size)
        ]


class TestSyntheticReviewGenerator:
    """Test suite for SyntheticReviewGenerator orchestrator."""

    def test_checkpoint_and_resumption(self, tmp_path):
        """Test generator correctly saves checkpoint and resumes progress."""
        mock_provider = MockSyntheticProvider()
        validator = SyntheticReviewValidator(min_char_length=15)
        
        # Generator with small batch size
        gen = SyntheticReviewGenerator(
            provider=mock_provider,
            validator=validator,
            output_dir=tmp_path,
            batch_size=5,
            rate_limit_delay=0.0,
        )
        # Override target counts for fast testing
        gen.TARGET_COUNTS = {"Negative": 5, "Neutral": 5, "Positive": 5}
        gen.TOTAL_TARGET = 15
        
        meta = gen.generate_all(max_batches_per_class=5)
        assert meta["is_complete"] is True
        assert meta["total_accepted"] == 15
        assert (tmp_path / "checkpoint_synthetic_generation.json").exists()
        assert (tmp_path / "synthetic_reviews.parquet").exists()
        
        # Second generator instance resumes from checkpoint
        gen2 = SyntheticReviewGenerator(
            provider=mock_provider,
            validator=validator,
            output_dir=tmp_path,
            batch_size=5,
            rate_limit_delay=0.0,
        )
        gen2.TARGET_COUNTS = {"Negative": 5, "Neutral": 5, "Positive": 5}
        gen2.TOTAL_TARGET = 15
        
        counts = gen2.get_current_counts()
        assert counts["Negative"] == 5
        assert counts["Neutral"] == 5
        assert counts["Positive"] == 5
        assert len(gen2.accepted_records) == 15
