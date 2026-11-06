"""Unit tests for GeminiProvider with mocked API calls and resilient error handling."""
import json
import pytest
from unittest.mock import MagicMock, patch
from src.llm.gemini_provider import GeminiProvider
from src.llm.prompts import format_zero_shot_prompt, format_few_shot_prompt


class TestGeminiProvider:
    """Test suite for GeminiProvider using mocked responses."""

    @pytest.fixture
    def mock_provider(self, tmp_path):
        return GeminiProvider(
            api_key="mock_test_key_xyz123",
            model_name="gemini-1.5-flash",
            cache_dir=tmp_path / "cache",
            max_retries=3,
            backoff_base=0.01,
            backoff_max=0.05,
            rate_limit_delay=0.0
        )

    def test_zero_shot_prompt_formatting(self):
        prompt = format_zero_shot_prompt("The pizza was amazing!")
        assert "The pizza was amazing!" in prompt
        assert "JSON Response:" in prompt

    def test_few_shot_prompt_formatting(self):
        prompt = format_few_shot_prompt("The steak was overcooked.")
        assert "The steak was overcooked." in prompt
        assert "Here are several examples" in prompt
        assert "Negative" in prompt
        assert "Neutral" in prompt
        assert "Positive" in prompt

    def test_clean_json_text_markdown_stripping(self, mock_provider):
        raw_markdown = '```json\n{"sentiment": "Positive", "label": 2, "confidence": 0.95, "reason": "Great!"}\n```'
        cleaned = mock_provider._clean_json_text(raw_markdown)
        parsed = json.loads(cleaned)
        assert parsed["sentiment"] == "Positive"
        assert parsed["label"] == 2

    def test_schema_validation_valid(self, mock_provider):
        raw_data = {
            "sentiment": "Negative",
            "label": 0,
            "confidence": 0.92,
            "reason": "Cold food."
        }
        validated = mock_provider._validate_schema(raw_data)
        assert validated["sentiment"] == "Negative"
        assert validated["label"] == 0
        assert validated["confidence"] == 0.92
        assert validated["reason"] == "Cold food."

    def test_schema_validation_corrects_mismatched_label(self, mock_provider):
        # Mismatched label: sentiment is Positive but label was sent as 0
        raw_data = {
            "sentiment": "Positive",
            "label": 0,
            "confidence": 0.85,
            "reason": "Good service."
        }
        validated = mock_provider._validate_schema(raw_data)
        assert validated["sentiment"] == "Positive"
        # Must be corrected to 2
        assert validated["label"] == 2

    def test_schema_validation_clamps_confidence(self, mock_provider):
        raw_data = {
            "sentiment": "Neutral",
            "label": 1,
            "confidence": 1.5,  # Out of range > 1.0
            "reason": "Average."
        }
        validated = mock_provider._validate_schema(raw_data)
        assert validated["confidence"] == 1.0

    def test_classify_sentiment_success_mocked(self, mock_provider):
        mock_response = '{"sentiment": "Positive", "label": 2, "confidence": 0.94, "reason": "Superb meal."}'
        
        with patch.object(mock_provider, "_call_api_with_backoff", return_value=mock_response):
            res = mock_provider.classify_sentiment("Fantastic experience!", mode="zero_shot")
            
            assert res.predicted_label == 2
            assert res.predicted_sentiment == "Positive"
            assert res.confidence == 0.94
            assert res.class_probabilities is None  # Distinguished from calibrated probabilities
            assert res.error_status is None

    def test_caching_prevents_duplicate_calls(self, mock_provider):
        mock_response = '{"sentiment": "Neutral", "label": 1, "confidence": 0.70, "reason": "Standard food."}'
        
        with patch.object(mock_provider, "_call_api_with_backoff", return_value=mock_response) as mock_api:
            # First call executes API
            res1 = mock_provider.classify_sentiment("Okay meal.", mode="zero_shot")
            assert mock_api.call_count == 1
            
            # Second call on identical text must hit cache
            res2 = mock_provider.classify_sentiment("Okay meal.", mode="zero_shot")
            assert mock_api.call_count == 1  # Not incremented
            assert res2.predicted_label == 1
            assert res2.metadata["source"] == "cache"

    def test_batch_classify_checkpoint_resume(self, mock_provider, tmp_path):
        chk_file = tmp_path / "chk.json"
        
        mock_response = '{"sentiment": "Negative", "label": 0, "confidence": 0.90, "reason": "Awful."}'
        with patch.object(mock_provider, "_call_api_with_backoff", return_value=mock_response):
            texts = ["Bad food 1", "Bad food 2"]
            results1 = mock_provider.batch_classify(texts, mode="zero_shot", checkpoint_path=str(chk_file))
            assert len(results1) == 2
            assert chk_file.exists()
            
            # Re-running with checkpoint should load from checkpoint without calling API
            with patch.object(mock_provider, "_call_api_with_backoff", side_effect=Exception("Should not be called!")):
                results2 = mock_provider.batch_classify(texts, mode="zero_shot", checkpoint_path=str(chk_file))
                assert len(results2) == 2
                assert results2[0].predicted_label == 0

    def test_no_api_key_exposure_in_result(self, mock_provider):
        mock_response = '{"sentiment": "Positive", "label": 2, "confidence": 0.99, "reason": "Top notch."}'
        with patch.object(mock_provider, "_call_api_with_backoff", return_value=mock_response):
            res = mock_provider.classify_sentiment("Great!", mode="zero_shot")
            res_dict = res.to_dict()
            
            # Ensure the secret key does not appear anywhere in output dictionary
            res_str = json.dumps(res_dict)
            assert "mock_test_key_xyz123" not in res_str
