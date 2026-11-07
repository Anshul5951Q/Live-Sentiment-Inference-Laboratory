"""Integration tests for the complete Live Sentiment Lab inference flow across all architectures (Phase 3 & 5)."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from config.settings import settings
from src.models.tfidf_vectorizer import TextTfidfVectorizer
from src.models.logistic_regression import LogisticRegressionClassifier
from src.models.calibration import MulticlassCalibrator
from src.models.text_tokenizer import TextTokenizer
from src.models.lstm_classifier import KerasBiLSTMClassifier
from src.llm.gemini_provider import GeminiProvider
from src.models.common_interface import ModelPredictionResult


class TestLiveSentimentLabFlow:
    """Test every model architecture and input variation in the Live Sentiment Lab."""

    @pytest.fixture(scope="class")
    def sample_reviews(self):
        return {
            "positive": "The artisanal wood-fired pizza was absolutely magnificent! Fresh basil and lovely hospitality.",
            "negative": "Waited over forty minutes for cold pasta and undercooked chicken. Terrible experience.",
            "neutral": "The food arrived within twenty minutes. Prices are typical for downtown dining. Acceptable lunch.",
            "ambiguous": "Oh wonderful, another place that charges twenty dollars for tap water and warm salad.",
            "very_long": "This is a detailed review. " * 50,
        }

    # 1. LOGISTIC REGRESSION (CALIBRATED & UNCALIBRATED)
    def test_logistic_regression_inference_flow(self, sample_reviews):
        """Test Logistic Regression feature extraction, raw predictions, and calibrated probabilities."""
        vec = TextTfidfVectorizer().load(settings.models_dir / "logistic_regression" / "tfidf_vectorizer.joblib")
        lr = LogisticRegressionClassifier().load(settings.models_dir / "logistic_regression" / "logistic_regression.joblib")
        cal = MulticlassCalibrator().load(settings.models_dir / "logistic_regression" / "calibrated_logistic_regression.joblib")

        for sentiment_type, text in sample_reviews.items():
            X = vec.transform([text])
            assert X.shape[1] == 10000

            # Raw uncalibrated
            raw_p = lr.predict_proba(X)[0]
            assert len(raw_p) == 3
            assert np.isclose(np.sum(raw_p), 1.0, atol=1e-4)
            pred_uncal = int(np.argmax(raw_p))
            assert pred_uncal in (0, 1, 2)

            # Calibrated
            cal_p = cal.predict_proba(X)[0]
            assert len(cal_p) == 3
            assert np.isclose(np.sum(cal_p), 1.0, atol=1e-4)
            pred_cal = int(np.argmax(cal_p))
            assert pred_cal in (0, 1, 2)

        # Ground truth checks
        X_pos = vec.transform([sample_reviews["positive"]])
        assert np.argmax(cal.predict_proba(X_pos)[0]) == 2  # Positive

        X_neg = vec.transform([sample_reviews["negative"]])
        assert np.argmax(cal.predict_proba(X_neg)[0]) == 0  # Negative

    # 2. BIDIRECTIONAL LSTM
    def test_bilstm_inference_flow(self, sample_reviews):
        """Test BiLSTM tokenization, shape transformation, and posterior predictions."""
        tok = TextTokenizer().load(settings.models_dir / "lstm" / "tokenizer.json")
        clf = KerasBiLSTMClassifier().load(settings.models_dir / "lstm" / "lstm_model.keras")

        for sentiment_type, text in sample_reviews.items():
            seq = tok.transform([text])
            assert seq.shape == (1, settings.lstm_max_sequence_length)

            probs = clf.predict_proba(seq)[0]
            assert len(probs) == 3
            assert np.isclose(np.sum(probs), 1.0, atol=1e-4)
            pred = int(np.argmax(probs))
            assert pred in (0, 1, 2)

        # Polarity checks
        seq_pos = tok.transform([sample_reviews["positive"]])
        assert np.argmax(clf.predict_proba(seq_pos)[0]) == 2

        seq_neg = tok.transform([sample_reviews["negative"]])
        assert np.argmax(clf.predict_proba(seq_neg)[0]) == 0

    # 3. GEMINI ZERO-SHOT & FEW-SHOT
    def test_gemini_zero_and_few_shot_flow(self, sample_reviews):
        """Test Gemini classification under zero-shot and few-shot modes with mocked response."""
        mock_resp_pos = '{"sentiment": "Positive", "label": 2, "confidence": 0.98, "reason": "Praise for pizza."}'
        mock_resp_neg = '{"sentiment": "Negative", "label": 0, "confidence": 0.96, "reason": "Cold pasta and long wait."}'

        provider = GeminiProvider(api_key="test_key_abc", model_name="gemini-3.5-flash-lite")

        # Zero-shot test
        with patch.object(provider, "_call_api_with_backoff", return_value=mock_resp_pos):
            res_zs = provider.classify_sentiment(sample_reviews["positive"], mode="zero_shot")
            assert res_zs.predicted_label == 2
            assert res_zs.predicted_sentiment == "Positive"
            assert res_zs.confidence == 0.98
            assert res_zs.error_status is None

        # Few-shot test
        with patch.object(provider, "_call_api_with_backoff", return_value=mock_resp_neg):
            res_fs = provider.classify_sentiment(sample_reviews["negative"], mode="few_shot")
            assert res_fs.predicted_label == 0
            assert res_fs.predicted_sentiment == "Negative"
            assert res_fs.confidence == 0.96
            assert res_fs.error_status is None

    # 4. ERROR HANDLING & RESILIENCE
    def test_gemini_error_handling_resilience(self):
        """Test provider handles API exceptions and rate limits cleanly without crashing."""
        provider = GeminiProvider(api_key="test_key_abc", model_name="gemini-3.5-flash-lite")

        with patch.object(provider, "_call_api_with_backoff", side_effect=RuntimeError("429 Resource Exhausted")):
            res = provider.classify_sentiment("Test review text", mode="zero_shot")
            assert res.error_status is not None
            assert "429" in res.error_status or "Resource Exhausted" in res.error_status
            assert res.predicted_label == 1  # Safe fallback
            assert res.confidence == 0.0

    def test_gemini_malformed_json_fallback(self):
        """Test provider safely handles invalid/malformed JSON returned by an LLM."""
        provider = GeminiProvider(api_key="test_key_abc", model_name="gemini-3.5-flash-lite")

        with patch.object(provider, "_call_api_with_backoff", return_value="Non-JSON raw string"):
            res = provider.classify_sentiment("Test review text", mode="zero_shot")
            assert res.error_status is not None
            assert res.predicted_label == 1
            assert res.predicted_sentiment == "Neutral"

