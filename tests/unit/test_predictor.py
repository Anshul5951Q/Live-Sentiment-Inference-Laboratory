"""Unit tests for SentimentPredictor inference service."""
import pytest
from src.models.tfidf_vectorizer import TextTfidfVectorizer
from src.models.logistic_regression import LogisticRegressionClassifier
from src.models.calibration import MulticlassCalibrator
from src.models.predictor import SentimentPredictor


class TestSentimentPredictor:
    """Test suite for SentimentPredictor."""

    @pytest.fixture
    def fitted_predictor(self):
        train_texts = [
            "Horrible disgusting meal and awful service.",
            "Worst dinner ever, terrible food.",
            "Average lunch, mediocre pizza.",
            "It was okay, ordinary experience.",
            "Outstanding dining, fantastic steak and superb ambiance!",
            "Delicious pasta, exceptional customer service!"
        ]
        train_labels = [0, 0, 1, 1, 2, 2]
        
        vec = TextTfidfVectorizer(max_features=50, ngram_range=(1, 2), min_df=1)
        X_train = vec.fit_transform(train_texts)
        
        clf = LogisticRegressionClassifier(max_iter=200, random_state=42)
        clf.fit(X_train, train_labels)
        
        cal = MulticlassCalibrator(method="sigmoid")
        cal.fit(clf.model, X_train, train_labels)
        
        return SentimentPredictor(
            vectorizer=vec,
            uncalibrated_model=clf,
            calibrated_model=cal,
            metadata={"version": "1.0.0"}
        )

    def test_single_review_prediction(self, fitted_predictor):
        result = fitted_predictor.predict("Delicious pizza and great service!")
        
        assert isinstance(result, dict)
        assert result["predicted_label"] in (0, 1, 2)
        assert result["predicted_sentiment"] in ("Negative", "Neutral", "Positive")
        assert "confidence" in result
        assert "calibrated_probabilities" in result
        assert "uncalibrated_probabilities" in result
        assert result["is_calibrated_output"] is True
        
        # Verify probability sums to 1.0
        cal_probs = list(result["calibrated_probabilities"].values())
        assert sum(cal_probs) == pytest.approx(1.0, abs=1e-4)

    def test_batch_review_prediction(self, fitted_predictor):
        batch = [
            "Terrible place, avoid!",
            "It was fine.",
            "Incredible food, loved it!"
        ]
        results = fitted_predictor.predict(batch)
        
        assert isinstance(results, list)
        assert len(results) == 3
        for res in results:
            assert res["predicted_label"] in (0, 1, 2)
            assert res["predicted_sentiment"] in ("Negative", "Neutral", "Positive")

    def test_empty_string_handling(self, fitted_predictor):
        # Empty and whitespace reviews should not crash
        result = fitted_predictor.predict("   \t\n  ")
        assert isinstance(result, dict)
        assert result["predicted_label"] in (0, 1, 2)
