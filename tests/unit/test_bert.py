"""Unit tests for Hugging Face BERT classifier and predictor interfaces."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from config.settings import settings
from src.models.bert_classifier import (
    BertSentimentClassifier,
    BertSentimentPredictor,
    get_pytorch_device,
)
from src.models.common_interface import ModelPredictionResult


class TestBertClassifier:
    """Test suite for BERT classifier component."""

    def test_device_detection(self):
        """Test PyTorch device detection returns valid device type and description."""
        dev_type, dev_desc = get_pytorch_device()
        assert dev_type in ("cuda", "cpu", "blocked", "none")
        assert isinstance(dev_desc, str)
        assert len(dev_desc) > 0

    def test_tokenization_shapes_and_keys(self):
        """Test tokenization generates correct tensor shapes and keys."""
        clf = BertSentimentClassifier(max_length=64)
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": np.zeros((2, 64)),
            "attention_mask": np.ones((2, 64)),
        }
        clf.tokenizer = mock_tokenizer
        
        texts = ["Outstanding meal!", "Terrible service."]
        encodings = clf.tokenize_texts(texts)
        assert "input_ids" in encodings
        assert "attention_mask" in encodings
        assert encodings["input_ids"].shape == (2, 64)

    def test_predict_proba_normalization_mocked(self):
        """Test that predict_proba returns row sums equal to 1.0."""
        clf = BertSentimentClassifier()
        clf.model = MagicMock()
        clf.tokenizer = MagicMock()
        
        # Mock PyTorch output
        mock_logits = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.logits = mock_logits
        clf.model.return_value = mock_outputs
        
        mock_torch = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.array([
            [0.1, 0.7, 0.2],
            [0.8, 0.1, 0.1],
        ])
        mock_torch.softmax.return_value = mock_tensor
        mock_torch.no_grad.return_value.__enter__.return_value = None
        mock_torch.no_grad.return_value.__exit__.return_value = None
        
        with patch.dict("sys.modules", {"torch": mock_torch}):
            clf.tokenize_texts = MagicMock(return_value={"input_ids": MagicMock()})
            probs = clf.predict_proba(["Review 1", "Review 2"])
            
            assert probs.shape == (2, 3)
            row_sums = np.sum(probs, axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_bert_predictor_single_and_batch(self):
        """Test BertSentimentPredictor returns valid ModelPredictionResult instances."""
        mock_clf = MagicMock()
        mock_clf.device = "cpu"
        mock_clf.predict_proba.return_value = np.array([
            [0.05, 0.15, 0.80],
            [0.70, 0.20, 0.10],
        ])
        
        predictor = BertSentimentPredictor(classifier=mock_clf)
        
        # Test single review
        res_single = predictor.predict_one("Exquisite dinner!")
        assert isinstance(res_single, ModelPredictionResult)
        assert res_single.model_name == "BERT_base_uncased"
        assert res_single.predicted_label == 2
        assert res_single.predicted_sentiment == "Positive"
        assert res_single.confidence == pytest.approx(0.80, abs=1e-4)
        assert res_single.class_probabilities["Positive"] == pytest.approx(0.80, abs=1e-4)
        
        # Test batch reviews
        res_batch = predictor.predict_batch(["Exquisite dinner!", "Terrible service."])
        assert len(res_batch) == 2
        assert res_batch[0].predicted_sentiment == "Positive"
        assert res_batch[1].predicted_sentiment == "Negative"
        assert res_batch[1].predicted_label == 0
