"""Unit tests for KerasBiLSTMClassifier architecture, shapes, and normalization."""
import pytest
import numpy as np


class TestKerasBiLSTMClassifier:
    """Test suite for KerasBiLSTMClassifier."""

    @pytest.fixture
    def mock_data(self):
        # 30 samples, sequence length 12
        np.random.seed(42)
        X = np.random.randint(0, 50, size=(30, 12), dtype=np.int32)
        y = np.repeat([0, 1, 2], 10)
        return X, y

    def test_model_build_and_shapes(self):
        import tensorflow as tf
        from src.models.lstm_classifier import KerasBiLSTMClassifier
        
        clf = KerasBiLSTMClassifier(
            vocab_size=100,
            max_length=12,
            embedding_dim=16,
            lstm_units=8,
            dense_units=8,
            dropout=0.1
        )
        model = clf.build_model()
        
        assert model is not None
        assert model.input_shape == (None, 12)
        assert model.output_shape == (None, 3)

    def test_training_and_probability_sums(self, mock_data, tmp_path):
        import tensorflow as tf
        from src.models.lstm_classifier import KerasBiLSTMClassifier
        
        X, y = mock_data
        clf = KerasBiLSTMClassifier(
            vocab_size=100,
            max_length=12,
            embedding_dim=16,
            lstm_units=8,
            dense_units=8,
            dropout=0.1
        )
        
        chk_path = tmp_path / "test_model.keras"
        history = clf.train(
            X_train=X,
            y_train=y,
            X_val=X,
            y_val=y,
            batch_size=10,
            epochs=2,
            checkpoint_path=chk_path
        )
        
        assert clf.is_trained
        assert "loss" in history
        assert "val_loss" in history
        
        probs = clf.predict_proba(X[:5])
        assert probs.shape == (5, 3)
        np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-4)

    def test_save_and_load(self, mock_data, tmp_path):
        import tensorflow as tf
        from src.models.lstm_classifier import KerasBiLSTMClassifier
        
        X, y = mock_data
        clf = KerasBiLSTMClassifier(
            vocab_size=100,
            max_length=12,
            embedding_dim=16,
            lstm_units=8,
            dense_units=8
        )
        save_path = tmp_path / "bilstm.keras"
        clf.train(X, y, X, y, batch_size=10, epochs=1, checkpoint_path=save_path)
        
        # Load from disk
        loaded = KerasBiLSTMClassifier(vocab_size=100, max_length=12).load(save_path)
        assert loaded.is_trained
        
        probs1 = clf.predict_proba(X[:3])
        probs2 = loaded.predict_proba(X[:3])
        np.testing.assert_allclose(probs1, probs2, atol=1e-4)

