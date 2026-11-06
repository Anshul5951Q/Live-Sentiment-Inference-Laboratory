"""Unit tests for Logistic Regression classification component."""
import pytest
import numpy as np
import scipy.sparse as sp
from src.models.logistic_regression import LogisticRegressionClassifier


class TestLogisticRegressionClassifier:
    """Test suite for LogisticRegressionClassifier."""

    @pytest.fixture
    def mock_training_data(self):
        np.random.seed(42)
        X = sp.csr_matrix(np.random.randn(90, 20))
        # 30 for each class 0, 1, 2
        y = np.repeat([0, 1, 2], 30)
        return X, y

    def test_training_and_classes(self, mock_training_data):
        X, y = mock_training_data
        clf = LogisticRegressionClassifier(C=1.0, max_iter=200, random_state=42)
        clf.fit(X, y)
        
        assert clf.is_trained
        assert clf.classes_ == [0, 1, 2]
        assert clf.n_features_in_ == 20

    def test_prediction_shapes_and_probabilities(self, mock_training_data):
        X, y = mock_training_data
        clf = LogisticRegressionClassifier(C=1.0, max_iter=200, random_state=42)
        clf.fit(X, y)
        
        preds = clf.predict(X[:10])
        probs = clf.predict_proba(X[:10])
        
        assert preds.shape == (10,)
        assert set(preds).issubset({0, 1, 2})
        assert probs.shape == (10, 3)
        
        # Assert probabilities sum to 1.0 within float tolerance
        row_sums = np.sum(probs, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_save_and_load(self, mock_training_data, tmp_path):
        X, y = mock_training_data
        clf = LogisticRegressionClassifier(C=1.0, max_iter=200, random_state=42)
        clf.fit(X, y)
        
        model_path = tmp_path / "lr.joblib"
        clf.save(model_path)
        
        loaded = LogisticRegressionClassifier().load(model_path)
        assert loaded.is_trained
        assert loaded.classes_ == [0, 1, 2]
        
        # Assert prediction parity
        np.testing.assert_array_equal(clf.predict(X[:5]), loaded.predict(X[:5]))
        np.testing.assert_allclose(clf.predict_proba(X[:5]), loaded.predict_proba(X[:5]), atol=1e-5)
