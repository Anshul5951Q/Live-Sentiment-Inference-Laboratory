"""Unit tests for MulticlassCalibrator."""
import pytest
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from src.models.calibration import MulticlassCalibrator


class TestMulticlassCalibrator:
    """Test suite for MulticlassCalibrator."""

    @pytest.fixture
    def mock_val_data(self):
        np.random.seed(42)
        X_train = sp.csr_matrix(np.random.randn(90, 10))
        y_train = np.repeat([0, 1, 2], 30)
        
        X_val = sp.csr_matrix(np.random.randn(60, 10))
        y_val = np.repeat([0, 1, 2], 20)
        
        base_lr = LogisticRegression(random_state=42).fit(X_train, y_train)
        return base_lr, X_val, y_val

    def test_calibration_fit_and_predict(self, mock_val_data):
        base_lr, X_val, y_val = mock_val_data
        calibrator = MulticlassCalibrator(method="sigmoid")
        calibrator.fit(base_lr, X_val, y_val)
        
        assert calibrator.is_calibrated
        assert calibrator.classes_ == [0, 1, 2]
        
        preds = calibrator.predict(X_val[:10])
        probs = calibrator.predict_proba(X_val[:10])
        
        assert preds.shape == (10,)
        assert probs.shape == (10, 3)
        np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

    def test_save_and_load(self, mock_val_data, tmp_path):
        base_lr, X_val, y_val = mock_val_data
        calibrator = MulticlassCalibrator(method="sigmoid")
        calibrator.fit(base_lr, X_val, y_val)
        
        save_path = tmp_path / "calibrated_lr.joblib"
        calibrator.save(save_path)
        
        loaded = MulticlassCalibrator().load(save_path)
        assert loaded.is_calibrated
        assert loaded.classes_ == [0, 1, 2]
        
        np.testing.assert_array_equal(calibrator.predict(X_val[:5]), loaded.predict(X_val[:5]))
        np.testing.assert_allclose(calibrator.predict_proba(X_val[:5]), loaded.predict_proba(X_val[:5]), atol=1e-5)

