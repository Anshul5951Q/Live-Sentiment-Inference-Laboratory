"""Multiclass probability calibration module using CalibratedClassifierCV."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from collections import Counter
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from config.logging_config import setup_logger

logger = setup_logger(__name__)

# Determine modern sklearn FrozenEstimator availability
try:
    from sklearn.frozen import FrozenEstimator
    HAS_FROZEN_ESTIMATOR = True
except ImportError:
    HAS_FROZEN_ESTIMATOR = False


class MulticlassCalibrator:
    """Wraps and calibrates a trained classifier using validation data strictly."""

    def __init__(self, method: str = "sigmoid"):
        if method not in ("sigmoid", "isotonic"):
            raise ValueError(f"Unsupported calibration method: {method}. Choose 'sigmoid' or 'isotonic'.")
        self.method = method
        self.calibrated_model: Optional[CalibratedClassifierCV] = None
        self.classes_: List[int] = [0, 1, 2]
        self.is_calibrated: bool = False

    def fit(
        self,
        base_estimator: Any,
        X_val: Union[sp.csr_matrix, np.ndarray],
        y_val: Union[List[int], np.ndarray]
    ) -> "MulticlassCalibrator":
        """Calibrate the base estimator strictly on validation data.
        
        Leakage guard: X_val and y_val must be from the validation split.
        The evaluation subset (eval_subset_real.parquet) must NEVER be used here.
        """
        y_arr = np.asarray(y_val, dtype=int)
        min_class_count = min(Counter(y_arr).values()) if len(y_arr) > 0 else 5
        cv_folds = min(5, max(2, min_class_count)) if min_class_count >= 2 else None
        
        logger.info(
            "Fitting multiclass '%s' calibration on %d validation samples (cv=%s)...",
            self.method,
            len(y_arr),
            cv_folds
        )
        
        if HAS_FROZEN_ESTIMATOR:
            # scikit-learn >= 1.4 API
            frozen_base = FrozenEstimator(base_estimator)
            self.calibrated_model = CalibratedClassifierCV(
                estimator=frozen_base,
                method=self.method,
                cv=cv_folds
            )
        else:
            # scikit-learn < 1.4 API
            self.calibrated_model = CalibratedClassifierCV(
                estimator=base_estimator,
                method=self.method,
                cv="prefit"
            )
            
        self.calibrated_model.fit(X_val, y_arr)
        self.classes_ = [int(c) for c in self.calibrated_model.classes_]
        self.is_calibrated = True
        logger.info("Multiclass calibration completed successfully.")
        return self

    def predict(self, X: Union[sp.csr_matrix, np.ndarray]) -> np.ndarray:
        """Predict calibrated class labels."""
        if not self.is_calibrated or self.calibrated_model is None:
            raise RuntimeError("Calibrator must be fitted before predict.")
        return self.calibrated_model.predict(X)

    def predict_proba(self, X: Union[sp.csr_matrix, np.ndarray]) -> np.ndarray:
        """Predict calibrated probabilities.
        
        Validates that row probabilities sum to 1.0 within tolerance.
        """
        if not self.is_calibrated or self.calibrated_model is None:
            raise RuntimeError("Calibrator must be fitted before predict_proba.")
            
        probs = self.calibrated_model.predict_proba(X)
        row_sums = np.sum(probs, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            raise ValueError(f"Calibrated probabilities do not sum to 1.0. Range: [{row_sums.min()}, {row_sums.max()}]")
            
        return probs

    def save(self, file_path: Union[str, Path]) -> Path:
        """Save the calibrated model to disk."""
        if not self.is_calibrated or self.calibrated_model is None:
            raise RuntimeError("Cannot save an unfitted calibrator.")
            
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "model_name": "CalibratedClassifierCV",
            "method": self.method,
            "classes_": self.classes_,
            "calibrated_model": self.calibrated_model,
            "is_calibrated": self.is_calibrated,
        }
        joblib.dump(state, path)
        logger.info("Saved MulticlassCalibrator to %s", path)
        return path

    def load(self, file_path: Union[str, Path]) -> "MulticlassCalibrator":
        """Load a calibrated model from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Calibrator file not found at {path}")
            
        state = joblib.load(path)
        self.method = state.get("method", "sigmoid")
        self.classes_ = state.get("classes_", [0, 1, 2])
        self.calibrated_model = state["calibrated_model"]
        self.is_calibrated = state.get("is_calibrated", True)
        logger.info("Loaded MulticlassCalibrator from %s", path)
        return self
