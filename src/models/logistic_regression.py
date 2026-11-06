"""Logistic Regression classifier component for multi-class sentiment analysis."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import warnings
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger(__name__)


class LogisticRegressionClassifier:
    """Multi-class Logistic Regression sentiment classifier."""

    def __init__(
        self,
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        class_weight: Optional[Union[str, Dict[int, float]]] = None,
        random_state: int = settings.random_seed,
    ):
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        
        self.classes_: List[int] = [0, 1, 2]
        self.model: Optional[LogisticRegression] = None
        self.is_trained: bool = False
        self.n_features_in_: Optional[int] = None

    def fit(
        self,
        X: Union[sp.csr_matrix, np.ndarray],
        y: Union[List[int], np.ndarray]
    ) -> "LogisticRegressionClassifier":
        """Train the Logistic Regression classifier on feature matrix X and target labels y."""
        y_arr = np.asarray(y, dtype=int)
        unique_labels = sorted(list(np.unique(y_arr)))
        if unique_labels != [0, 1, 2]:
            logger.warning(
                "Training labels %s differ from standard 3-class set [0, 1, 2].",
                unique_labels
            )
            
        logger.info(
            "Training LogisticRegression (C=%.3f, solver=%s, max_iter=%d)...",
            self.C,
            self.solver,
            self.max_iter,
        )
        
        self.model = LogisticRegression(
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        
        # Capture and log convergence warnings visibly rather than suppressing
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            self.model.fit(X, y_arr)
            for w in captured_warnings:
                if issubclass(w.category, ConvergenceWarning):
                    logger.warning("LogisticRegression Convergence Warning: %s", w.message)
                    
        self.classes_ = [int(c) for c in self.model.classes_]
        self.n_features_in_ = X.shape[1]
        self.is_trained = True
        logger.info("LogisticRegression fitted successfully across classes %s.", self.classes_)
        return self

    def predict(self, X: Union[sp.csr_matrix, np.ndarray]) -> np.ndarray:
        """Predict class labels [0, 1, 2] for input feature matrix X."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before calling predict.")
            
        return self.model.predict(X)

    def predict_proba(self, X: Union[sp.csr_matrix, np.ndarray]) -> np.ndarray:
        """Predict class probability distributions [P(0), P(1), P(2)].
        
        Validates that each row sums approximately to 1.0.
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before calling predict_proba.")
            
        probs = self.model.predict_proba(X)
        
        # Numerical sanity assertion: rows must sum to 1.0 within floating tolerance
        row_sums = np.sum(probs, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            raise ValueError(f"Predicted probabilities do not sum to 1.0. Range: [{row_sums.min()}, {row_sums.max()}]")
            
        return probs

    def save(self, file_path: Union[str, Path]) -> Path:
        """Save the trained classifier to disk."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Cannot save an untrained model.")
            
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "model_name": "LogisticRegression",
            "C": self.C,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "classes_": self.classes_,
            "n_features_in_": self.n_features_in_,
            "model": self.model,
            "is_trained": self.is_trained,
        }
        joblib.dump(state, path)
        logger.info("Saved LogisticRegressionClassifier to %s", path)
        return path

    def load(self, file_path: Union[str, Path]) -> "LogisticRegressionClassifier":
        """Load a trained classifier from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
            
        state = joblib.load(path)
        self.C = state.get("C", 1.0)
        self.solver = state.get("solver", "lbfgs")
        self.max_iter = state.get("max_iter", 1000)
        self.class_weight = state.get("class_weight", None)
        self.random_state = state.get("random_state", settings.random_seed)
        self.classes_ = state.get("classes_", [0, 1, 2])
        self.n_features_in_ = state.get("n_features_in_", None)
        self.model = state["model"]
        self.is_trained = state.get("is_trained", True)
        logger.info("Loaded LogisticRegressionClassifier from %s", path)
        return self
