"""Inference service for Logistic Regression sentiment classification."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from config.settings import settings
from config.logging_config import setup_logger
from src.models.tfidf_vectorizer import TextTfidfVectorizer
from src.models.logistic_regression import LogisticRegressionClassifier
from src.models.calibration import MulticlassCalibrator
from src.utils.file_io import load_json

logger = setup_logger(__name__)


class SentimentPredictor:
    """Production-ready inference service for multi-class sentiment analysis."""

    def __init__(
        self,
        vectorizer: TextTfidfVectorizer,
        uncalibrated_model: LogisticRegressionClassifier,
        calibrated_model: Optional[MulticlassCalibrator] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.vectorizer = vectorizer
        self.uncalibrated_model = uncalibrated_model
        self.calibrated_model = calibrated_model
        self.metadata = metadata or {}
        self.model_version = self.metadata.get("version", "1.0.0")

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: Optional[Union[str, Path]] = None
    ) -> "SentimentPredictor":
        """Load inference components from saved disk artifacts."""
        base_path = Path(artifacts_dir or (settings.models_dir / "logistic_regression"))
        
        vec_path = base_path / "tfidf_vectorizer.joblib"
        uncal_path = base_path / "logistic_regression.joblib"
        cal_path = base_path / "calibrated_logistic_regression.joblib"
        meta_path = base_path / "model_metadata.json"
        
        logger.info("Loading SentimentPredictor components from %s...", base_path)
        vectorizer = TextTfidfVectorizer().load(vec_path)
        uncalibrated_model = LogisticRegressionClassifier().load(uncal_path)
        
        calibrated_model = None
        if cal_path.exists():
            calibrated_model = MulticlassCalibrator().load(cal_path)
            
        metadata = load_json(meta_path) if meta_path.exists() else {}
        
        return cls(
            vectorizer=vectorizer,
            uncalibrated_model=uncalibrated_model,
            calibrated_model=calibrated_model,
            metadata=metadata,
        )

    def _predict_batch(
        self,
        texts: List[str],
        use_calibrated: bool = True
    ) -> List[Dict[str, Any]]:
        """Run batch inference on a list of texts."""
        sanitized = [str(t).strip() if t is not None else "" for t in texts]
        
        # Transform features
        X = self.vectorizer.transform(sanitized)
        
        # Uncalibrated probabilities
        uncal_probs = self.uncalibrated_model.predict_proba(X)
        
        # Calibrated probabilities (if available and requested)
        has_calibrated = (self.calibrated_model is not None and self.calibrated_model.is_calibrated)
        if has_calibrated:
            cal_probs = self.calibrated_model.predict_proba(X)
        else:
            cal_probs = uncal_probs
            
        active_probs = cal_probs if (use_calibrated and has_calibrated) else uncal_probs
        predicted_labels = np.argmax(active_probs, axis=1)
        confidences = np.max(active_probs, axis=1)
        
        results = []
        for i, text in enumerate(sanitized):
            pred_lbl = int(predicted_labels[i])
            pred_sent = settings.label_to_sentiment[pred_lbl]
            
            uncal_dict = {
                settings.label_to_sentiment[k]: float(uncal_probs[i, k])
                for k in range(3)
            }
            cal_dict = {
                settings.label_to_sentiment[k]: float(cal_probs[i, k])
                for k in range(3)
            } if has_calibrated else uncal_dict
            
            results.append({
                "review": text,
                "predicted_label": pred_lbl,
                "predicted_sentiment": pred_sent,
                "confidence": float(confidences[i]),
                "calibrated_probabilities": cal_dict,
                "uncalibrated_probabilities": uncal_dict,
                "is_calibrated_output": bool(use_calibrated and has_calibrated),
                "model_name": "LogisticRegression",
                "model_version": self.model_version,
            })
            
        return results

    def predict(
        self,
        text_or_texts: Union[str, List[str]],
        use_calibrated: bool = True
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Classify single review string or list of review strings."""
        if isinstance(text_or_texts, str):
            results = self._predict_batch([text_or_texts], use_calibrated=use_calibrated)
            return results[0]
        elif isinstance(text_or_texts, (list, tuple, np.ndarray)):
            return self._predict_batch(list(text_or_texts), use_calibrated=use_calibrated)
        else:
            raise TypeError(f"Expected str or List[str], got {type(text_or_texts).__name__}")

