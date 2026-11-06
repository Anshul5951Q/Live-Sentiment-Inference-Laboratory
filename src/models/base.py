"""Abstract Base Class for Sentiment Analysis Models."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np


class BaseSentimentModel(ABC):
    """Abstract base class that all sentiment classifiers must implement."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_trained: bool = False

    @abstractmethod
    def train(self, texts: List[str], labels: List[int], val_texts: Optional[List[str]] = None, val_labels: Optional[List[int]] = None) -> Dict[str, Any]:
        """Train the model on the provided text samples and labels."""
        pass

    @abstractmethod
    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict integer class labels (0=Negative, 1=Neutral, 2=Positive) for input texts."""
        pass

    @abstractmethod
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict class probability distributions [P(0), P(1), P(2)] for input texts."""
        pass

    @abstractmethod
    def save(self, file_path: Union[str, Path]) -> Path:
        """Serialize and save model weights/pipeline to disk."""
        pass

    @abstractmethod
    def load(self, file_path: Union[str, Path]) -> None:
        """Deserialize and load model weights/pipeline from disk."""
        pass
