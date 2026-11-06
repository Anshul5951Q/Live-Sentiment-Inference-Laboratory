"""Abstract Base Class for Large Language Model Providers."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from src.models.common_interface import ModelPredictionResult


class BaseLLMProvider(ABC):
    """Abstract interface defining required LLM capabilities for sentiment classification."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def classify_sentiment(
        self,
        text: str,
        mode: str = "zero_shot",
        temperature: float = 0.0
    ) -> ModelPredictionResult:
        """Classify a single review using either 'zero_shot' or 'few_shot' prompting."""
        pass

    @abstractmethod
    def batch_classify(
        self,
        texts: List[str],
        mode: str = "zero_shot",
        temperature: float = 0.0,
        checkpoint_path: Optional[str] = None
    ) -> List[ModelPredictionResult]:
        """Classify a list of reviews with rate-limiting, retry backoff, and checkpointing."""
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Check API connectivity and model availability."""
        pass

