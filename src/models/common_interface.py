"""Standardized common prediction interface across all discriminative and generative models."""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


@dataclass
class ModelPredictionResult:
    """Standardized output structure for single-review sentiment predictions."""
    
    model_name: str
    model_version: str
    review: str
    predicted_label: int  # 0: Negative, 1: Neutral, 2: Positive
    predicted_sentiment: str  # "Negative", "Neutral", "Positive"
    confidence: float  # In [0.0, 1.0]
    class_probabilities: Optional[Dict[str, float]] = None  # Full posterior distribution [P(Neg), P(Neu), P(Pos)]
    inference_duration_ms: float = 0.0
    is_calibrated: bool = False
    error_status: Optional[str] = None
    reason: Optional[str] = None  # LLM in-context justification if applicable
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        return asdict(self)

