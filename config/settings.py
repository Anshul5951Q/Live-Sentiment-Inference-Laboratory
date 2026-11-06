"""Centralized configuration module for SentimentLab AI."""
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable override and strict validation."""
    
    # Environment & Seed
    environment: str = Field(default="development", description="Runtime environment")
    random_seed: int = Field(default=42, description="Global random seed for reproducibility")
    
    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "artifacts" / "data")
    models_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "artifacts" / "models")
    results_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "artifacts" / "results")
    
    # Dataset Parameters
    dataset_name: str = Field(default="Yelp/yelp_review_full", description="Hugging Face dataset identifier")
    total_real_samples: int = Field(default=24000, description="Total balanced real samples to select")
    samples_per_class: int = Field(default=8000, description="Real samples per class (8k Neg, 8k Neu, 8k Pos)")
    train_split_ratio: float = Field(default=0.8, description="Stratified train/val ratio (0.8 -> 19,200 train, 4,800 val)")
    eval_subset_samples_per_class: int = Field(default=1000, description="Evaluation subset per class (total 3,000)")
    
    # Label Names & Standardization
    # 0 = Negative, 1 = Neutral, 2 = Positive
    class_names: List[str] = Field(default=["Negative", "Neutral", "Positive"])
    label_to_sentiment: Dict[int, str] = Field(
        default={0: "Negative", 1: "Neutral", 2: "Positive"}
    )
    sentiment_to_label: Dict[str, int] = Field(
        default={"Negative": 0, "Neutral": 1, "Positive": 2}
    )
    
    # Synthetic Data Generation
    synthetic_total_target: int = Field(default=500, description="Total synthetic reviews to generate")
    synthetic_distribution: Dict[str, int] = Field(
        default={"Positive": 167, "Negative": 167, "Neutral": 166}
    )
    synthetic_batch_size: int = Field(default=15, description="Controlled batch size (10-20 reviews per call)")
    
    # LLM Settings
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model_name: str = Field(default="gemini-1.5-flash", alias="GEMINI_MODEL_NAME")
    gemini_max_retries: int = Field(default=5, description="Max retry attempts with exponential backoff")
    gemini_backoff_base_seconds: float = Field(default=2.0)
    gemini_backoff_max_seconds: float = Field(default=60.0)
    
    # Discriminative Models
    tfidf_max_features: int = Field(default=10000)
    tfidf_ngram_range: tuple = Field(default=(1, 2))
    
    lstm_max_sequence_length: int = Field(default=128)
    lstm_vocab_size: int = Field(default=20000)
    lstm_embedding_dim: int = Field(default=128)
    lstm_units: int = Field(default=64)
    lstm_batch_size: int = Field(default=64)
    lstm_epochs: int = Field(default=10)
    
    bert_model_name: str = Field(default="bert-base-uncased")
    bert_max_length: int = Field(default=128)
    bert_batch_size: int = Field(default=16)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def ensure_directories(self) -> None:
        """Create artifact directories if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

