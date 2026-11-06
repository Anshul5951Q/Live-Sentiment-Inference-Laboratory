"""TF-IDF feature extraction module for text classification."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger(__name__)


class TextTfidfVectorizer:
    """Configurable TF-IDF feature extractor with strict leakage prevention."""

    def __init__(
        self,
        max_features: int = settings.tfidf_max_features,
        ngram_range: Tuple[int, int] = settings.tfidf_ngram_range,
        min_df: Union[int, float] = 2,
        max_df: Union[int, float] = 0.95,
        sublinear_tf: bool = True,
        stop_words: Optional[Union[str, List[str]]] = None,
        token_pattern: str = r"(?u)\b\w+\b",
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.stop_words = stop_words
        self.token_pattern = token_pattern
        
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.is_fitted: bool = False

    def _sanitize_input(self, texts: Union[List[str], np.ndarray]) -> List[str]:
        """Validate and sanitize input review texts."""
        if texts is None:
            raise ValueError("Input texts cannot be None.")
            
        sanitized = []
        for t in texts:
            if not isinstance(t, str) or not t.strip():
                # Provide a neutral empty placeholder for whitespace/empty strings
                sanitized.append("")
            else:
                sanitized.append(t.strip())
        return sanitized

    def fit(self, texts: Union[List[str], np.ndarray]) -> "TextTfidfVectorizer":
        """Fit the TF-IDF vocabulary strictly on training texts.
        
        Leakage guard: Must ONLY be called on the training split.
        """
        clean_texts = self._sanitize_input(texts)
        logger.info("Fitting TF-IDF vectorizer on %d training texts...", len(clean_texts))
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            stop_words=self.stop_words,
            token_pattern=self.token_pattern,
        )
        self.vectorizer.fit(clean_texts)
        self.is_fitted = True
        logger.info(
            "TF-IDF vectorizer fitted successfully. Vocabulary size: %d features.",
            len(self.vectorizer.vocabulary_)
        )
        return self

    def transform(self, texts: Union[List[str], np.ndarray]) -> sp.csr_matrix:
        """Transform texts into TF-IDF sparse matrix using the fitted vocabulary."""
        if not self.is_fitted or self.vectorizer is None:
            raise RuntimeError("TextTfidfVectorizer must be fitted before transforming.")
            
        clean_texts = self._sanitize_input(texts)
        return self.vectorizer.transform(clean_texts)

    def fit_transform(self, texts: Union[List[str], np.ndarray]) -> sp.csr_matrix:
        """Fit on training texts and transform in a single pass."""
        return self.fit(texts).transform(texts)

    def get_feature_names(self) -> List[str]:
        """Return the list of vocabulary feature names."""
        if not self.is_fitted or self.vectorizer is None:
            raise RuntimeError("Vectorizer is not fitted.")
        return list(self.vectorizer.get_feature_names_out())

    def save(self, file_path: Union[str, Path]) -> Path:
        """Serialize and save the fitted vectorizer to disk."""
        if not self.is_fitted or self.vectorizer is None:
            raise RuntimeError("Cannot save an unfitted vectorizer.")
            
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "sublinear_tf": self.sublinear_tf,
            "stop_words": self.stop_words,
            "token_pattern": self.token_pattern,
            "vectorizer": self.vectorizer,
            "vocabulary_size": len(self.vectorizer.vocabulary_),
            "is_fitted": self.is_fitted,
        }
        joblib.dump(state, path)
        logger.info("Saved TextTfidfVectorizer to %s", path)
        return path

    def load(self, file_path: Union[str, Path]) -> "TextTfidfVectorizer":
        """Deserialize and restore a fitted vectorizer from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Vectorizer file not found at {path}")
            
        state = joblib.load(path)
        self.max_features = state.get("max_features", settings.tfidf_max_features)
        self.ngram_range = state.get("ngram_range", settings.tfidf_ngram_range)
        self.min_df = state.get("min_df", 2)
        self.max_df = state.get("max_df", 0.95)
        self.sublinear_tf = state.get("sublinear_tf", True)
        self.stop_words = state.get("stop_words", None)
        self.token_pattern = state.get("token_pattern", r"(?u)\b\w+\b")
        self.vectorizer = state["vectorizer"]
        self.is_fitted = state.get("is_fitted", True)
        logger.info("Loaded TextTfidfVectorizer from %s (vocab: %d)", path, len(self.vectorizer.vocabulary_))
        return self
