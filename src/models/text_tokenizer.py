"""Text tokenizer and sequence padder for Deep Learning models."""
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger(__name__)

DEFAULT_PUNCTUATION_FILTERS = string.punctuation + "\t\n\r"


class TextTokenizer:
    """Configurable text tokenizer with integer encoding, OOV handling, and padding."""

    def __init__(
        self,
        vocab_size: int = settings.lstm_vocab_size,
        max_length: int = settings.lstm_max_sequence_length,
        oov_token: str = "<OOV>",
        pad_token: str = "<PAD>",
        lower: bool = True,
        filters: str = DEFAULT_PUNCTUATION_FILTERS
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.oov_token = oov_token
        self.pad_token = pad_token
        self.lower = lower
        self.filters = filters
        
        # Index 0 is reserved for padding, index 1 for OOV
        self.word_index: Dict[str, int] = {self.pad_token: 0, self.oov_token: 1}
        self.index_word: Dict[int, str] = {0: self.pad_token, 1: self.oov_token}
        self.word_counts: Counter = Counter()
        self.is_fitted: bool = False

    def _tokenize_text(self, text: Optional[str]) -> List[str]:
        """Convert a single string into a list of cleaned word tokens."""
        if text is None or not isinstance(text, str):
            return []
            
        t = text.lower() if self.lower else text
        # Remove specified filter characters
        translate_table = str.maketrans(self.filters, " " * len(self.filters))
        t = t.translate(translate_table)
        tokens = t.strip().split()
        return tokens

    def fit(self, texts: Union[List[str], np.ndarray]) -> "TextTokenizer":
        """Fit vocabulary exclusively on training texts.
        
        Leakage prevention: Must only be fitted on train_real.parquet.
        """
        logger.info("Fitting TextTokenizer on %d training samples (vocab_size=%d)...", len(texts), self.vocab_size)
        self.word_counts = Counter()
        
        for text in texts:
            tokens = self._tokenize_text(text)
            self.word_counts.update(tokens)
            
        # Select the most frequent words up to vocab_size - 2 (accounting for PAD and OOV)
        max_words = max(0, self.vocab_size - 2)
        most_common = self.word_counts.most_common(max_words)
        
        self.word_index = {self.pad_token: 0, self.oov_token: 1}
        self.index_word = {0: self.pad_token, 1: self.oov_token}
        
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word_index[word] = idx
            self.index_word[idx] = word
            
        self.is_fitted = True
        logger.info(
            "TextTokenizer fitted. Vocabulary size: %d words (OOV index=%d, PAD index=%d).",
            len(self.word_index),
            self.word_index[self.oov_token],
            self.word_index[self.pad_token]
        )
        return self

    def texts_to_sequences(self, texts: Union[List[str], np.ndarray]) -> List[List[int]]:
        """Transform input texts into lists of integer token IDs."""
        if not self.is_fitted:
            raise RuntimeError("TextTokenizer must be fitted before calling texts_to_sequences.")
            
        oov_idx = self.word_index[self.oov_token]
        sequences: List[List[int]] = []
        
        for text in texts:
            tokens = self._tokenize_text(text)
            seq = [self.word_index.get(token, oov_idx) for token in tokens]
            sequences.append(seq)
            
        return sequences

    def pad_sequences(
        self,
        sequences: List[List[int]],
        padding: str = "post",
        truncating: str = "post"
    ) -> np.ndarray:
        """Pad or truncate integer sequences to fixed max_length."""
        padded = np.zeros((len(sequences), self.max_length), dtype=np.int32)
        
        for i, seq in enumerate(sequences):
            if not seq:
                continue
                
            # Truncate if longer than max_length
            if len(seq) > self.max_length:
                if truncating == "post":
                    seq_proc = seq[:self.max_length]
                else:
                    seq_proc = seq[-self.max_length:]
            else:
                seq_proc = seq
                
            # Place in array according to padding rule
            if padding == "post":
                padded[i, :len(seq_proc)] = seq_proc
            else:
                padded[i, -len(seq_proc):] = seq_proc
                
        return padded

    def transform(
        self,
        texts: Union[List[str], np.ndarray],
        padding: str = "post",
        truncating: str = "post"
    ) -> np.ndarray:
        """Tokenize and pad texts into a fixed-shape numpy matrix (N, max_length)."""
        seqs = self.texts_to_sequences(texts)
        return self.pad_sequences(seqs, padding=padding, truncating=truncating)

    def fit_transform(self, texts: Union[List[str], np.ndarray]) -> np.ndarray:
        """Fit on training texts and return padded sequence matrix."""
        return self.fit(texts).transform(texts)

    def save(self, file_path: Union[str, Path]) -> Path:
        """Save tokenizer configuration and vocabulary mapping to JSON."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted tokenizer.")
            
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "oov_token": self.oov_token,
            "pad_token": self.pad_token,
            "lower": self.lower,
            "filters": self.filters,
            "word_index": self.word_index,
            "num_words_fitted": len(self.word_index),
            "is_fitted": self.is_fitted,
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
            
        logger.info("Saved TextTokenizer to %s (vocab: %d)", path, len(self.word_index))
        return path

    def load(self, file_path: Union[str, Path]) -> "TextTokenizer":
        """Load tokenizer configuration and vocabulary mapping from JSON."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
            
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        self.vocab_size = state["vocab_size"]
        self.max_length = state["max_length"]
        self.oov_token = state["oov_token"]
        self.pad_token = state["pad_token"]
        self.lower = state.get("lower", True)
        self.filters = state.get("filters", DEFAULT_PUNCTUATION_FILTERS)
        self.word_index = state["word_index"]
        self.index_word = {int(v): k for k, v in self.word_index.items()}
        self.is_fitted = state.get("is_fitted", True)
        logger.info("Loaded TextTokenizer from %s (vocab: %d)", path, len(self.word_index))
        return self
