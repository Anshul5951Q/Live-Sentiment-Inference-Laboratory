"""Validation pipeline for synthetically generated customer reviews."""
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger(__name__)


class SyntheticReviewValidator:
    """Strict validator enforcing quality, schema, uniqueness, and label consistency."""

    def __init__(
        self,
        real_review_texts: Optional[Set[str]] = None,
        min_char_length: int = 25,
        max_char_length: int = 2500,
    ):
        self.real_review_texts = real_review_texts or set()
        self.min_char_length = min_char_length
        self.max_char_length = max_char_length
        self.seen_synthetic_texts: Set[str] = set()
        
        # Regex for detecting unfilled template placeholders
        self.template_placeholder_pattern = re.compile(
            r"\[(?:insert|dish|restaurant|name|food|city|price|item|location)[^\]]*\]|\{(?:insert|dish|restaurant|name|food|city|price|item|location)[^\}]*\}|<(?:insert|dish|restaurant|name|food|city|price|item|location)[^>]*>",
            flags=re.IGNORECASE
        )

    def set_real_corpus(self, real_texts: Set[str]) -> None:
        """Set the set of real review strings for duplicate checking."""
        self.real_review_texts = {t.strip().lower() for t in real_texts if t}

    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate an individual candidate review record.
        
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, rejection_reason)
        """
        if not isinstance(record, dict):
            return False, "Record is not a valid dictionary."
            
        review = str(record.get("review", "")).strip()
        sentiment = str(record.get("sentiment", "")).capitalize()
        label = record.get("label")
        
        # 1. Non-empty and length bounds
        if not review:
            return False, "Review text is empty or whitespace-only."
        if len(review) < self.min_char_length:
            return False, f"Review too short ({len(review)} chars < {self.min_char_length})."
        if len(review) > self.max_char_length:
            return False, f"Review too long ({len(review)} chars > {self.max_char_length})."
            
        # 2. Template placeholders check
        if self.template_placeholder_pattern.search(review):
            return False, "Review contains unpopulated template placeholders."
            
        # 3. Sentiment class validation
        if sentiment not in ("Negative", "Neutral", "Positive"):
            return False, f"Invalid sentiment '{sentiment}'. Must be Negative, Neutral, or Positive."
            
        # 4. Label range and agreement
        expected_label = settings.sentiment_to_label[sentiment]
        if label != expected_label:
            return False, f"Label {label} does not match sentiment '{sentiment}' (expected {expected_label})."
            
        # 5. Exact duplicate in current synthetic batch
        norm_text = review.lower()
        if norm_text in self.seen_synthetic_texts:
            return False, "Exact duplicate of another synthetic review."
            
        # 6. Exact overlap with real Yelp dataset
        if norm_text in self.real_review_texts:
            return False, "Exact collision with existing real Yelp review."
            
        # 7. Secondary Heuristic Label-Consistency Pass
        # E.g. Check for blatant semantic mismatch (e.g., negative review with 0 negation and extreme superlatives)
        extreme_positive_words = {"superb", "perfection", "magnificent", "masterpiece", "heavenly"}
        extreme_negative_words = {"disgusting", "horrible", "inedible", "revolting", "infestation"}
        
        words = set(re.findall(r"\b[a-z]+\b", norm_text))
        if sentiment == "Negative" and any(w in words for w in extreme_positive_words) and not any(w in words for w in {"not", "never", "no", "hardly", "barely", "wasn't", "didn't"}):
            return False, "Secondary validation failed: Negative review contains extreme positive superlative without negation."
        if sentiment == "Positive" and any(w in words for w in extreme_negative_words) and not any(w in words for w in {"not", "never", "no", "wasn't"}):
            return False, "Secondary validation failed: Positive review contains extreme negative descriptor without negation."
            
        # All checks passed
        self.seen_synthetic_texts.add(norm_text)
        return True, None

    def validate_batch(
        self,
        candidate_records: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Validate a list of candidate review records.
        
        Returns:
            Tuple[List[accepted_records], List[rejected_records]]
        """
        accepted = []
        rejected = []
        
        for rec in candidate_records:
            is_valid, reason = self.validate_record(rec)
            if is_valid:
                accepted.append(rec)
            else:
                rej_rec = dict(rec)
                rej_rec["rejection_reason"] = reason
                rejected.append(rej_rec)
                
        logger.info(
            "Batch validation completed: %d accepted, %d rejected.",
            len(accepted), len(rejected)
        )
        return accepted, rejected

