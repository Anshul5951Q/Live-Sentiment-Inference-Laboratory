"""Synthetic review generator using Google Gemini with structured JSON output and checkpointing."""
import abc
import datetime
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd

from config.settings import settings
from config.logging_config import setup_logger
from src.llm.gemini_provider import GeminiProvider
from src.llm.synthetic_review_validator import SyntheticReviewValidator
from src.utils.file_io import save_json, load_json, save_dataframe, load_dataframe

logger = setup_logger(__name__)

# Aspects and topics for rich, realistic customer reviews
ASPECT_CATEGORIES = [
    "food quality, taste, freshness, and menu variety",
    "waitstaff service, attentiveness, and ordering speed",
    "pricing, value for money, portions, and hidden charges",
    "restaurant ambiance, noise levels, seating comfort, and lighting",
    "wait times, reservations, line management, and table turnaround",
    "cleanliness, hygiene, restroom upkeep, and kitchen visibility",
    "takeout, food packaging, delivery promptness, and temperature",
    "mixed experience: exceptional food but frustrating service or loud environment",
    "mixed experience: friendly welcoming staff but bland disappointing food",
]

SYNTHETIC_GENERATION_PROMPT = """You are a natural customer review synthesis engine.
Generate exactly {batch_size} realistic, diverse, Yelp-style customer reviews for restaurants and local businesses with sentiment: {target_sentiment}.

Requirements:
1. Target Sentiment: {target_sentiment} (Label: {target_label}).
2. Focus topics for this batch: {aspect_topic}.
3. Variety: Vary review lengths (some short 2-3 sentences, some detailed paragraphs).
4. Realism: Write in authentic human voices (casual phrasing, specific details, natural punctuation, emotional cadence).
5. For Neutral reviews: Include genuinely mixed experiences (e.g. praised the appetizers but criticized the main course, or great ambience but mediocre coffee) or ordinary middle-of-the-road feedback (3 stars).
6. DO NOT use generic placeholders like '[Restaurant]' or '<Item Name>'. Invent realistic business and dish names.
7. Return strictly valid JSON containing a JSON list of review objects. No markdown formatting outside the JSON.

Expected JSON output format:
[
  {{
    "review": "Actual review text here...",
    "sentiment": "{target_sentiment}",
    "label": {target_label}
  }}
]
"""


class BaseSyntheticGeneratorProvider(abc.ABC):
    """Abstract provider for synthetic review generation."""

    @abc.abstractmethod
    def generate_batch(
        self,
        batch_size: int,
        target_sentiment: str,
        target_label: int,
        aspect_topic: str,
    ) -> List[Dict[str, Any]]:
        """Generate a batch of candidate reviews."""
        pass

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Name of the generative model provider."""
        pass

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Model identifier used for generation."""
        pass


class GeminiSyntheticProvider(BaseSyntheticGeneratorProvider):
    """Google Gemini implementation for synthetic review generation."""

    def __init__(self, gemini_provider: Optional[GeminiProvider] = None):
        self.provider = gemini_provider or GeminiProvider()
        
    @property
    def provider_name(self) -> str:
        return "Google Gemini (Academic Substitution for GPT)"

    @property
    def model_name(self) -> str:
        return self.provider.model_name

    def generate_batch(
        self,
        batch_size: int,
        target_sentiment: str,
        target_label: int,
        aspect_topic: str,
    ) -> List[Dict[str, Any]]:
        """Generate reviews via Google Gemini with JSON cleaning."""
        prompt = SYNTHETIC_GENERATION_PROMPT.format(
            batch_size=batch_size,
            target_sentiment=target_sentiment,
            target_label=target_label,
            aspect_topic=aspect_topic,
        )
        
        raw_text = self.provider._call_api_with_backoff(prompt, temperature=0.7)
        cleaned_json = self.provider._clean_json_text(raw_text)
        
        try:
            records = json.loads(cleaned_json)
            if isinstance(records, dict) and "reviews" in records:
                records = records["reviews"]
            if not isinstance(records, list):
                logger.warning("Gemini returned non-list JSON: %s", type(records))
                return []
                
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            formatted = []
            for r in records:
                if isinstance(r, dict) and "review" in r:
                    formatted.append({
                        "review": str(r["review"]).strip(),
                        "sentiment": target_sentiment,
                        "label": target_label,
                        "source": "synthetic_gemini",
                        "generation_model": self.model_name,
                        "generation_timestamp": timestamp,
                    })
            return formatted
        except Exception as e:
            logger.error("Failed to parse Gemini synthetic review batch: %s", str(e))
            return []


class SyntheticReviewGenerator:
    """Orchestrator for producing exactly 500 validated synthetic customer reviews."""

    TARGET_COUNTS = {
        "Negative": 167,
        "Neutral": 167,
        "Positive": 166,
    }
    TOTAL_TARGET = 500

    def __init__(
        self,
        provider: Optional[BaseSyntheticGeneratorProvider] = None,
        validator: Optional[SyntheticReviewValidator] = None,
        output_dir: Optional[Union[str, Path]] = None,
        batch_size: int = 15,
        rate_limit_delay: float = 4.0,
    ):
        self.provider = provider or GeminiSyntheticProvider()
        self.validator = validator or SyntheticReviewValidator()
        self.output_dir = Path(output_dir or settings.data_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_path = self.output_dir / "checkpoint_synthetic_generation.json"
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        
        self.accepted_records: List[Dict[str, Any]] = []
        self.rejected_records: List[Dict[str, Any]] = []
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Load state from checkpoint if it exists."""
        if self.checkpoint_path.exists():
            try:
                data = load_json(self.checkpoint_path)
                self.accepted_records = data.get("accepted_records", [])
                self.rejected_records = data.get("rejected_records", [])
                for r in self.accepted_records:
                    self.validator.seen_synthetic_texts.add(r["review"].strip().lower())
                logger.info(
                    "Resumed from synthetic checkpoint: %d accepted (%s), %d rejected.",
                    len(self.accepted_records),
                    self.get_current_counts(),
                    len(self.rejected_records)
                )
            except Exception as e:
                logger.warning("Could not read synthetic checkpoint: %s. Starting fresh.", str(e))

    def _save_checkpoint(self) -> None:
        """Save incremental progress to disk."""
        data = {
            "accepted_records": self.accepted_records,
            "rejected_records": self.rejected_records,
            "counts": self.get_current_counts(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "provider": self.provider.provider_name,
            "model_name": self.provider.model_name,
        }
        save_json(data, self.checkpoint_path)

    def get_current_counts(self) -> Dict[str, int]:
        """Get current accepted counts per sentiment."""
        counts = {"Negative": 0, "Neutral": 0, "Positive": 0}
        for r in self.accepted_records:
            sent = r.get("sentiment")
            if sent in counts:
                counts[sent] += 1
        return counts

    def generate_all(self, max_batches_per_class: int = 25) -> Dict[str, Any]:
        """Generate until exactly 500 valid records (167 Neg, 167 Neu, 166 Pos) are secured."""
        logger.info("=" * 60)
        logger.info("STARTING SYNTHETIC REVIEW GENERATION (TARGET: 500 REVIEWS)")
        logger.info("Provider: %s | Model: %s", self.provider.provider_name, self.provider.model_name)
        logger.info("=" * 60)
        
        start_time = time.time()
        aspect_idx = 0
        
        for sentiment, target_count in self.TARGET_COUNTS.items():
            label = settings.sentiment_to_label[sentiment]
            current_counts = self.get_current_counts()
            needed = target_count - current_counts[sentiment]
            batches_run = 0
            
            logger.info("Class '%s': currently have %d / %d needed (gap: %d)", sentiment, current_counts[sentiment], target_count, needed)
            
            while needed > 0 and batches_run < max_batches_per_class:
                batches_run += 1
                batch_n = min(self.batch_size, max(5, needed))
                aspect_topic = ASPECT_CATEGORIES[aspect_idx % len(ASPECT_CATEGORIES)]
                aspect_idx += 1
                
                logger.info("Requesting batch of %d '%s' reviews (Topic: %s)...", batch_n, sentiment, aspect_topic[:40])
                
                try:
                    candidates = self.provider.generate_batch(
                        batch_size=batch_n,
                        target_sentiment=sentiment,
                        target_label=label,
                        aspect_topic=aspect_topic,
                    )
                except Exception as e:
                    logger.error("Generation batch failed: %s. Preserving checkpoint and pausing.", str(e))
                    break
                    
                accepted, rejected = self.validator.validate_batch(candidates)
                
                # Append accepted up to target
                for rec in accepted:
                    if self.get_current_counts()[sentiment] < target_count:
                        self.accepted_records.append(rec)
                    else:
                        # Surplus record
                        pass
                        
                self.rejected_records.extend(rejected)
                self._save_checkpoint()
                
                current_counts = self.get_current_counts()
                needed = target_count - current_counts[sentiment]
                logger.info("Progress for '%s': %d / %d", sentiment, current_counts[sentiment], target_count)
                
                # Rate limit throttle
                time.sleep(self.rate_limit_delay)
                
        # Finalize and save datasets
        final_counts = self.get_current_counts()
        total_accepted = len(self.accepted_records)
        is_complete = total_accepted == self.TOTAL_TARGET and all(
            final_counts[s] == self.TARGET_COUNTS[s] for s in self.TARGET_COUNTS
        )
        
        # Save accepted parquet and metadata
        accepted_df = pd.DataFrame(self.accepted_records)
        rejected_df = pd.DataFrame(self.rejected_records)
        
        if not accepted_df.empty:
            save_dataframe(accepted_df, self.output_dir / "synthetic_reviews.parquet")
        if not rejected_df.empty:
            save_dataframe(rejected_df, self.output_dir / "synthetic_reviews_rejected.parquet")
            
        metadata = {
            "total_target": self.TOTAL_TARGET,
            "total_accepted": total_accepted,
            "is_complete": is_complete,
            "counts": final_counts,
            "total_rejected": len(self.rejected_records),
            "provider": self.provider.provider_name,
            "model_name": self.provider.model_name,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "elapsed_seconds": round(time.time() - start_time, 2),
        }
        save_json(metadata, self.output_dir / "synthetic_reviews_metadata.json")
        
        logger.info(
            "Synthetic generation run finished. Status: %s | Total: %d / %d",
            "COMPLETE" if is_complete else "PARTIALLY COMPLETE",
            total_accepted, self.TOTAL_TARGET
        )
        return metadata

