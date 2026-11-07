"""Google Gemini LLM provider for zero-shot and few-shot sentiment classification."""
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from config.settings import settings
from config.logging_config import setup_logger
from src.llm.base_provider import BaseLLMProvider
from src.llm.prompts import (
    SYSTEM_INSTRUCTION,
    format_zero_shot_prompt,
    format_few_shot_prompt,
)
from src.models.common_interface import ModelPredictionResult
from src.utils.file_io import save_json, load_json

logger = setup_logger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini sentiment classification provider with exponential backoff and caching."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        max_retries: int = settings.gemini_max_retries,
        backoff_base: float = settings.gemini_backoff_base_seconds,
        backoff_max: float = settings.gemini_backoff_max_seconds,
        rate_limit_delay: float = 4.0,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or settings.gemini_api_key
        super().__init__(model_name=model_name or os.getenv("GEMINI_MODEL_NAME") or settings.gemini_model_name)
        
        self.cache_dir = cache_dir or (settings.results_dir / "gemini")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / "response_cache.json"
        
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.rate_limit_delay = rate_limit_delay
        
        self._cache: Dict[str, Dict[str, Any]] = self._load_cache()
        self._client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Google GenAI client if API key is present."""
        if not self.api_key:
            logger.warning("No GEMINI_API_KEY provided. Provider will operate in offline/mock mode.")
            return
            
        try:
            # Try new official google-genai SDK first
            from google import genai
            self._client = genai.Client(api_key=self.api_key, http_options={"api_version": "v1beta"})
            self._sdk_type = "google-genai"
            logger.info("Initialized Google GenAI client with model '%s'.", self.model_name)
        except (ImportError, Exception):
            try:
                # Fall back to google-generativeai SDK
                import google.generativeai as genai_legacy
                genai_legacy.configure(api_key=self.api_key)
                self._client = genai_legacy.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=SYSTEM_INSTRUCTION
                )
                self._sdk_type = "google-generativeai"
                logger.info("Initialized legacy GenerativeModel with model '%s'.", self.model_name)
            except Exception as e:
                logger.error("Failed to initialize Gemini client: %s", str(e))
                self._client = None
                self._sdk_type = "none"

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load persistent cache from disk."""
        if self.cache_path.exists():
            try:
                return load_json(self.cache_path)
            except Exception:
                return {}
        return {}

    def _save_cache(self) -> None:
        """Persist cache to disk."""
        try:
            save_json(self._cache, self.cache_path)
        except Exception as e:
            logger.warning("Failed to save Gemini cache: %s", str(e))

    def health_check(self) -> Dict[str, Any]:
        """Check API connection status without consuming significant quota."""
        if not self.api_key:
            return {"status": "UNCONFIGURED", "message": "GEMINI_API_KEY is not set."}
        if self._client is None:
            return {"status": "ERROR", "message": "Failed to initialize Gemini client."}
            
        return {
            "status": "CONFIGURED",
            "model_name": self.model_name,
            "sdk_type": getattr(self, "_sdk_type", "unknown")
        }

    def _clean_json_text(self, text: str) -> str:
        """Extract and clean raw JSON from potentially markdown-wrapped LLM text."""
        t = text.strip()
        # Remove markdown code fence if present
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
        if match:
            t = match.group(1).strip()
        return t

    def _validate_schema(self, raw_json: Dict[str, Any]) -> Dict[str, Any]:
        """Validate LLM output schema strictly."""
        sentiment = str(raw_json.get("sentiment", "")).capitalize()
        if sentiment not in ("Negative", "Neutral", "Positive"):
            raise ValueError(f"Invalid sentiment string '{sentiment}'. Must be Negative, Neutral, or Positive.")
            
        expected_label = settings.sentiment_to_label[sentiment]
        label = raw_json.get("label")
        if label != expected_label:
            logger.warning("Correcting mismatched label %s to expected %d for %s.", label, expected_label, sentiment)
            label = expected_label
            
        conf = float(raw_json.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        
        reason = str(raw_json.get("reason", "")).strip()
        if not reason:
            reason = f"Classified as {sentiment} based on review polarity."
            
        return {
            "sentiment": sentiment,
            "label": label,
            "confidence": conf,
            "reason": reason,
        }

    def _call_api_with_backoff(self, prompt: str, temperature: float = 0.0) -> str:
        """Execute Gemini API call with bounded exponential backoff on 429/503."""
        if not self.api_key or self._client is None:
            raise RuntimeError("Gemini API key is not configured or client failed to initialize.")
            
        attempt = 0
        while attempt < self.max_retries:
            try:
                if self._sdk_type == "google-genai":
                    response = self._client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config={"temperature": temperature, "system_instruction": SYSTEM_INSTRUCTION}
                    )
                    return response.text
                else:
                    response = self._client.generate_content(
                        prompt,
                        generation_config={"temperature": temperature}
                    )
                    return response.text
            except Exception as e:
                err_str = str(e).lower()
                attempt += 1
                
                # Check for rate limit or quota exhaustion
                is_rate_limit = "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str
                is_transient = "503" in err_str or "unavailable" in err_str or "timeout" in err_str
                
                if (is_rate_limit or is_transient) and attempt < self.max_retries:
                    # Exponential backoff with jitter
                    sleep_time = min(self.backoff_max, self.backoff_base * (2 ** (attempt - 1)))
                    jitter = random.uniform(0.1, 1.0)
                    total_sleep = sleep_time + jitter
                    logger.warning(
                        "Gemini API rate-limit/transient error (%s). Backing off for %.2fs (attempt %d/%d)...",
                        err_str[:80], total_sleep, attempt, self.max_retries
                    )
                    time.sleep(total_sleep)
                else:
                    logger.error("Gemini API request failed after %d attempts: %s", attempt, str(e))
                    raise

        raise RuntimeError(f"Gemini API exceeded maximum retries ({self.max_retries}).")

    def classify_sentiment(
        self,
        text: str,
        mode: str = "zero_shot",
        temperature: float = 0.0
    ) -> ModelPredictionResult:
        """Classify a single review using specified prompt mode."""
        start_time = time.time()
        sanitized_text = str(text).strip() if text else ""
        
        # Check cache
        text_hash = hashlib.sha256(sanitized_text.encode("utf-8")).hexdigest()[:16]
        cache_key = f"{self.model_name}:{mode}:{text_hash}"
        if cache_key in self._cache:
            cached_val = self._cache[cache_key]
            return ModelPredictionResult(
                model_name=f"Gemini-{mode.replace('_', '').title()}",
                model_version=self.model_name,
                review=sanitized_text,
                predicted_label=cached_val["label"],
                predicted_sentiment=cached_val["sentiment"],
                confidence=cached_val["confidence"],
                class_probabilities=None,  # LLM confidence is NOT a calibrated probability distribution
                inference_duration_ms=0.0,
                is_calibrated=False,
                reason=cached_val.get("reason"),
                metadata={"source": "cache", "mode": mode}
            )
            
        # Format prompt
        if mode == "few_shot":
            prompt = format_few_shot_prompt(sanitized_text)
        else:
            prompt = format_zero_shot_prompt(sanitized_text)
            
        try:
            response_text = self._call_api_with_backoff(prompt, temperature=temperature)
            cleaned_json_str = self._clean_json_text(response_text)
            parsed = json.loads(cleaned_json_str)
            validated = self._validate_schema(parsed)
            
            # Cache successful response
            self._cache[cache_key] = validated
            self._save_cache()
            
            duration_ms = (time.time() - start_time) * 1000.0
            return ModelPredictionResult(
                model_name=f"Gemini-{mode.replace('_', '').title()}",
                model_version=self.model_name,
                review=sanitized_text,
                predicted_label=validated["label"],
                predicted_sentiment=validated["sentiment"],
                confidence=validated["confidence"],
                class_probabilities=None,
                inference_duration_ms=round(duration_ms, 2),
                is_calibrated=False,
                reason=validated["reason"],
                metadata={"source": "live_api", "mode": mode}
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000.0
            logger.error("Gemini classification failed for review '%s': %s", sanitized_text[:40], str(e))
            return ModelPredictionResult(
                model_name=f"Gemini-{mode.replace('_', '').title()}",
                model_version=self.model_name,
                review=sanitized_text,
                predicted_label=1,  # Neutral fallback on error
                predicted_sentiment="Neutral",
                confidence=0.0,
                class_probabilities=None,
                inference_duration_ms=round(duration_ms, 2),
                is_calibrated=False,
                error_status=str(e),
                reason="Classification failed due to API error or quota limit.",
                metadata={"source": "error_fallback", "mode": mode}
            )

    def batch_classify(
        self,
        texts: List[str],
        mode: str = "zero_shot",
        temperature: float = 0.0,
        checkpoint_path: Optional[str] = None
    ) -> List[ModelPredictionResult]:
        """Classify a batch of reviews with progress tracking and state checkpointing."""
        results: List[ModelPredictionResult] = []
        chk_file = Path(checkpoint_path) if checkpoint_path else (self.cache_dir / f"checkpoint_{mode}.json")
        
        # Load checkpoint if exists to allow resumption
        completed_records: Dict[int, Dict[str, Any]] = {}
        if chk_file.exists():
            try:
                completed_records = {int(k): v for k, v in load_json(chk_file).items()}
                logger.info("Resuming from checkpoint %s with %d existing records.", chk_file, len(completed_records))
            except Exception:
                completed_records = {}
                
        for idx, text in enumerate(texts):
            if idx in completed_records:
                rec = completed_records[idx]
                results.append(ModelPredictionResult(
                    model_name=rec["model_name"],
                    model_version=rec["model_version"],
                    review=rec["review"],
                    predicted_label=rec["predicted_label"],
                    predicted_sentiment=rec["predicted_sentiment"],
                    confidence=rec["confidence"],
                    class_probabilities=rec.get("class_probabilities"),
                    inference_duration_ms=rec.get("inference_duration_ms", 0.0),
                    is_calibrated=False,
                    reason=rec.get("reason"),
                    metadata=rec.get("metadata", {})
                ))
                continue
                
            res = self.classify_sentiment(text, mode=mode, temperature=temperature)
            results.append(res)
            completed_records[idx] = res.to_dict()
            
            # Periodic checkpoint saving
            if idx % 5 == 0 or idx == len(texts) - 1:
                save_json(completed_records, chk_file)
                
            # Polite throttle to respect free tier RPM (15 RPM -> ~4s per request)
            if res.metadata.get("source") == "live_api" and self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)
                
        return results
