"""Service for orchestrating Gemini Zero-Shot and Few-Shot sentiment benchmarking."""
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from config.settings import settings
from config.logging_config import setup_logger
from src.llm.gemini_provider import GeminiProvider
from src.models.common_interface import ModelPredictionResult
from src.evaluation.classification_metrics import evaluate_sentiment_model
from src.utils.file_io import save_dataframe, load_dataframe, save_json, load_json

logger = setup_logger(__name__)


def extract_gemini_benchmark_subset(
    eval_df: pd.DataFrame,
    samples_per_class: int = 100,
    seed: int = 42,
    output_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Extract a reproducible stratified sample of 300 records (100 per class) for LLM benchmarking."""
    out_dir = output_dir or settings.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    benchmark_path = out_dir / "gemini_benchmark_300.parquet"
    meta_path = out_dir / "gemini_benchmark_300_metadata.json"
    
    if benchmark_path.exists() and meta_path.exists():
        logger.info("Loading existing Gemini 300 benchmark from %s...", benchmark_path)
        return load_dataframe(benchmark_path), load_json(meta_path)
        
    logger.info("Extracting stratified 300-sample benchmark from evaluation subset...")
    selected_indices: List[int] = []
    
    for label in (0, 1, 2):
        class_indices = eval_df[eval_df["label"] == label].index.tolist()
        if len(class_indices) < samples_per_class:
            raise ValueError(f"Insufficient samples for class {label}: {len(class_indices)} < {samples_per_class}")
        rng = np.random.default_rng(seed)
        sampled = rng.choice(class_indices, size=samples_per_class, replace=False).tolist()
        selected_indices.extend(sampled)
        
    benchmark_df = eval_df.loc[selected_indices].copy()
    benchmark_df = benchmark_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    meta = {
        "random_seed": seed,
        "samples_per_class": samples_per_class,
        "total_samples": len(benchmark_df),
        "class_distribution": benchmark_df["label"].value_counts().to_dict(),
        "selected_original_indices": [int(idx) for idx in selected_indices],
    }
    
    save_dataframe(benchmark_df, benchmark_path)
    save_json(meta, meta_path)
    logger.info("Saved Gemini 300 benchmark to %s and metadata to %s", benchmark_path, meta_path)
    return benchmark_df, meta


class GeminiSentimentService:
    """Orchestrates zero-shot and few-shot evaluation against the 300-sample benchmark."""

    def __init__(self, provider: Optional[GeminiProvider] = None):
        self.provider = provider or GeminiProvider()
        self.results_dir = settings.results_dir / "gemini"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_benchmark(
        self,
        benchmark_df: pd.DataFrame,
        mode: str = "zero_shot",
        max_samples: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute benchmark for the specified prompting mode ('zero_shot' or 'few_shot')."""
        subset = benchmark_df.iloc[:max_samples] if max_samples else benchmark_df
        logger.info("Starting Gemini '%s' benchmarking on %d reviews...", mode, len(subset))
        
        texts = subset["review"].tolist()
        y_true = subset["label"].values.tolist()
        
        chk_path = self.results_dir / f"checkpoint_{mode}.json"
        results = self.provider.batch_classify(
            texts=texts,
            mode=mode,
            temperature=0.0,
            checkpoint_path=str(chk_path)
        )
        
        # Compile predictions DataFrame
        pred_records = []
        y_pred = []
        confidences = []
        successful_calls = 0
        failed_calls = 0
        total_latency_ms = 0.0
        
        for i, res in enumerate(results):
            pred_records.append({
                "review": res.review,
                "true_label": int(y_true[i]),
                "true_sentiment": settings.label_to_sentiment[y_true[i]],
                "predicted_label": res.predicted_label,
                "predicted_sentiment": res.predicted_sentiment,
                "confidence": res.confidence,
                "reason": res.reason,
                "inference_duration_ms": res.inference_duration_ms,
                "error_status": res.error_status,
                "source": res.metadata.get("source", "unknown"),
            })
            y_pred.append(res.predicted_label)
            confidences.append(res.confidence)
            total_latency_ms += res.inference_duration_ms
            
            if res.error_status:
                failed_calls += 1
            else:
                successful_calls += 1
                
        pred_df = pd.DataFrame(pred_records)
        pred_path = self.results_dir / f"{mode}_predictions.parquet"
        save_dataframe(pred_df, pred_path)
        
        # Calculate classification metrics
        # Note: Do not pass confidences as probability distributions for Brier/log-loss
        metrics = evaluate_sentiment_model(y_true, y_pred, y_prob=None)
        
        metrics["api_stats"] = {
            "mode": mode,
            "total_requested": len(subset),
            "successful_predictions": successful_calls,
            "failed_predictions": failed_calls,
            "coverage_ratio": float(successful_calls / len(subset)) if len(subset) > 0 else 0.0,
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "mean_latency_ms": float(total_latency_ms / len(subset)) if len(subset) > 0 else 0.0,
        }
        
        metrics_path = self.results_dir / f"{mode}_metrics.json"
        save_json(metrics, metrics_path)
        logger.info("Gemini '%s' benchmark complete. Accuracy: %.4f | Coverage: %d/%d", mode, metrics["accuracy"], successful_calls, len(subset))
        return pred_df, metrics

