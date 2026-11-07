"""Hallucination and groundedness evaluation for generative sentiment explanations (Unit 5).

Measures whether generative LLM explanations introduce unsupported facts, fictitious entities,
or imaginary experiences not present in the original customer review.
"""
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.settings import settings
from config.logging_config import setup_logger
from src.utils.file_io import save_json

logger = setup_logger(__name__)

# Grounded benchmark reviews with known entities and factual boundaries
GROUNDED_EVAL_BENCHMARK = [
    {
        "review_id": "halluc_01",
        "review": "The sourdough bread was freshly baked and warm, but the minestrone soup tasted like canned tomatoes.",
        "grounded_entities": ["sourdough bread", "minestrone soup", "tomatoes"],
        "polarity": "Neutral",
    },
    {
        "review_id": "halluc_02",
        "review": "Our server David was extremely attentive, though the ribeye steak took 45 minutes to arrive.",
        "grounded_entities": ["server", "David", "ribeye steak", "45 minutes"],
        "polarity": "Neutral",
    },
    {
        "review_id": "halluc_03",
        "review": "The iced matcha latte was far too bitter, and they charged $8.50 for a small cup with no refills.",
        "grounded_entities": ["iced matcha latte", "small cup", "$8.50", "refills"],
        "polarity": "Negative",
    },
    {
        "review_id": "halluc_04",
        "review": "Fantastic crispy calamari and great jazz music in the background. We loved the outdoor patio.",
        "grounded_entities": ["calamari", "jazz music", "outdoor patio"],
        "polarity": "Positive",
    },
    {
        "review_id": "halluc_05",
        "review": "The hostess greeted us with a smile and seated us right away. The margherita pizza had fresh basil.",
        "grounded_entities": ["hostess", "margherita pizza", "basil"],
        "polarity": "Positive",
    },
    {
        "review_id": "halluc_06",
        "review": "We waited in the rain outside for an hour because their reservation system crashed.",
        "grounded_entities": ["rain", "outside", "hour", "reservation system"],
        "polarity": "Negative",
    },
]

# Common fabricated restaurant entities often hallucinated by LLMs
HALLUCINATED_ENTITY_PATTERNS = [
    r"\bchef\b", r"\bmanager\b", r"\bowner\b", r"\bdessert\b", r"\btiramisu\b",
    r"\bwine\b", r"\bbeer\b", r"\bcocktail\b", r"\bpasta\b", r"\bbill\b",
    r"\btip\b", r"\bparking\b", r"\bvalet\b", r"\brestroom\b", r"\bbathroom\b",
]


class HallucinationEvaluator:
    """Evaluates factuality and groundedness in LLM-generated sentiment rationales."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (settings.results_dir / "responsible_ai")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark = GROUNDED_EVAL_BENCHMARK

    def evaluate_explanation(
        self,
        review_text: str,
        explanation: str,
        known_entities: List[str]
    ) -> Dict[str, Any]:
        """Assess whether an explanation contains claims unsupported by the review text."""
        review_lower = review_text.lower()
        expl_lower = str(explanation).lower()

        # Check for ungrounded entity intrusions
        hallucinated_mentions = []
        for pattern in HALLUCINATED_ENTITY_PATTERNS:
            match = re.search(pattern, expl_lower)
            if match:
                matched_word = match.group(0)
                # If the word appears in the explanation but NOT in the original review, flag hallucination
                if matched_word not in review_lower:
                    hallucinated_mentions.append(matched_word)

        # Check grounded entity recall: did the explanation reference actual entities from the review?
        referenced_entities = [e for e in known_entities if e.lower() in expl_lower]
        grounded_recall = len(referenced_entities) / max(1, len(known_entities))
        is_grounded = len(hallucinated_mentions) == 0

        return {
            "is_grounded": is_grounded,
            "hallucinated_entities": hallucinated_mentions,
            "hallucination_count": len(hallucinated_mentions),
            "referenced_entities": referenced_entities,
            "grounded_entity_recall": round(grounded_recall, 4),
        }

    def evaluate_provider(self, provider, model_name: str = "Google Gemini") -> Dict[str, Any]:
        """Run groundedness evaluation across the benchmark using an actual or mock LLM provider."""
        logger.info("Evaluating hallucination and groundedness on %s...", model_name)
        records = []
        start_time = time.time()

        for item in self.benchmark:
            # Prompt provider for sentiment and explanation
            res = provider.classify_sentiment(item["review"], mode="zero_shot")
            eval_res = self.evaluate_explanation(
                review_text=item["review"],
                explanation=res.reason,
                known_entities=item["grounded_entities"]
            )
            records.append({
                "review_id": item["review_id"],
                "review": item["review"],
                "predicted_sentiment": res.predicted_sentiment,
                "confidence": res.confidence,
                "explanation": res.reason,
                **eval_res,
            })

        df = pd.DataFrame(records)
        groundedness_rate = float(df["is_grounded"].mean())
        unsupported_claim_rate = float(1.0 - groundedness_rate)
        mean_recall = float(df["grounded_entity_recall"].mean())

        results = {
            "model_name": model_name,
            "total_samples": len(df),
            "groundedness_rate": round(groundedness_rate, 4),
            "unsupported_claim_rate": round(unsupported_claim_rate, 4),
            "mean_grounded_entity_recall": round(mean_recall, 4),
            "samples": records,
        }

        save_json(results, self.output_dir / "hallucination_evaluation.json")
        self._plot_groundedness_chart(results)
        return results

    def _plot_groundedness_chart(self, results: Dict[str, Any]) -> Path:
        """Plot groundedness score and entity recall metrics."""
        chart_path = self.output_dir / "hallucination_groundedness_chart.png"
        metrics = ["Groundedness Rate", "Entity Recall", "Unsupported Claim Rate"]
        values = [
            results["groundedness_rate"] * 100,
            results["mean_grounded_entity_recall"] * 100,
            results["unsupported_claim_rate"] * 100,
        ]
        colors = ["#2ecc71", "#3498db", "#e74c3c"]

        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
        bars = ax.bar(metrics, values, color=colors, width=0.5)
        ax.set_title(f"LLM Explanation Groundedness & Hallucination Audit ({results['model_name']})", fontweight="bold")
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, 110)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontweight="bold")

        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return chart_path
