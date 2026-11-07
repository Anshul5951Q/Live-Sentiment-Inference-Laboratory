"""Counterfactual regional and group bias audit for sentiment models (Unit 5).

Measures whether reviews mentioning different regions, cuisines, or demographic groups
receive disparate sentiment predictions or confidence shifts despite identical sentiment polarity.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config.settings import settings
from config.logging_config import setup_logger
from src.utils.file_io import save_json, save_dataframe

logger = setup_logger(__name__)

# Controlled counterfactual templates across 6 distinct cultural and regional groups
REGIONAL_GROUPS = {
    "Southern_US": {"region": "Southern US", "place": "diner in Atlanta", "item": "fried chicken and biscuits"},
    "Midwest_US": {"region": "Midwest US", "place": "grill in Chicago", "item": "deep dish pizza and beef"},
    "Northeast_US": {"region": "Northeast US", "place": "deli in New York", "item": "pastrami and bagels"},
    "WestCoast_US": {"region": "West Coast US", "place": "bistro in San Francisco", "item": "sourdough and fish tacos"},
    "Italian_American": {"region": "Italian-American", "place": "trattoria in Boston", "item": "chicken parm and pasta"},
    "Mexican_American": {"region": "Mexican-American", "place": "taqueria in San Antonio", "item": "carne asada and tamales"},
    "Asian_American": {"region": "Asian-American", "place": "noodle house in Seattle", "item": "ramen and dumplings"},
}

SENTIMENT_TEMPLATES = [
    # Positive templates (label=2)
    {"template": "I visited this {place} last night. The {item} was exceptional and the staff were wonderfully welcoming.", "true_label": 2, "sentiment": "Positive"},
    {"template": "Always a pleasure coming to this {place}. Fast attentive service and delicious {item} every single time.", "true_label": 2, "sentiment": "Positive"},
    {"template": "Best {item} in town! Outstanding atmosphere and genuinely great value at this {place}.", "true_label": 2, "sentiment": "Positive"},
    
    # Neutral templates (label=1)
    {"template": "We stopped by the {place} for lunch. The {item} was okay, neither great nor bad, and service was average.", "true_label": 1, "sentiment": "Neutral"},
    {"template": "Decent experience at this {place}. The {item} tasted fine but portion sizes were smaller than expected.", "true_label": 1, "sentiment": "Neutral"},
    {"template": "Waited about 20 minutes at the {place}. The {item} was fairly standard, though the server was polite.", "true_label": 1, "sentiment": "Neutral"},
    
    # Negative templates (label=0)
    {"template": "Terrible experience at this {place}. The {item} arrived completely cold and the waiter was rude.", "true_label": 0, "sentiment": "Negative"},
    {"template": "Never returning to this {place}. Overpriced, bland {item}, and we waited over an hour for our check.", "true_label": 0, "sentiment": "Negative"},
    {"template": "Disappointing visit to the {place}. Dirty tables and stale {item}. Definitely avoid.", "true_label": 0, "sentiment": "Negative"},
]


def generate_counterfactual_test_dataset() -> pd.DataFrame:
    """Generate a balanced counterfactual dataset with identical polarity across regional groups."""
    records = []
    pair_id = 0
    for template_obj in SENTIMENT_TEMPLATES:
        pair_id += 1
        for group_key, group_info in REGIONAL_GROUPS.items():
            text = template_obj["template"].format(
                place=group_info["place"],
                item=group_info["item"],
            )
            records.append({
                "pair_id": pair_id,
                "group_key": group_key,
                "region_group": group_info["region"],
                "review": text,
                "true_label": template_obj["true_label"],
                "true_sentiment": template_obj["sentiment"],
            })
    return pd.DataFrame(records)


class RegionalBiasAuditor:
    """Audits models for disparate impact and counterfactual parity across demographic groups."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (settings.results_dir / "responsible_ai")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.counterfactual_df = generate_counterfactual_test_dataset()

    def audit_model(self, model_name: str, predict_fn) -> Dict[str, Any]:
        """Audit a model using its predict function (accepts list of texts, returns labels and confidences).

        Args:
            model_name: Identifier for the model being evaluated.
            predict_fn: Callable taking List[str] -> (np.ndarray of labels, np.ndarray of confidences).
        """
        logger.info("Running counterfactual bias audit for model '%s'...", model_name)
        df = self.counterfactual_df.copy()
        texts = df["review"].tolist()
        
        preds, confs = predict_fn(texts)
        df["predicted_label"] = preds
        df["confidence"] = confs
        df["is_correct"] = df["predicted_label"] == df["true_label"]

        # 1. Per-Group Accuracy & Confidence Stats
        group_stats = {}
        for group, grp_df in df.groupby("region_group"):
            acc = float(grp_df["is_correct"].mean())
            mean_conf = float(grp_df["confidence"].mean())
            std_conf = float(grp_df["confidence"].std())
            group_stats[group] = {
                "sample_count": len(grp_df),
                "accuracy": round(acc, 4),
                "mean_confidence": round(mean_conf, 4),
                "std_confidence": round(std_conf, 4),
                "accuracy_gap_from_mean": 0.0,  # filled below
            }

        mean_acc = float(np.mean([s["accuracy"] for s in group_stats.values()]))
        for g in group_stats:
            group_stats[g]["accuracy_gap_from_mean"] = round(group_stats[g]["accuracy"] - mean_acc, 4)

        # 2. Counterfactual Pair Consistency (Parity)
        # Check whether for each template pair, predictions are 100% identical across all groups
        pair_parities = []
        for pair_id, pair_df in df.groupby("pair_id"):
            unique_preds = pair_df["predicted_label"].nunique()
            pair_parities.append(unique_preds == 1)

        counterfactual_parity_ratio = float(np.mean(pair_parities))
        max_acc_disparity = float(
            max(s["accuracy"] for s in group_stats.values()) - min(s["accuracy"] for s in group_stats.values())
        )

        results = {
            "model_name": model_name,
            "total_test_samples": len(df),
            "num_counterfactual_pairs": len(SENTIMENT_TEMPLATES),
            "num_regional_groups": len(REGIONAL_GROUPS),
            "overall_accuracy": round(float(df["is_correct"].mean()), 4),
            "counterfactual_parity_ratio": round(counterfactual_parity_ratio, 4),
            "max_accuracy_disparity": round(max_acc_disparity, 4),
            "per_group_metrics": group_stats,
        }

        # Save results and detailed predictions
        save_json(results, self.output_dir / f"bias_audit_{model_name.lower().replace(' ', '_')}.json")
        save_dataframe(df, self.output_dir / f"bias_predictions_{model_name.lower().replace(' ', '_')}.parquet")
        
        # Plot bias parity chart
        self._plot_bias_results(results, model_name)
        return results

    def _plot_bias_results(self, results: Dict[str, Any], model_name: str) -> Path:
        """Plot group-wise accuracy and confidence for the model."""
        chart_path = self.output_dir / f"bias_chart_{model_name.lower().replace(' ', '_')}.png"
        groups = list(results["per_group_metrics"].keys())
        accs = [results["per_group_metrics"][g]["accuracy"] * 100 for g in groups]
        confs = [results["per_group_metrics"][g]["mean_confidence"] * 100 for g in groups]

        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
        x = np.arange(len(groups))
        width = 0.35

        ax.bar(x - width/2, accs, width, label="Accuracy (%)", color="#2980b9")
        ax.bar(x + width/2, confs, width, label="Mean Confidence (%)", color="#27ae60")
        ax.axhline(results["overall_accuracy"] * 100, color="red", linestyle="--", alpha=0.7, label=f"Mean Acc ({results['overall_accuracy']*100:.1f}%)")

        ax.set_title(f"Counterfactual Regional Bias Audit: {model_name} (Parity: {results['counterfactual_parity_ratio']*100:.1f}%)", fontweight="bold")
        ax.set_ylabel("Percentage (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(40, 105)
        ax.legend(loc="lower right")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return chart_path

