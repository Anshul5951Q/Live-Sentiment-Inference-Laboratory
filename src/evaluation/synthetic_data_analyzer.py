"""Quality assessment and distribution analysis for synthetically generated reviews."""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config.settings import settings
from config.logging_config import setup_logger
from src.utils.file_io import save_json, load_json, load_dataframe

logger = setup_logger(__name__)


def compute_text_length_stats(texts: List[str]) -> Dict[str, Any]:
    """Calculate descriptive statistics for text lengths (character and word counts)."""
    char_lens = [len(str(t)) for t in texts]
    word_lens = [len(re.findall(r"\b\w+\b", str(t))) for t in texts]
    
    return {
        "character_length": {
            "mean": float(np.mean(char_lens)) if char_lens else 0.0,
            "median": float(np.median(char_lens)) if char_lens else 0.0,
            "std": float(np.std(char_lens)) if char_lens else 0.0,
            "min": int(np.min(char_lens)) if char_lens else 0,
            "max": int(np.max(char_lens)) if char_lens else 0,
        },
        "word_length": {
            "mean": float(np.mean(word_lens)) if word_lens else 0.0,
            "median": float(np.median(word_lens)) if word_lens else 0.0,
            "std": float(np.std(word_lens)) if word_lens else 0.0,
            "min": int(np.min(word_lens)) if word_lens else 0,
            "max": int(np.max(word_lens)) if word_lens else 0,
        }
    }


def compute_near_duplicates(texts: List[str], threshold: float = 0.85) -> int:
    """Estimate near-duplicate count using word-level Jaccard similarity."""
    word_sets = [set(re.findall(r"\b[a-z]{3,}\b", str(t).lower())) for t in texts]
    near_dups = 0
    n = len(word_sets)
    # Check sample if n is large to bound O(n^2)
    sample_limit = min(n, 500)
    for i in range(sample_limit):
        for j in range(i + 1, sample_limit):
            s1, s2 = word_sets[i], word_sets[j]
            if s1 and s2:
                sim = len(s1 & s2) / len(s1 | s2)
                if sim >= threshold:
                    near_dups += 1
    return near_dups


def analyze_synthetic_dataset(
    synthetic_parquet_path: Optional[Path] = None,
    real_parquet_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete quality analysis and generate publication charts."""
    synth_path = synthetic_parquet_path or (settings.data_dir / "synthetic_reviews.parquet")
    out_dir = output_dir or (settings.results_dir / "synthetic_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not synth_path.exists():
        logger.warning("Synthetic dataset not found at %s. Cannot run quality analysis.", synth_path)
        return {"status": "DATASET_NOT_FOUND", "path": str(synth_path)}
        
    synth_df = load_dataframe(synth_path)
    synth_texts = synth_df["review"].dropna().tolist()
    
    # 1. Load Real Data for overlap and distribution comparison (if available)
    real_path = real_parquet_path or (settings.data_dir / "train_real.parquet")
    if real_path.exists():
        real_df = load_dataframe(real_path)
        real_texts = set(real_df["review"].dropna().str.lower().str.strip().tolist())
    else:
        real_df = pd.DataFrame({"review": []})
        real_texts = set()
    
    # 2. Duplicate and Overlap Analysis
    exact_synth_duplicates = len(synth_texts) - len(set(t.lower().strip() for t in synth_texts))
    exact_real_overlaps = sum(1 for t in synth_texts if t.lower().strip() in real_texts)
    near_duplicates = compute_near_duplicates(synth_texts, threshold=0.85)
    
    # 3. Length statistics
    synth_stats = compute_text_length_stats(synth_texts)
    real_stats = compute_text_length_stats(real_df["review"].dropna().tolist()[:len(synth_texts)]) if not real_df.empty else synth_stats
    
    # 4. Class Distribution
    counts = synth_df["sentiment"].value_counts().to_dict()
    
    # 5. Metadata if available
    meta_path = settings.data_dir / "synthetic_reviews_metadata.json"
    metadata = load_json(meta_path) if meta_path.exists() else {}
    
    quality_report = {
        "status": "COMPLETED",
        "total_records": len(synth_df),
        "class_distribution": counts,
        "length_statistics": synth_stats,
        "exact_duplicates_within_synthetic": exact_synth_duplicates,
        "exact_overlaps_with_real_yelp": exact_real_overlaps,
        "estimated_near_duplicates": near_duplicates,
        "provider": metadata.get("provider", "Google Gemini"),
        "model_name": metadata.get("model_name", "gemini-1.5-flash"),
        "total_rejected": metadata.get("total_rejected", 0),
    }
    
    save_json(quality_report, out_dir / "synthetic_quality_report.json")
    
    # 6. Plots
    # Plot 1: Synthetic Class Distribution
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    classes = ["Negative", "Neutral", "Positive"]
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    vals = [counts.get(c, 0) for c in classes]
    bars = ax.bar(classes, vals, color=colors, width=0.5)
    ax.set_title("Synthetic Reviews Class Balance (Target: 500 Total)", fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel("Number of Validated Reviews", fontsize=10)
    ax.set_ylim([0, max(vals) * 1.2 if vals else 200])
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h}", xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "synthetic_class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Plot 2: Synthetic Length Distribution
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    char_lens = [len(t) for t in synth_texts]
    sns.histplot(char_lens, bins=20, kde=True, color="#3498db", ax=ax)
    ax.set_title("Synthetic Review Character Length Distribution", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Review Length (Characters)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    if char_lens:
        ax.axvline(np.mean(char_lens), color="red", linestyle="--", label=f"Mean: {np.mean(char_lens):.1f}")
        ax.axvline(np.median(char_lens), color="orange", linestyle=":", label=f"Median: {np.median(char_lens):.1f}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "synthetic_length_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Plot 3: Real vs Synthetic Length Comparison
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    if not real_df.empty and len(real_df["review"].dropna()) > 0:
        real_sample_lens = [len(str(t)) for t in real_df["review"].dropna().sample(min(len(real_df), 1000), random_state=42)]
        sns.kdeplot(real_sample_lens, label="Real Yelp Reviews", color="#9b59b6", fill=True, alpha=0.3, ax=ax)
        max_xlim = max(max(real_sample_lens), max(char_lens)) if char_lens else 1000
    else:
        max_xlim = max(char_lens) if char_lens else 1000
        
    if char_lens:
        sns.kdeplot(char_lens, label="Synthetic Gemini Reviews", color="#2ecc71", fill=True, alpha=0.3, ax=ax)
    ax.set_title("Length Distribution Comparison: Real vs Synthetic", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Character Length", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_xlim([0, min(3000, max_xlim)])
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "real_vs_synthetic_length_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    logger.info("Synthetic data quality analysis completed and charts saved to %s", out_dir)
    return quality_report
