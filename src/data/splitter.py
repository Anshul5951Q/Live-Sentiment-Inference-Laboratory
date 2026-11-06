"""Stratified sampling, splitting, evaluation subset generation, and leakage checks."""
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger(__name__)


def create_stratified_subsample(
    df: pd.DataFrame,
    samples_per_class: int = 8000,
    seed: int = 42
) -> pd.DataFrame:
    """Create a balanced stratified subsample with exactly `samples_per_class` per label."""
    logger.info("Extracting balanced subsample (%d per class)...", samples_per_class)
    sampled_frames = []
    
    for label in (0, 1, 2):
        class_df = df[df["label"] == label]
        if len(class_df) < samples_per_class:
            raise ValueError(
                f"Class {label} ({settings.label_to_sentiment[label]}) only has {len(class_df)} samples, "
                f"required {samples_per_class}."
            )
        sampled = class_df.sample(n=samples_per_class, random_state=seed)
        sampled_frames.append(sampled)
        
    combined = pd.concat(sampled_frames, ignore_index=True)
    # Shuffle the combined dataset deterministically
    shuffled = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    logger.info("Stratified subsample created with %d total records.", len(shuffled))
    return shuffled


def split_train_validation(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a stratified split of the dataset into train and validation sets.
    
    Default: 24,000 samples -> 19,200 train (6,400 per class), 4,800 val (1,600 per class).
    """
    logger.info("Performing stratified train/validation split (ratio=%.2f)...", train_ratio)
    train_frames = []
    val_frames = []
    
    for label in (0, 1, 2):
        class_df = df[df["label"] == label].reset_index(drop=True)
        n_total = len(class_df)
        n_train = int(n_total * train_ratio)
        
        # Shuffle class_df deterministically
        shuffled = class_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        train_part = shuffled.iloc[:n_train]
        val_part = shuffled.iloc[n_train:]
        
        train_frames.append(train_part)
        val_frames.append(val_part)
        
    train_df = pd.concat(train_frames, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_frames, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    logger.info("Split completed: Train=%d, Validation=%d", len(train_df), len(val_df))
    return train_df, val_df


def create_evaluation_subset(
    test_df: pd.DataFrame,
    samples_per_class: int = 1000,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """Create a fixed, reproducible, stratified 3,000-sample evaluation subset from test data.
    
    Returns:
        eval_subset_df (pd.DataFrame): 3,000 samples (1,000 Neg, 1,000 Neu, 1,000 Pos)
        metadata (Dict): Selected indices, random seed, and class distributions
    """
    logger.info("Creating stratified evaluation subset (%d per class)...", samples_per_class)
    selected_indices: List[int] = []
    
    for label in (0, 1, 2):
        class_indices = test_df[test_df["label"] == label].index.tolist()
        if len(class_indices) < samples_per_class:
            raise ValueError(
                f"Test dataset has only {len(class_indices)} for class {label}, needed {samples_per_class}"
            )
        rng = np.random.default_rng(seed)
        sampled = rng.choice(class_indices, size=samples_per_class, replace=False).tolist()
        selected_indices.extend(sampled)
        
    eval_subset = test_df.loc[selected_indices].copy()
    eval_subset = eval_subset.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    metadata = {
        "random_seed": seed,
        "samples_per_class": samples_per_class,
        "total_eval_samples": len(eval_subset),
        "class_distribution": eval_subset["label"].value_counts().to_dict(),
        "selected_original_indices": [int(idx) for idx in selected_indices],
    }
    
    logger.info(
        "Evaluation subset created: Total=%d (distribution: %s)",
        len(eval_subset),
        metadata["class_distribution"]
    )
    return eval_subset, metadata


def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    eval_subset_df: Optional[pd.DataFrame] = None
) -> Dict[str, int]:
    """Verify strictly zero data leakage across train, validation, and test datasets.
    
    Raises:
        ValueError: If any review text appears across train/val and test sets.
    """
    train_texts: Set[str] = set(train_df["review"].str.strip())
    val_texts: Set[str] = set(val_df["review"].str.strip())
    test_texts: Set[str] = set(test_df["review"].str.strip())
    
    # Check train vs val
    train_val_overlap = train_texts.intersection(val_texts)
    # Check train vs test
    train_test_overlap = train_texts.intersection(test_texts)
    # Check val vs test
    val_test_overlap = val_texts.intersection(test_texts)
    
    leakage_stats = {
        "train_val_overlap": len(train_val_overlap),
        "train_test_overlap": len(train_test_overlap),
        "val_test_overlap": len(val_test_overlap),
    }
    
    if train_val_overlap:
        raise ValueError(f"Data leakage detected! {len(train_val_overlap)} reviews overlap between train and validation.")
    if train_test_overlap:
        raise ValueError(f"Data leakage detected! {len(train_test_overlap)} reviews overlap between train and test.")
    if val_test_overlap:
        raise ValueError(f"Data leakage detected! {len(val_test_overlap)} reviews overlap between validation and test.")
        
    if eval_subset_df is not None:
        eval_texts: Set[str] = set(eval_subset_df["review"].str.strip())
        train_eval_overlap = train_texts.intersection(eval_texts)
        val_eval_overlap = val_texts.intersection(eval_texts)
        leakage_stats["train_eval_overlap"] = len(train_eval_overlap)
        leakage_stats["val_eval_overlap"] = len(val_eval_overlap)
        
        if train_eval_overlap:
            raise ValueError(f"Data leakage detected! {len(train_eval_overlap)} reviews overlap between train and eval subset.")
        if val_eval_overlap:
            raise ValueError(f"Data leakage detected! {len(val_eval_overlap)} reviews overlap between validation and eval subset.")
            
    logger.info("Data leakage verification passed cleanly: %s", leakage_stats)
    return leakage_stats
