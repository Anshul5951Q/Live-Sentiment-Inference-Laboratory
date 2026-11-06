"""Dataset acquisition, caching, and pipeline orchestration for Yelp Review Full."""
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
from datasets import load_dataset
from config.settings import settings
from config.logging_config import setup_logger
from src.data.preprocessor import preprocess_dataframe
from src.data.splitter import (
    create_stratified_subsample,
    split_train_validation,
    create_evaluation_subset,
    verify_no_leakage,
)
from src.utils.file_io import save_dataframe, load_dataframe, save_json, load_json

logger = setup_logger(__name__)


class YelpDataLoader:
    """Manages downloading, caching, preprocessing, and splitting the Yelp dataset."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_path = self.data_dir / "train_real.parquet"
        self.val_path = self.data_dir / "val_real.parquet"
        self.test_path = self.data_dir / "test_real.parquet"
        self.eval_subset_path = self.data_dir / "eval_subset_real.parquet"
        self.eval_meta_path = self.data_dir / "eval_subset_metadata.json"
        self.stats_path = self.data_dir / "dataset_stats.json"

    def splits_exist(self) -> bool:
        """Check if all preprocessed parquet splits exist on disk."""
        return (
            self.train_path.exists()
            and self.val_path.exists()
            and self.test_path.exists()
            and self.eval_subset_path.exists()
            and self.eval_meta_path.exists()
        )

    def load_cached_splits(self) -> Dict[str, pd.DataFrame]:
        """Load already processed splits from disk."""
        logger.info("Loading cached dataset splits from %s", self.data_dir)
        return {
            "train": load_dataframe(self.train_path),
            "val": load_dataframe(self.val_path),
            "test": load_dataframe(self.test_path),
            "eval_subset": load_dataframe(self.eval_subset_path),
        }

    def fetch_hf_dataset(self, split: str) -> pd.DataFrame:
        """Download raw Yelp Review Full split from Hugging Face."""
        logger.info("Fetching '%s' split of %s from Hugging Face...", split, settings.dataset_name)
        ds = load_dataset(settings.dataset_name, split=split)
        df = ds.to_pandas()
        logger.info("Fetched %d raw rows for split '%s'.", len(df), split)
        return df

    def prepare_dataset(self, force_reload: bool = False) -> Tuple[Dict[str, pd.DataFrame], Dict]:
        """Download, preprocess, split, validate, and cache the complete dataset pipeline.
        
        Returns:
            splits (Dict[str, pd.DataFrame]): train, val, test, eval_subset
            stats (Dict): Detailed dataset summary statistics
        """
        if self.splits_exist() and not force_reload:
            logger.info("Cached dataset splits found. Loading from cache.")
            splits = self.load_cached_splits()
            stats = load_json(self.stats_path) if self.stats_path.exists() else self.compute_stats(splits)
            return splits, stats

        logger.info("Preparing fresh dataset splits (force_reload=%s)...", force_reload)
        
        # 1. Fetch raw splits from Hugging Face
        raw_train = self.fetch_hf_dataset("train")
        raw_test = self.fetch_hf_dataset("test")
        
        # 2. Preprocess and map to 3 classes
        logger.info("Preprocessing raw train split...")
        processed_train_full = preprocess_dataframe(raw_train, text_col="text", label_col="label", source_tag="real")
        
        logger.info("Preprocessing raw test split...")
        processed_test_full = preprocess_dataframe(raw_test, text_col="text", label_col="label", source_tag="real")
        
        # 3. Create 24,000 balanced sample from train (8k Neg, 8k Neu, 8k Pos)
        subsample_24k = create_stratified_subsample(
            processed_train_full,
            samples_per_class=settings.samples_per_class,
            seed=settings.random_seed
        )
        
        # 4. Stratified split into 19,200 Train & 4,800 Val
        train_df, val_df = split_train_validation(
            subsample_24k,
            train_ratio=settings.train_split_ratio,
            seed=settings.random_seed
        )
        
        # 5. Full test split preserved untouched
        test_df = processed_test_full.reset_index(drop=True)
        
        # 6. Create fixed 3,000 evaluation subset (1,000 Neg, 1,000 Neu, 1,000 Pos)
        eval_subset_df, eval_meta = create_evaluation_subset(
            test_df,
            samples_per_class=settings.eval_subset_samples_per_class,
            seed=settings.random_seed
        )
        
        # 7. Strictly verify zero train/val/test data leakage
        verify_no_leakage(train_df, val_df, test_df, eval_subset_df)
        
        # 8. Save all splits to disk (atomic parquet)
        logger.info("Saving splits to %s...", self.data_dir)
        save_dataframe(train_df, self.train_path)
        save_dataframe(val_df, self.val_path)
        save_dataframe(test_df, self.test_path)
        save_dataframe(eval_subset_df, self.eval_subset_path)
        save_json(eval_meta, self.eval_meta_path)
        
        splits = {
            "train": train_df,
            "val": val_df,
            "test": test_df,
            "eval_subset": eval_subset_df,
        }
        
        # 9. Compute and persist statistics
        stats = self.compute_stats(splits)
        save_json(stats, self.stats_path)
        
        logger.info("Dataset preparation complete and cached successfully.")
        return splits, stats

    def compute_stats(self, splits: Dict[str, pd.DataFrame]) -> Dict:
        """Compute statistical breakdown for all splits."""
        stats = {}
        for split_name, df in splits.items():
            class_counts = df["label"].value_counts().to_dict()
            sentiment_counts = df["sentiment"].value_counts().to_dict()
            review_lengths = df["review"].str.split().str.len()
            
            stats[split_name] = {
                "total_samples": len(df),
                "label_distribution": {int(k): int(v) for k, v in class_counts.items()},
                "sentiment_distribution": {str(k): int(v) for k, v in sentiment_counts.items()},
                "token_length_mean": float(review_lengths.mean()),
                "token_length_median": float(review_lengths.median()),
                "token_length_min": int(review_lengths.min()),
                "token_length_max": int(review_lengths.max()),
            }
        return stats
