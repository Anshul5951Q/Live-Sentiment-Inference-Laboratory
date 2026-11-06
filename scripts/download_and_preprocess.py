"""Script to download, clean, preprocess, split, and cache the Yelp dataset."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.logging_config import setup_logger
from src.data.loader import YelpDataLoader
from src.utils.seed import set_seed

logger = setup_logger("download_and_preprocess")


def main():
    """Execute dataset preparation pipeline."""
    set_seed(settings.random_seed)
    settings.ensure_directories()
    
    logger.info("Initializing YelpDataLoader...")
    loader = YelpDataLoader()
    
    force_reload = "--force" in sys.argv
    splits, stats = loader.prepare_dataset(force_reload=force_reload)
    
    print("\n" + "="*50)
    print("DATASET PIPELINE PREPARATION SUMMARY")
    print("="*50)
    for split_name, split_stats in stats.items():
        print(f"\nSplit: {split_name.upper()}")
        print(f"  Total Samples:        {split_stats['total_samples']}")
        print(f"  Label Distribution:   {split_stats['label_distribution']}")
        print(f"  Sentiment Breakdown:  {split_stats['sentiment_distribution']}")
        print(f"  Mean Token Length:    {split_stats['token_length_mean']:.2f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
