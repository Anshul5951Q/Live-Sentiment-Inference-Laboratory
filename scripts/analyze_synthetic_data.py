"""CLI entrypoint to run quality assessment on synthetically generated reviews."""
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.logging_config import setup_logger
from src.evaluation.synthetic_data_analyzer import analyze_synthetic_dataset

logger = setup_logger("analyze_synthetic_data_cli")


def main():
    """Execute synthetic review quality analysis."""
    logger.info("Initializing synthetic dataset quality analysis...")
    report = analyze_synthetic_dataset()
    if report.get("status") == "DATASET_NOT_FOUND":
        print("\n" + "!" * 75)
        print("SYNTHETIC DATASET NOT FOUND:")
        print("Run 'python scripts/generate_synthetic_reviews.py' first to generate reviews.")
        print("!" * 75 + "\n")
        return
        
    print("\n" + "=" * 75)
    print("SYNTHETIC DATASET QUALITY REPORT")
    print("=" * 75)
    print(f"Total Records:                 {report['total_records']}")
    print(f"Negative Count:                {report['class_distribution'].get('Negative', 0)}")
    print(f"Neutral Count:                 {report['class_distribution'].get('Neutral', 0)}")
    print(f"Positive Count:                {report['class_distribution'].get('Positive', 0)}")
    print(f"Exact Duplicates in Synthetic: {report['exact_duplicates_within_synthetic']}")
    print(f"Exact Overlaps with Real Yelp: {report['exact_overlaps_with_real_yelp']}")
    print(f"Estimated Near-Duplicates:     {report['estimated_near_duplicates']}")
    print(f"Mean Character Length:         {report['length_statistics']['character_length']['mean']:.1f}")
    print(f"Median Character Length:       {report['length_statistics']['character_length']['median']:.1f}")
    print(f"Mean Word Count:               {report['length_statistics']['word_length']['mean']:.1f}")
    print(f"Generation Model:              {report['model_name']}")
    print(f"Provider:                      {report['provider']}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
