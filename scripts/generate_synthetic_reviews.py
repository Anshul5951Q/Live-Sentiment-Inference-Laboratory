"""CLI entrypoint for generating and validating 500 synthetic customer reviews."""
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.logging_config import setup_logger
from src.llm.gemini_provider import GeminiProvider
from src.llm.synthetic_review_generator import (
    GeminiSyntheticProvider,
    SyntheticReviewGenerator,
)
from src.llm.synthetic_review_validator import SyntheticReviewValidator
from src.utils.file_io import load_dataframe

logger = setup_logger("generate_synthetic_reviews_cli")


def main():
    """Execute synthetic review generation CLI."""
    logger.info("Initializing synthetic review generation pipeline...")
    
    # 1. Load real corpus to prevent data leakage or overlap
    real_texts = set()
    for split_file in ["train_real.parquet", "val_real.parquet", "eval_subset_real.parquet"]:
        p = settings.data_dir / split_file
        if p.exists():
            df = load_dataframe(p)
            real_texts.update(df["review"].dropna().astype(str).tolist())
            
    logger.info("Loaded %d real Yelp reviews for duplicate/collision prevention.", len(real_texts))
    
    # 2. Check API key status
    gemini_provider = GeminiProvider()
    health = gemini_provider.health_check()
    if health.get("status") != "CONFIGURED":
        print("\n" + "!" * 75)
        print("GEMINI API KEY REQUIRED FOR SYNTHETIC GENERATION:")
        print(health.get("message"))
        print("\nTo generate the 500 synthetic reviews via Google Gemini:")
        print("  1. Copy .env.example to .env and set GEMINI_API_KEY=your_key")
        print("  2. Or run: $env:GEMINI_API_KEY='your_key'")
        print("  3. Run: python scripts/generate_synthetic_reviews.py")
        print("!" * 75 + "\n")
        return
        
    # 3. Initialize generator and validator
    validator = SyntheticReviewValidator(real_review_texts=real_texts)
    provider = GeminiSyntheticProvider(gemini_provider=gemini_provider)
    generator = SyntheticReviewGenerator(
        provider=provider,
        validator=validator,
        batch_size=15,
        rate_limit_delay=4.0
    )
    
    # 4. Run generation
    metadata = generator.generate_all()
    
    # 5. Print status
    print("\n" + "=" * 75)
    print("SYNTHETIC REVIEW GENERATION SUMMARY")
    print("=" * 75)
    print(f"Provider:           {metadata['provider']}")
    print(f"Model Identifier:   {metadata['model_name']}")
    print(f"Total Accepted:     {metadata['total_accepted']} / {metadata['total_target']}")
    print(f"Negative (Target 167): {metadata['counts'].get('Negative', 0)}")
    print(f"Neutral  (Target 167): {metadata['counts'].get('Neutral', 0)}")
    print(f"Positive (Target 166): {metadata['counts'].get('Positive', 0)}")
    print(f"Total Rejected:     {metadata['total_rejected']}")
    print(f"Completion Status:  {'COMPLETE' if metadata['is_complete'] else 'PARTIALLY COMPLETE'}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
