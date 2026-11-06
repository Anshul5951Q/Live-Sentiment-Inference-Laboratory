"""CLI entrypoint to train and evaluate Hugging Face BERT classifier."""
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.logging_config import setup_logger
from src.training.train_bert import run_bert_training_pipeline

logger = setup_logger("train_bert_cli")


def main():
    """Execute BERT training pipeline CLI."""
    logger.info("Initializing Hugging Face BERT training...")
    result = run_bert_training_pipeline()
    if result.get("status") == "BLOCKED_BY_HARDWARE_OR_POLICY":
        print("\n" + "!" * 75)
        print("LOCAL BERT EXECUTION NOTICE:")
        print(result.get("message"))
        print("\nTo train BERT on a free NVIDIA T4 GPU via Google Colab:")
        print("  1. Open: notebooks/01_colab_bert_training.ipynb")
        print("  2. Select Runtime -> Change runtime type -> T4 GPU")
        print("  3. Run all cells to fine-tune BERT and download the artifact zip")
        print("  4. Extract the artifacts into 'artifacts/models/bert/' and 'artifacts/results/bert/'")
        print("!" * 75 + "\n")
    else:
        print("\nBERT training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
