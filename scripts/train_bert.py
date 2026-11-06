"""CLI entrypoint to train, evaluate, or ingest Hugging Face BERT classifier artifacts."""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.logging_config import setup_logger
from src.training.train_bert import run_bert_training_pipeline, ingest_colab_bert_artifacts

logger = setup_logger("train_bert_cli")


def main():
    """Execute BERT training pipeline CLI or ingest Colab artifacts."""
    parser = argparse.ArgumentParser(description="SentimentLab AI - BERT Training and Ingestion CLI")
    parser.add_argument(
        "--ingest",
        type=str,
        default=None,
        help="Path to downloaded bert_artifacts.zip or extracted folder from Google Colab."
    )
    args = parser.parse_args()
    
    if args.ingest:
        ingest_path = Path(args.ingest)
        if not ingest_path.exists():
            print(f"\n[ERROR] Ingestion path not found: {ingest_path}\n")
            sys.exit(1)
            
        if ingest_path.suffix.lower() == ".zip":
            extract_dir = project_root / "temp_colab_extract"
            extract_dir.mkdir(parents=True, exist_ok=True)
            print(f"Unpacking {ingest_path.name}...")
            with zipfile.ZipFile(ingest_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            # Find the inner directory if nested
            chk_dirs = list(extract_dir.glob("**/bert_checkpoint"))
            target_dir = chk_dirs[0] if chk_dirs else extract_dir
            res = ingest_colab_bert_artifacts(target_dir)
            shutil.rmtree(extract_dir, ignore_errors=True)
        else:
            res = ingest_colab_bert_artifacts(ingest_path)
            
        print("\n" + "=" * 75)
        print("COLAB ARTIFACT INGESTION COMPLETE")
        print("=" * 75)
        print(f"Ingested {res['count']} artifact files into artifacts/models/bert/ and artifacts/results/bert/.")
        print("=" * 75 + "\n")
        return

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
        print("  4. Run: python scripts/train_bert.py --ingest bert_artifacts.zip")
        print("!" * 75 + "\n")
    else:
        print("\nBERT training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
