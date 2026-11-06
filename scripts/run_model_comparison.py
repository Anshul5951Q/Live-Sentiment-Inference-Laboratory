"""CLI script to run direct five-way benchmark across LR, BiLSTM, BERT, and Gemini."""
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.logging_config import setup_logger
from src.models.predictor import SentimentPredictor
from src.models.lstm_classifier import KerasBiLSTMClassifier
from src.models.text_tokenizer import TextTokenizer
from src.llm.sentiment_classifier import (
    extract_gemini_benchmark_subset,
    GeminiSentimentService,
)
from src.evaluation.model_comparison import run_four_way_comparison
from src.utils.file_io import load_dataframe

logger = setup_logger("run_model_comparison_cli")


def main():
    """Execute model benchmark comparison across all models."""
    logger.info("Initializing five-way model comparison pipeline...")
    
    # 1. Load evaluation subset and extract/load 300-sample benchmark
    eval_df = load_dataframe(settings.data_dir / "eval_subset_real.parquet")
    benchmark_df, meta = extract_gemini_benchmark_subset(eval_df, samples_per_class=100, seed=settings.random_seed)
    
    # 2. Load Logistic Regression Predictor
    lr_models_dir = settings.models_dir / "logistic_regression"
    lr_predictor = SentimentPredictor.from_artifacts(lr_models_dir)
    
    # 3. Load BiLSTM Classifier & Tokenizer
    lstm_models_dir = settings.models_dir / "lstm"
    lstm_tokenizer = TextTokenizer().load(lstm_models_dir / "tokenizer.json")
    lstm_classifier = KerasBiLSTMClassifier().load(lstm_models_dir / "lstm_model.keras")
    
    # 4. Check for BERT Predictor
    bert_predictor = None
    bert_dir = settings.models_dir / "bert"
    if (bert_dir / "config.json").exists() and ((bert_dir / "model.safetensors").exists() or (bert_dir / "pytorch_model.bin").exists()):
        try:
            from src.models.bert_classifier import BertSentimentPredictor
            logger.info("Loading fine-tuned BERT predictor from %s...", bert_dir)
            bert_predictor = BertSentimentPredictor.from_artifacts(bert_dir)
        except Exception as e:
            logger.warning("Could not load BERT predictor: %s", str(e))
    else:
        logger.info("BERT checkpoint not found in %s.", bert_dir)
        
    # 5. Check for Gemini predictions
    gemini_dir = settings.results_dir / "gemini"
    zs_path = gemini_dir / "zero_shot_predictions.parquet"
    fs_path = gemini_dir / "few_shot_predictions.parquet"
    
    zs_df = None
    fs_df = None
    
    # Attempt to load or run Gemini if API key is configured
    if zs_path.exists():
        logger.info("Loading existing Gemini Zero-Shot predictions from %s...", zs_path)
        zs_df = load_dataframe(zs_path)
    else:
        logger.info("Running Gemini Zero-Shot benchmark...")
        gemini_service = GeminiSentimentService()
        health = gemini_service.provider.health_check()
        if health.get("status") == "CONFIGURED":
            zs_df, _ = gemini_service.run_benchmark(benchmark_df, mode="zero_shot")
        else:
            logger.warning("Gemini API not configured (%s). Skipping live Gemini Zero-Shot API calls.", health.get("message"))
            
    if fs_path.exists():
        logger.info("Loading existing Gemini Few-Shot predictions from %s...", fs_path)
        fs_df = load_dataframe(fs_path)
    else:
        logger.info("Running Gemini Few-Shot benchmark...")
        gemini_service = GeminiSentimentService()
        health = gemini_service.provider.health_check()
        if health.get("status") == "CONFIGURED":
            fs_df, _ = gemini_service.run_benchmark(benchmark_df, mode="few_shot")
        else:
            logger.warning("Gemini API not configured (%s). Skipping live Gemini Few-Shot API calls.", health.get("message"))
            
    # 6. Execute benchmark comparison
    comparison_summary = run_four_way_comparison(
        lr_predictor=lr_predictor,
        lstm_classifier=lstm_classifier,
        lstm_tokenizer=lstm_tokenizer,
        gemini_zero_shot_df=zs_df,
        gemini_few_shot_df=fs_df,
        bert_predictor=bert_predictor,
        benchmark_df=benchmark_df
    )
    
    # 7. Display formatted summary table
    print("\n" + "=" * 80)
    print("FIVE-WAY BENCHMARK COMPARISON ON 300 IDENTICAL EVALUATION REVIEWS")
    print("=" * 80)
    print(f"{'Model':<24} | {'Accuracy':<10} | {'Macro F1':<10} | {'Brier':<10} | {'Latency (ms)':<12}")
    print("-" * 80)
    for model_name, metrics in comparison_summary.items():
        brier_str = f"{metrics['brier_score']:.4f}" if "brier_score" in metrics else "N/A (LLM)"
        lat_str = f"{metrics.get('latency_per_sample_ms', 0.0):.1f}"
        print(f"{model_name:<24} | {metrics['accuracy']:<10.4f} | {metrics['macro_f1']:<10.4f} | {brier_str:<10} | {lat_str:<12}")
        
    if "HuggingFace_BERT" not in comparison_summary:
        print(f"{'HuggingFace_BERT':<24} | {'PENDING':<10} | {'PENDING':<10} | {'PENDING':<10} | {'PENDING_CHECKPOINT':<12}")
    if "Gemini_Zero_Shot" not in comparison_summary:
        print(f"{'Gemini_Zero_Shot':<24} | {'PENDING':<10} | {'PENDING':<10} | {'N/A (LLM)':<10} | {'PENDING_API_KEY':<12}")
    if "Gemini_Few_Shot" not in comparison_summary:
        print(f"{'Gemini_Few_Shot':<24} | {'PENDING':<10} | {'PENDING':<10} | {'N/A (LLM)':<10} | {'PENDING_API_KEY':<12}")
    print("=" * 80)
    
    notes = []
    if "HuggingFace_BERT" not in comparison_summary:
        notes.append("BERT checkpoint pending. To fine-tune on free GPU: see notebooks/01_colab_bert_training.ipynb")
    if "Gemini_Zero_Shot" not in comparison_summary:
        notes.append("Gemini live benchmark pending API key ($env:GEMINI_API_KEY='your_key').")
        
    if notes:
        print("\n[BENCHMARK STATUS NOTES]")
        for n in notes:
            print(f"  * {n}")
        print()


if __name__ == "__main__":
    main()
