"""Training and evaluation pipeline for Keras Bidirectional LSTM."""
import datetime
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from config.settings import settings
from config.logging_config import setup_logger
from src.models.text_tokenizer import TextTokenizer
from src.models.lstm_classifier import KerasBiLSTMClassifier
from src.evaluation.classification_metrics import evaluate_sentiment_model
from src.evaluation.visualizer import plot_confusion_matrix
from src.utils.file_io import save_json, save_dataframe, load_dataframe
from src.utils.seed import set_seed

logger = setup_logger(__name__)


def plot_training_history(history: Dict[str, list], output_dir: Path) -> Tuple[Path, Path]:
    """Plot training vs validation loss and accuracy curves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Loss Curve
    loss_path = output_dir / "loss_curve.png"
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    epochs = range(1, len(history["loss"]) + 1)
    ax.plot(epochs, history["loss"], "o-", label="Training Loss", color="#e74c3c", linewidth=2)
    ax.plot(epochs, history["val_loss"], "s--", label="Validation Loss", color="#3498db", linewidth=2)
    ax.set_title("BiLSTM: Training vs Validation Loss", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Sparse Cross-Entropy Loss", fontsize=10)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # 2. Accuracy Curve
    acc_path = output_dir / "accuracy_curve.png"
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(epochs, history["accuracy"], "o-", label="Training Accuracy", color="#2ecc71", linewidth=2)
    ax.plot(epochs, history["val_accuracy"], "s--", label="Validation Accuracy", color="#9b59b6", linewidth=2)
    ax.set_title("BiLSTM: Training vs Validation Accuracy", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(acc_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    logger.info("Saved loss curve to %s and accuracy curve to %s", loss_path, acc_path)
    return loss_path, acc_path


def run_lstm_pipeline() -> Dict[str, Any]:
    """Execute the full Bidirectional LSTM training and evaluation workflow."""
    start_time = time.time()
    set_seed(settings.random_seed)
    
    models_dir = settings.models_dir / "lstm"
    results_dir = settings.results_dir / "lstm"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("STARTING KERAS BIDIRECTIONAL LSTM TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # 1. Load dataset splits
    train_df = load_dataframe(settings.data_dir / "train_real.parquet")
    val_df = load_dataframe(settings.data_dir / "val_real.parquet")
    eval_df = load_dataframe(settings.data_dir / "eval_subset_real.parquet")
    
    logger.info(
        "Loaded splits: Train=%d, Validation=%d, Evaluation Subset=%d",
        len(train_df), len(val_df), len(eval_df)
    )
    
    # 2. Fit Tokenizer strictly on training data
    tokenizer_start = time.time()
    tokenizer = TextTokenizer(
        vocab_size=settings.lstm_vocab_size,
        max_length=settings.lstm_max_sequence_length,
    )
    tokenizer.fit(train_df["review"].values)
    tokenizer_duration = time.time() - tokenizer_start
    
    tok_path = models_dir / "tokenizer.json"
    tokenizer.save(tok_path)
    
    # 3. Transform splits to padded integer sequences
    X_train = tokenizer.transform(train_df["review"].values)
    y_train = train_df["label"].values.astype(np.int32)
    
    X_val = tokenizer.transform(val_df["review"].values)
    y_val = val_df["label"].values.astype(np.int32)
    
    X_eval = tokenizer.transform(eval_df["review"].values)
    y_eval = eval_df["label"].values.astype(np.int32)
    
    # 4. Initialize and Train Keras BiLSTM
    clf = KerasBiLSTMClassifier(
        vocab_size=len(tokenizer.word_index),
        max_length=settings.lstm_max_sequence_length,
        embedding_dim=settings.lstm_embedding_dim,
        lstm_units=settings.lstm_units,
        dropout=0.3,
        random_seed=settings.random_seed,
    )
    
    checkpoint_path = models_dir / "lstm_model.keras"
    train_start = time.time()
    history = clf.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=settings.lstm_batch_size,
        epochs=settings.lstm_epochs,
        checkpoint_path=checkpoint_path
    )
    training_duration = time.time() - train_start
    
    # Save training history
    save_json(history, results_dir / "training_history.json")
    
    # 5. Evaluate on Validation Set
    val_probs = clf.predict_proba(X_val)
    val_preds = np.argmax(val_probs, axis=1)
    val_metrics = evaluate_sentiment_model(y_val, val_preds, val_probs)
    save_json(val_metrics, results_dir / "validation_metrics.json")
    
    # 6. Evaluate on fixed 3,000-sample Evaluation Subset
    eval_start = time.time()
    eval_probs = clf.predict_proba(X_eval)
    eval_preds = np.argmax(eval_probs, axis=1)
    eval_duration = time.time() - eval_start
    
    eval_metrics = evaluate_sentiment_model(y_eval, eval_preds, eval_probs)
    eval_metrics["inference_latency_per_sample_ms"] = float((eval_duration / len(X_eval)) * 1000.0)
    save_json(eval_metrics, results_dir / "evaluation_metrics.json")
    
    # 7. Generate Predictions Parquet for Evaluation Subset
    pred_df = eval_df.copy()
    pred_df["predicted_label"] = eval_preds
    pred_df["predicted_sentiment"] = [settings.label_to_sentiment[l] for l in eval_preds]
    pred_df["confidence"] = np.max(eval_probs, axis=1)
    pred_df["prob_negative"] = eval_probs[:, 0]
    pred_df["prob_neutral"] = eval_probs[:, 1]
    pred_df["prob_positive"] = eval_probs[:, 2]
    save_dataframe(pred_df, results_dir / "predictions.parquet")
    
    # 8. Visualizations
    plot_training_history(history, results_dir)
    plot_confusion_matrix(
        eval_metrics["confusion_matrix"],
        title="Bidirectional LSTM Confusion Matrix (3,000 Eval Subset)",
        output_path=results_dir / "confusion_matrix.png"
    )
    
    # Plot Confidence Distribution
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    sns.histplot(pred_df["confidence"], bins=15, kde=True, color="#9b59b6", ax=ax)
    ax.set_title("BiLSTM Confidence Distribution (3,000 Eval Subset)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Confidence (Max Softmax Probability)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    plt.tight_layout()
    plt.savefig(results_dir / "confidence_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # 9. Model and Experiment Metadata
    total_duration = time.time() - start_time
    model_metadata = {
        "model_name": "Bidirectional_LSTM",
        "version": "1.0.0",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "random_seed": settings.random_seed,
        "device_info": clf.device_info,
        "architecture": {
            "vocab_size": len(tokenizer.word_index),
            "max_length": settings.lstm_max_sequence_length,
            "embedding_dim": settings.lstm_embedding_dim,
            "lstm_units": settings.lstm_units,
            "dropout": 0.3,
            "dense_units": 64,
            "output_classes": 3,
        },
        "training_config": {
            "batch_size": settings.lstm_batch_size,
            "epochs_configured": settings.lstm_epochs,
            "epochs_completed": len(history["loss"]),
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "loss": "sparse_categorical_crossentropy",
        },
        "timings_seconds": {
            "tokenizer_fitting": round(tokenizer_duration, 3),
            "training_duration": round(training_duration, 3),
            "evaluation_duration": round(eval_duration, 3),
            "total_duration": round(total_duration, 3),
        },
    }
    save_json(model_metadata, models_dir / "model_metadata.json")
    
    logger.info("=" * 60)
    logger.info("LSTM PIPELINE COMPLETED IN %.2f SECONDS", total_duration)
    logger.info("=" * 60)
    
    return {
        "history": history,
        "val_metrics": val_metrics,
        "eval_metrics": eval_metrics,
        "model_metadata": model_metadata,
    }


if __name__ == "__main__":
    run_lstm_pipeline()
