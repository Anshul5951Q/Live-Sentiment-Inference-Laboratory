"""Training and evaluation pipeline for Hugging Face BERT classifier."""
import datetime
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config.settings import settings
from config.logging_config import setup_logger
from src.models.bert_classifier import BertSentimentClassifier, get_pytorch_device
from src.evaluation.classification_metrics import evaluate_sentiment_model
from src.evaluation.visualizer import plot_confusion_matrix
from src.utils.file_io import save_json, save_dataframe, load_dataframe
from src.utils.seed import set_seed

logger = setup_logger(__name__)


def plot_bert_training_history(history: Dict[str, list], output_dir: Path) -> Tuple[Path, Path]:
    """Plot training vs validation loss and validation macro-F1 curves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Loss Curve
    loss_path = output_dir / "loss_curve.png"
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "o-", label="Training Loss", color="#e74c3c", linewidth=2)
    ax.plot(epochs, history["val_loss"], "s--", label="Validation Loss", color="#3498db", linewidth=2)
    ax.set_title("BERT: Training vs Validation Loss", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # 2. Macro F1 Curve
    f1_path = output_dir / "f1_curve.png"
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(epochs, history["val_macro_f1"], "d-", label="Validation Macro F1", color="#2ecc71", linewidth=2)
    ax.set_title("BERT: Validation Macro F1 Score Across Epochs", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Macro F1", fontsize=10)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f1_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    logger.info("Saved BERT loss curve to %s and F1 curve to %s", loss_path, f1_path)
    return loss_path, f1_path


def run_bert_training_pipeline(
    batch_size: int = settings.bert_batch_size,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    max_length: int = settings.bert_max_length,
) -> Dict[str, Any]:
    """Execute BERT fine-tuning pipeline with validation and 3,000-sample benchmark evaluation."""
    start_time = time.time()
    set_seed(settings.random_seed)
    
    models_dir = settings.models_dir / "bert"
    results_dir = settings.results_dir / "bert"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("STARTING HUGGING FACE BERT FINE-TUNING PIPELINE")
    logger.info("=" * 60)
    
    device_type, device_desc = get_pytorch_device()
    logger.info("Hardware execution environment: %s (%s)", device_type, device_desc)
    
    if device_type in ("blocked", "none"):
        logger.error(
            "Local PyTorch execution is unavailable (%s). "
            "Please run the Google Colab GPU notebook: notebooks/01_colab_bert_training.ipynb",
            device_desc
        )
        return {
            "status": "BLOCKED_BY_HARDWARE_OR_POLICY",
            "message": device_desc,
            "colab_notebook_path": "notebooks/01_colab_bert_training.ipynb"
        }
        
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    from torch.optim import AdamW
    
    # 1. Load dataset splits
    train_df = load_dataframe(settings.data_dir / "train_real.parquet")
    val_df = load_dataframe(settings.data_dir / "val_real.parquet")
    eval_df = load_dataframe(settings.data_dir / "eval_subset_real.parquet")
    
    logger.info(
        "Loaded splits: Train=%d, Validation=%d, Evaluation Subset=%d",
        len(train_df), len(val_df), len(eval_df)
    )
    
    # 2. Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(settings.bert_model_name)
    
    class ReviewDataset(Dataset):
        def __init__(self, texts: List[str], labels: List[int]):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            self.labels = torch.tensor(labels, dtype=torch.long)
            
        def __len__(self):
            return len(self.labels)
            
        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item
            
    train_dataset = ReviewDataset(train_df["review"].tolist(), train_df["label"].tolist())
    val_dataset = ReviewDataset(val_df["review"].tolist(), val_df["label"].tolist())
    eval_dataset = ReviewDataset(eval_df["review"].tolist(), eval_df["label"].tolist())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Model setup
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.bert_model_name,
        num_labels=3,
        id2label={i: name for i, name in settings.label_to_sentiment.items()},
        label2id=settings.sentiment_to_label,
    )
    model.to(device_type)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    # 4. Training loop with early stopping
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_val_loss = float("inf")
    best_epoch = 0
    patience = 2
    patience_counter = 0
    
    train_start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device_type)
            attention_mask = batch["attention_mask"].to(device_type)
            labels = batch["labels"].to(device_type)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation pass
        model.eval()
        total_val_loss = 0.0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device_type)
                attention_mask = batch["attention_mask"].to(device_type)
                labels = batch["labels"].to(device_type)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())
                
        avg_val_loss = total_val_loss / len(val_loader)
        val_eval = evaluate_sentiment_model(val_targets, val_preds)
        val_macro_f1 = val_eval["macro_f1"]
        
        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(float(avg_val_loss))
        history["val_macro_f1"].append(float(val_macro_f1))
        
        logger.info(
            "Epoch %d/%d - Train Loss: %.4f | Val Loss: %.4f | Val Macro F1: %.4f",
            epoch, epochs, avg_train_loss, avg_val_loss, val_macro_f1
        )
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            model.save_pretrained(str(models_dir))
            tokenizer.save_pretrained(str(models_dir))
            logger.info("Saved best BERT checkpoint at epoch %d (val_loss=%.4f)", epoch, avg_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break
                
    training_duration = time.time() - train_start_time
    save_json(history, results_dir / "training_history.json")
    
    # 5. Reload best model for final evaluation
    model = AutoModelForSequenceClassification.from_pretrained(str(models_dir))
    model.to(device_type)
    model.eval()
    
    # 6. Evaluate on 3,000-sample Evaluation Subset
    eval_start_time = time.time()
    eval_preds, eval_targets, eval_probs = [], [], []
    
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device_type)
            attention_mask = batch["attention_mask"].to(device_type)
            labels = batch["labels"].to(device_type)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            eval_probs.extend(probs)
            eval_preds.extend(preds)
            eval_targets.extend(labels.cpu().numpy())
            
    eval_duration = time.time() - eval_start_time
    eval_probs_arr = np.array(eval_probs)
    eval_metrics = evaluate_sentiment_model(eval_targets, eval_preds, y_prob=eval_probs_arr)
    eval_metrics["inference_latency_per_sample_ms"] = float((eval_duration / len(eval_df)) * 1000.0)
    save_json(eval_metrics, results_dir / "evaluation_metrics.json")
    
    # 7. Generate Predictions Parquet for Evaluation Subset
    pred_df = eval_df.copy()
    pred_df["predicted_label"] = eval_preds
    pred_df["predicted_sentiment"] = [settings.label_to_sentiment[l] for l in eval_preds]
    pred_df["confidence"] = np.max(eval_probs_arr, axis=1)
    pred_df["prob_negative"] = eval_probs_arr[:, 0]
    pred_df["prob_neutral"] = eval_probs_arr[:, 1]
    pred_df["prob_positive"] = eval_probs_arr[:, 2]
    save_dataframe(pred_df, results_dir / "predictions.parquet")
    
    # 8. Visualizations
    plot_bert_training_history(history, results_dir)
    plot_confusion_matrix(
        eval_metrics["confusion_matrix"],
        title="BERT Confusion Matrix (3,000 Eval Subset)",
        output_path=results_dir / "confusion_matrix.png"
    )
    
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    sns.histplot(pred_df["confidence"], bins=15, kde=True, color="#2ecc71", ax=ax)
    ax.set_title("BERT Confidence Distribution (3,000 Eval Subset)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Confidence (Softmax Probability)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    plt.tight_layout()
    plt.savefig(results_dir / "confidence_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # 9. Model Metadata
    total_duration = time.time() - start_time
    model_metadata = {
        "model_name": settings.bert_model_name,
        "version": "1.0.0",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "random_seed": settings.random_seed,
        "device_info": device_desc,
        "training_config": {
            "batch_size": batch_size,
            "epochs_configured": epochs,
            "epochs_completed": len(history["train_loss"]),
            "best_epoch": best_epoch,
            "learning_rate": learning_rate,
            "optimizer": "AdamW",
            "max_length": max_length,
        },
        "timings_seconds": {
            "training_duration": round(training_duration, 3),
            "evaluation_duration": round(eval_duration, 3),
            "total_duration": round(total_duration, 3),
        },
    }
    save_json(model_metadata, models_dir / "model_metadata.json")
    
    logger.info("BERT pipeline completed successfully in %.2f seconds.", total_duration)
    return {
        "status": "COMPLETED",
        "history": history,
        "eval_metrics": eval_metrics,
        "model_metadata": model_metadata,
    }


def ingest_colab_bert_artifacts(source_dir: Union[str, Path]) -> Dict[str, Any]:
    """Ingest BERT artifacts exported from Google Colab into project directory."""
    src = Path(source_dir)
    target_models = settings.models_dir / "bert"
    target_results = settings.results_dir / "bert"
    target_models.mkdir(parents=True, exist_ok=True)
    target_results.mkdir(parents=True, exist_ok=True)
    
    logger.info("Ingesting Colab BERT artifacts from %s...", src)
    copied = []
    import shutil
    for f in src.glob("*"):
        if f.is_file():
            results_keywords = ("metrics", "history", "confusion", "curve", "prediction")
            if f.suffix in (".json", ".png", ".parquet") and any(k in f.name for k in results_keywords):
                dest = target_results / f.name
            else:
                dest = target_models / f.name
            shutil.copy2(f, dest)
            copied.append(str(dest))
            
    logger.info("Ingested %d artifact files from Colab.", len(copied))
    return {"ingested_files": copied, "count": len(copied)}

