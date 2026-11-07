"""Condition A (Real-Only) vs Condition B (Real + Synthetic) retraining pipeline.

Demonstrates the empirical effects of Generative AI synthetic data augmentation
on discriminative sentiment classifiers (Logistic Regression and Keras BiLSTM).
"""
import datetime
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from config.settings import settings
from config.logging_config import setup_logger
from src.models.tfidf_vectorizer import TextTfidfVectorizer
from src.models.logistic_regression import LogisticRegressionClassifier
from src.models.calibration import MulticlassCalibrator
from src.models.text_tokenizer import TextTokenizer
from src.models.lstm_classifier import KerasBiLSTMClassifier
from src.evaluation.classification_metrics import evaluate_sentiment_model
from src.evaluation.visualizer import plot_confusion_matrix
from src.utils.file_io import save_json, save_dataframe, load_dataframe
from src.utils.seed import set_seed

logger = setup_logger(__name__)


def create_augmented_dataset(
    real_train_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    synthetic_fraction: float = 1.0,
    seed: int = 42
) -> pd.DataFrame:
    """Combine real training data with a controlled fraction of synthetic reviews.

    Args:
        real_train_df: Real training DataFrame (19,200 rows).
        synthetic_df: Validated synthetic DataFrame (500 rows).
        synthetic_fraction: Fraction of synthetic reviews to include (e.g. 0.25, 0.5, 1.0).
        seed: Random seed for sampling and shuffling.

    Returns:
        Combined, reproducible, stratified-augmented training DataFrame.
    """
    if synthetic_df.empty or synthetic_fraction <= 0.0:
        return real_train_df.copy()

    # Sample stratified fraction of synthetic reviews
    if synthetic_fraction < 1.0:
        sampled_synth = pd.concat(
            [grp.sample(frac=synthetic_fraction, random_state=seed) for _, grp in synthetic_df.groupby("label")],
            ignore_index=True
        )
    else:
        sampled_synth = synthetic_df.copy()

    # Standardize column schema
    cols = ["review", "label", "sentiment", "source"]
    real_clean = real_train_df[cols].copy()
    synth_clean = sampled_synth[cols].copy()

    augmented_df = pd.concat([real_clean, synth_clean], ignore_index=True)
    # Shuffle deterministically
    augmented_df = augmented_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return augmented_df


def retrain_logistic_regression(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    condition_name: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Train, calibrate, and evaluate Logistic Regression on a specific dataset condition."""
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # 1. Fit TF-IDF Vectorizer
    vectorizer = TextTfidfVectorizer(
        max_features=settings.tfidf_max_features,
        ngram_range=settings.tfidf_ngram_range,
    )
    X_train = vectorizer.fit_transform(train_df["review"])
    X_val = vectorizer.transform(val_df["review"])
    X_eval = vectorizer.transform(eval_df["review"])
    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_eval = eval_df["label"].values

    # 2. Fit Logistic Regression
    clf = LogisticRegressionClassifier(
        C=1.0,
        max_iter=1000,
        random_state=settings.random_seed,
    )
    clf.fit(X_train, y_train)

    # 3. Multiclass Calibration on untouched validation set
    calibrator = MulticlassCalibrator(method="sigmoid")
    calibrator.fit(clf.model, X_val, y_val)

    training_duration = time.time() - start_time

    # 4. Evaluate Calibrated Model on Evaluation Set
    eval_start = time.time()
    cal_probs = calibrator.predict_proba(X_eval)
    y_pred = calibrator.predict(X_eval)
    eval_duration = time.time() - eval_start

    metrics = evaluate_sentiment_model(y_eval, y_pred, y_prob=cal_probs)
    metrics["training_duration_seconds"] = round(training_duration, 2)
    metrics["inference_latency_per_sample_ms"] = float((eval_duration / len(eval_df)) * 1000.0)
    metrics["condition"] = condition_name
    metrics["train_size"] = len(train_df)
    metrics["eval_size"] = len(eval_df)

    # Save retrained model artifacts
    vectorizer.save(output_dir / "tfidf_vectorizer.joblib")
    clf.save(output_dir / "logistic_regression.joblib")
    calibrator.save(output_dir / "calibrated_logistic_regression.joblib")
    save_json(metrics, output_dir / "evaluation_metrics.json")

    return metrics


def retrain_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    condition_name: str,
    output_dir: Path,
    epochs: int = 6,
) -> Dict[str, Any]:
    """Train and evaluate Keras BiLSTM on a specific dataset condition."""
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # 1. Fit Tokenizer
    tokenizer = TextTokenizer(
        vocab_size=settings.lstm_vocab_size,
        max_length=settings.lstm_max_sequence_length,
    )
    tokenizer.fit(train_df["review"].values)
    X_train = tokenizer.transform(train_df["review"].values)
    X_val = tokenizer.transform(val_df["review"].values)
    X_eval = tokenizer.transform(eval_df["review"].values)
    y_train = train_df["label"].values.astype(np.int32)
    y_val = val_df["label"].values.astype(np.int32)
    y_eval = eval_df["label"].values.astype(np.int32)

    # 2. Build and Train BiLSTM Classifier
    classifier = KerasBiLSTMClassifier(
        vocab_size=len(tokenizer.word_index),
        embedding_dim=settings.lstm_embedding_dim,
        lstm_units=settings.lstm_units,
        dropout=0.3,
        max_length=settings.lstm_max_sequence_length,
        random_seed=settings.random_seed,
    )
    checkpoint_path = output_dir / "lstm_model.keras"
    history = classifier.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=settings.lstm_batch_size,
        checkpoint_path=checkpoint_path,
    )
    training_duration = time.time() - start_time

    # 3. Evaluate on untouched Evaluation Set
    eval_start = time.time()
    probs = classifier.predict_proba(X_eval)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    y_pred = np.argmax(probs, axis=1)
    eval_duration = time.time() - eval_start

    metrics = evaluate_sentiment_model(y_eval, y_pred, y_prob=probs)
    metrics["training_duration_seconds"] = round(training_duration, 2)
    metrics["inference_latency_per_sample_ms"] = float((eval_duration / len(eval_df)) * 1000.0)
    metrics["condition"] = condition_name
    metrics["train_size"] = len(train_df)
    metrics["eval_size"] = len(eval_df)
    metrics["history"] = history

    # Save model and tokenizer
    tokenizer.save(output_dir / "tokenizer.json")
    classifier.save(output_dir / "lstm_model.keras")
    save_json(metrics, output_dir / "evaluation_metrics.json")

    return metrics


def run_milestone5_experiments() -> Dict[str, Any]:
    """Execute complete real-only vs real-plus-synthetic retraining experiments."""
    logger.info("=" * 70)
    logger.info("STARTING MILESTONE 5: REAL VS REAL+SYNTHETIC RETRAINING EXPERIMENTS")
    logger.info("=" * 70)

    data_dir = settings.data_dir
    results_dir = settings.results_dir / "retraining_experiments"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    train_real_df = load_dataframe(data_dir / "train_real.parquet")
    val_real_df = load_dataframe(data_dir / "val_real.parquet")
    eval_subset_df = load_dataframe(data_dir / "eval_subset_real.parquet")
    benchmark_300_df = load_dataframe(data_dir / "gemini_benchmark_300.parquet")
    synth_path = data_dir / "synthetic_reviews.parquet"

    if not synth_path.exists():
        raise FileNotFoundError(f"Synthetic dataset not found at {synth_path}. Generate reviews first.")

    synthetic_df = load_dataframe(synth_path)
    logger.info(
        "Loaded datasets: Real Train=%d, Real Val=%d, Eval 3k=%d, Benchmark 300=%d, Synthetic=%d",
        len(train_real_df), len(val_real_df), len(eval_subset_df), len(benchmark_300_df), len(synthetic_df)
    )

    # 2. Build Condition Datasets
    # Condition A: Real Only (19,200)
    # Condition B: Real + 100% Synthetic (19,700)
    # Condition C: Real + 50% Synthetic (19,450)
    train_cond_a = train_real_df.copy()
    train_cond_b = create_augmented_dataset(train_real_df, synthetic_df, synthetic_fraction=1.0)
    train_cond_c = create_augmented_dataset(train_real_df, synthetic_df, synthetic_fraction=0.5)

    experiments = {}

    # -------------------------------------------------------------
    # Experiment 1: Logistic Regression Retraining
    # -------------------------------------------------------------
    logger.info("Retraining Logistic Regression: Condition A (Real-Only)...")
    lr_a_dir = settings.models_dir / "logistic_regression"
    # Load or evaluate baseline on 3k
    with open(settings.results_dir / "logistic_regression" / "evaluation_metrics.json") as f:
        lr_a_eval3k = json.load(f)["calibrated"]

    logger.info("Retraining Logistic Regression: Condition B (Real + 100% Synthetic)...")
    lr_b_dir = settings.models_dir / "logistic_regression_augmented"
    lr_b_eval3k = retrain_logistic_regression(
        train_df=train_cond_b,
        val_df=val_real_df,
        eval_df=eval_subset_df,
        condition_name="Real + 100% Synthetic",
        output_dir=lr_b_dir,
    )

    logger.info("Retraining Logistic Regression: Condition C (Real + 50% Synthetic)...")
    lr_c_dir = settings.models_dir / "logistic_regression_aug_50"
    lr_c_eval3k = retrain_logistic_regression(
        train_df=train_cond_c,
        val_df=val_real_df,
        eval_df=eval_subset_df,
        condition_name="Real + 50% Synthetic",
        output_dir=lr_c_dir,
    )

    # Also evaluate both on the 300 benchmark for direct 5-model comparison parity
    cal_a = MulticlassCalibrator().load(lr_a_dir / "calibrated_logistic_regression.joblib")
    vec_a = TextTfidfVectorizer().load(lr_a_dir / "tfidf_vectorizer.joblib")

    cal_b = MulticlassCalibrator().load(lr_b_dir / "calibrated_logistic_regression.joblib")
    vec_b = TextTfidfVectorizer().load(lr_b_dir / "tfidf_vectorizer.joblib")

    X_b300_a = vec_a.transform(benchmark_300_df["review"])
    p_b300_a = cal_a.predict_proba(X_b300_a)
    lr_a_eval300 = evaluate_sentiment_model(benchmark_300_df["label"].values, cal_a.predict(X_b300_a), y_prob=p_b300_a)

    X_b300_b = vec_b.transform(benchmark_300_df["review"])
    p_b300_b = cal_b.predict_proba(X_b300_b)
    lr_b_eval300 = evaluate_sentiment_model(benchmark_300_df["label"].values, cal_b.predict(X_b300_b), y_prob=p_b300_b)

    # -------------------------------------------------------------
    # Experiment 2: BiLSTM Retraining
    # -------------------------------------------------------------
    logger.info("Loading BiLSTM Condition A baseline...")
    with open(settings.results_dir / "lstm" / "evaluation_metrics.json") as f:
        lstm_a_eval3k = json.load(f)

    logger.info("Retraining BiLSTM: Condition B (Real + 100% Synthetic)...")
    lstm_b_dir = settings.models_dir / "lstm_augmented"
    lstm_b_eval3k = retrain_lstm(
        train_df=train_cond_b,
        val_df=val_real_df,
        eval_df=eval_subset_df,
        condition_name="Real + 100% Synthetic",
        output_dir=lstm_b_dir,
        epochs=5,
    )

    # Evaluate BiLSTM on the 300 benchmark
    lstm_a = KerasBiLSTMClassifier().load(settings.models_dir / "lstm" / "lstm_model.keras")
    tok_a = TextTokenizer().load(settings.models_dir / "lstm" / "tokenizer.json")
    seq_a = tok_a.transform(benchmark_300_df["review"].values)
    p_lstm_a = lstm_a.predict_proba(seq_a)
    p_lstm_a = p_lstm_a / np.sum(p_lstm_a, axis=1, keepdims=True)
    lstm_a_eval300 = evaluate_sentiment_model(benchmark_300_df["label"].values, np.argmax(p_lstm_a, axis=1), y_prob=p_lstm_a)

    lstm_b = KerasBiLSTMClassifier().load(lstm_b_dir / "lstm_model.keras")
    tok_b = TextTokenizer().load(lstm_b_dir / "tokenizer.json")
    seq_b = tok_b.transform(benchmark_300_df["review"].values)
    p_lstm_b = lstm_b.predict_proba(seq_b)
    p_lstm_b = p_lstm_b / np.sum(p_lstm_b, axis=1, keepdims=True)
    lstm_b_eval300 = evaluate_sentiment_model(benchmark_300_df["label"].values, np.argmax(p_lstm_b, axis=1), y_prob=p_lstm_b)

    # Compile Summary Dictionary
    summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "real_train_size": len(train_real_df),
        "synthetic_size": len(synthetic_df),
        "evaluation_subset_3000": {
            "Logistic_Regression": {
                "Condition_A_Real_Only": {
                    "accuracy": lr_a_eval3k["accuracy"],
                    "macro_f1": lr_a_eval3k["macro_f1"],
                    "brier_score": lr_a_eval3k["brier_score"],
                    "ece": lr_a_eval3k["ece"],
                    "per_class": lr_a_eval3k["per_class"],
                },
                "Condition_B_Real_Plus_100_Synthetic": {
                    "accuracy": lr_b_eval3k["accuracy"],
                    "macro_f1": lr_b_eval3k["macro_f1"],
                    "brier_score": lr_b_eval3k["brier_score"],
                    "ece": lr_b_eval3k["ece"],
                    "per_class": lr_b_eval3k["per_class"],
                },
                "Condition_C_Real_Plus_50_Synthetic": {
                    "accuracy": lr_c_eval3k["accuracy"],
                    "macro_f1": lr_c_eval3k["macro_f1"],
                    "brier_score": lr_c_eval3k["brier_score"],
                    "ece": lr_c_eval3k["ece"],
                },
                "delta_accuracy_pct": round((lr_b_eval3k["accuracy"] - lr_a_eval3k["accuracy"]) * 100, 3),
                "delta_macro_f1_pct": round((lr_b_eval3k["macro_f1"] - lr_a_eval3k["macro_f1"]) * 100, 3),
                "delta_neutral_f1_pct": round(
                    (lr_b_eval3k["per_class"]["Neutral"]["f1_score"] - lr_a_eval3k["per_class"]["Neutral"]["f1_score"]) * 100,
                    3
                ),
            },
            "Bidirectional_LSTM": {
                "Condition_A_Real_Only": {
                    "accuracy": lstm_a_eval3k["accuracy"],
                    "macro_f1": lstm_a_eval3k["macro_f1"],
                    "brier_score": lstm_a_eval3k["brier_score"],
                    "ece": lstm_a_eval3k["ece"],
                    "per_class": lstm_a_eval3k["per_class"],
                },
                "Condition_B_Real_Plus_100_Synthetic": {
                    "accuracy": lstm_b_eval3k["accuracy"],
                    "macro_f1": lstm_b_eval3k["macro_f1"],
                    "brier_score": lstm_b_eval3k["brier_score"],
                    "ece": lstm_b_eval3k["ece"],
                    "per_class": lstm_b_eval3k["per_class"],
                },
                "delta_accuracy_pct": round((lstm_b_eval3k["accuracy"] - lstm_a_eval3k["accuracy"]) * 100, 3),
                "delta_macro_f1_pct": round((lstm_b_eval3k["macro_f1"] - lstm_a_eval3k["macro_f1"]) * 100, 3),
                "delta_neutral_f1_pct": round(
                    (lstm_b_eval3k["per_class"]["Neutral"]["f1_score"] - lstm_a_eval3k["per_class"]["Neutral"]["f1_score"]) * 100,
                    3
                ),
            },
        },
        "benchmark_subset_300": {
            "Logistic_Regression": {
                "Condition_A_Real_Only": {"accuracy": lr_a_eval300["accuracy"], "macro_f1": lr_a_eval300["macro_f1"], "brier_score": lr_a_eval300["brier_score"]},
                "Condition_B_Real_Plus_100_Synthetic": {"accuracy": lr_b_eval300["accuracy"], "macro_f1": lr_b_eval300["macro_f1"], "brier_score": lr_b_eval300["brier_score"]},
                "delta_accuracy_pct": round((lr_b_eval300["accuracy"] - lr_a_eval300["accuracy"]) * 100, 3),
                "delta_macro_f1_pct": round((lr_b_eval300["macro_f1"] - lr_a_eval300["macro_f1"]) * 100, 3),
            },
            "Bidirectional_LSTM": {
                "Condition_A_Real_Only": {"accuracy": lstm_a_eval300["accuracy"], "macro_f1": lstm_a_eval300["macro_f1"], "brier_score": lstm_a_eval300["brier_score"]},
                "Condition_B_Real_Plus_100_Synthetic": {"accuracy": lstm_b_eval300["accuracy"], "macro_f1": lstm_b_eval300["macro_f1"], "brier_score": lstm_b_eval300["brier_score"]},
                "delta_accuracy_pct": round((lstm_b_eval300["accuracy"] - lstm_a_eval300["accuracy"]) * 100, 3),
                "delta_macro_f1_pct": round((lstm_b_eval300["macro_f1"] - lstm_a_eval300["macro_f1"]) * 100, 3),
            },
        },
    }

    # Save experiment report
    save_json(summary, results_dir / "augmented_experiment_report.json")

    # 3. Generate Publication Visualizations
    generate_augmented_plots(summary, results_dir)

    logger.info("Milestone 5 Retraining Experiments completed successfully.")
    return summary


def generate_augmented_plots(summary: Dict[str, Any], output_dir: Path) -> None:
    """Generate publication comparison plots for Real vs Real+Synthetic retraining."""
    eval_data = summary["evaluation_subset_3000"]
    models = ["Logistic_Regression", "Bidirectional_LSTM"]
    display_names = ["Logistic Regression", "Bidirectional LSTM"]

    # 1. Accuracy & Macro-F1 Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    x = np.arange(len(models))
    width = 0.35

    acc_a = [eval_data[m]["Condition_A_Real_Only"]["accuracy"] * 100 for m in models]
    acc_b = [eval_data[m]["Condition_B_Real_Plus_100_Synthetic"]["accuracy"] * 100 for m in models]

    f1_a = [eval_data[m]["Condition_A_Real_Only"]["macro_f1"] * 100 for m in models]
    f1_b = [eval_data[m]["Condition_B_Real_Plus_100_Synthetic"]["macro_f1"] * 100 for m in models]

    # Accuracy
    axes[0].bar(x - width/2, acc_a, width, label="Real Only (N=19,200)", color="#3498db")
    axes[0].bar(x + width/2, acc_b, width, label="Real + 500 Synth (N=19,700)", color="#2ecc71")
    axes[0].set_title("Accuracy Comparison (n=3,000 Eval)", fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(display_names)
    axes[0].set_ylim(60, 80)
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)

    # Macro-F1
    axes[1].bar(x - width/2, f1_a, width, label="Real Only (N=19,200)", color="#9b59b6")
    axes[1].bar(x + width/2, f1_b, width, label="Real + 500 Synth (N=19,700)", color="#e67e22")
    axes[1].set_title("Macro-F1 Comparison (n=3,000 Eval)", fontweight="bold")
    axes[1].set_ylabel("Macro-F1 (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(display_names)
    axes[1].set_ylim(60, 80)
    axes[1].legend(loc="lower right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_dir / "real_vs_augmented_accuracy_f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Per-Class F1 Delta Plot (Focus on Neutral Class)
    classes = ["Negative", "Neutral", "Positive"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    for i, m in enumerate(models):
        f1_class_a = [eval_data[m]["Condition_A_Real_Only"]["per_class"][c]["f1_score"] * 100 for c in classes]
        f1_class_b = [eval_data[m]["Condition_B_Real_Plus_100_Synthetic"]["per_class"][c]["f1_score"] * 100 for c in classes]
        xc = np.arange(len(classes))

        axes[i].bar(xc - width/2, f1_class_a, width, label="Real Only", color="#34495e")
        axes[i].bar(xc + width/2, f1_class_b, width, label="Real + Synthetic", color="#1abc9c")
        axes[i].set_title(f"{display_names[i]}: Per-Class F1 Score", fontweight="bold")
        axes[i].set_ylabel("F1 Score (%)")
        axes[i].set_xticks(xc)
        axes[i].set_xticklabels(classes)
        axes[i].set_ylim(50, 85)
        axes[i].legend(loc="lower right")
        axes[i].grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_dir / "real_vs_augmented_per_class_f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison plots to %s", output_dir)
