"""Training, calibration, and evaluation pipeline for Logistic Regression."""
import datetime
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
import sklearn
from config.settings import settings
from config.logging_config import setup_logger
from src.models.tfidf_vectorizer import TextTfidfVectorizer
from src.models.logistic_regression import LogisticRegressionClassifier
from src.models.calibration import MulticlassCalibrator
from src.evaluation.classification_metrics import evaluate_sentiment_model
from src.evaluation.visualizer import (
    plot_confusion_matrix,
    plot_reliability_diagram,
    plot_metrics_comparison,
    plot_confidence_distribution,
)
from src.utils.file_io import save_json, save_dataframe, load_dataframe
from src.utils.seed import set_seed

logger = setup_logger(__name__)


def run_logistic_regression_pipeline() -> Dict[str, Any]:
    """Execute the full Logistic Regression training, calibration, and evaluation pipeline."""
    pipeline_start_time = time.time()
    set_seed(settings.random_seed)
    
    # Setup artifact paths
    models_dir = settings.models_dir / "logistic_regression"
    results_dir = settings.results_dir / "logistic_regression"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("STARTING LOGISTIC REGRESSION TRAINING & CALIBRATION PIPELINE")
    logger.info("=" * 60)
    
    # 1. Load dataset splits
    train_path = settings.data_dir / "train_real.parquet"
    val_path = settings.data_dir / "val_real.parquet"
    eval_path = settings.data_dir / "eval_subset_real.parquet"
    
    logger.info("Loading dataset splits...")
    train_df = load_dataframe(train_path)
    val_df = load_dataframe(val_path)
    eval_df = load_dataframe(eval_path)
    
    logger.info(
        "Splits loaded: Train=%d, Validation=%d, Evaluation Subset=%d",
        len(train_df), len(val_df), len(eval_df)
    )
    
    # Validate schemas
    for split_name, df in [("Train", train_df), ("Val", val_df), ("Eval", eval_df)]:
        required_cols = {"review", "label", "sentiment", "source"}
        if not required_cols.issubset(df.columns):
            raise KeyError(f"{split_name} missing required columns: {required_cols - set(df.columns)}")
            
    # 2. Fit TF-IDF strictly on training reviews
    vectorizer_start = time.time()
    vectorizer = TextTfidfVectorizer(
        max_features=settings.tfidf_max_features,
        ngram_range=settings.tfidf_ngram_range,
        sublinear_tf=True
    )
    vectorizer.fit(train_df["review"].values)
    vectorizer_duration = time.time() - vectorizer_start
    
    # Save vectorizer artifact
    vec_artifact_path = models_dir / "tfidf_vectorizer.joblib"
    vectorizer.save(vec_artifact_path)
    
    # 3. Transform features
    X_train = vectorizer.transform(train_df["review"].values)
    y_train = train_df["label"].values
    
    X_val = vectorizer.transform(val_df["review"].values)
    y_val = val_df["label"].values
    
    X_eval = vectorizer.transform(eval_df["review"].values)
    y_eval = eval_df["label"].values
    
    # 4. Train uncalibrated Logistic Regression on train split
    train_start = time.time()
    classifier = LogisticRegressionClassifier(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=settings.random_seed
    )
    classifier.fit(X_train, y_train)
    training_duration = time.time() - train_start
    
    # Save uncalibrated model artifact
    uncal_model_path = models_dir / "logistic_regression.joblib"
    classifier.save(uncal_model_path)
    
    # 5. Fit multiclass probability calibration strictly on validation split
    cal_start = time.time()
    calibrator = MulticlassCalibrator(method="sigmoid")
    calibrator.fit(classifier.model, X_val, y_val)
    calibration_duration = time.time() - cal_start
    
    # Save calibrated model artifact
    cal_model_path = models_dir / "calibrated_logistic_regression.joblib"
    calibrator.save(cal_model_path)
    
    # 6. Evaluate on validation set
    logger.info("Evaluating models on validation split (%d samples)...", len(val_df))
    uncal_val_probs = classifier.predict_proba(X_val)
    uncal_val_preds = classifier.predict(X_val)
    val_uncal_metrics = evaluate_sentiment_model(y_val, uncal_val_preds, uncal_val_probs)
    
    cal_val_probs = calibrator.predict_proba(X_val)
    cal_val_preds = calibrator.predict(X_val)
    val_cal_metrics = evaluate_sentiment_model(y_val, cal_val_preds, cal_val_probs)
    
    validation_results = {
        "uncalibrated": val_uncal_metrics,
        "calibrated": val_cal_metrics,
    }
    save_json(validation_results, results_dir / "validation_metrics.json")
    
    # 7. Evaluate on fixed 3,000 evaluation subset
    eval_start = time.time()
    logger.info("Evaluating models on fixed 3,000-review evaluation subset...")
    uncal_eval_probs = classifier.predict_proba(X_eval)
    uncal_eval_preds = classifier.predict(X_eval)
    eval_uncal_metrics = evaluate_sentiment_model(y_eval, uncal_eval_preds, uncal_eval_probs)
    
    cal_eval_probs = calibrator.predict_proba(X_eval)
    cal_eval_preds = calibrator.predict(X_eval)
    eval_cal_metrics = evaluate_sentiment_model(y_eval, cal_eval_preds, cal_eval_probs)
    evaluation_duration = time.time() - eval_start
    
    evaluation_results = {
        "uncalibrated": eval_uncal_metrics,
        "calibrated": eval_cal_metrics,
        "comparison": {
            "accuracy_delta": eval_cal_metrics["accuracy"] - eval_uncal_metrics["accuracy"],
            "macro_f1_delta": eval_cal_metrics["macro_f1"] - eval_uncal_metrics["macro_f1"],
            "brier_score_uncalibrated": eval_uncal_metrics["brier_score"],
            "brier_score_calibrated": eval_cal_metrics["brier_score"],
            "brier_improvement": eval_uncal_metrics["brier_score"] - eval_cal_metrics["brier_score"],
            "log_loss_uncalibrated": eval_uncal_metrics["log_loss"],
            "log_loss_calibrated": eval_cal_metrics["log_loss"],
            "log_loss_improvement": eval_uncal_metrics["log_loss"] - eval_cal_metrics["log_loss"],
            "ece_uncalibrated": eval_uncal_metrics["ece"],
            "ece_calibrated": eval_cal_metrics["ece"],
            "ece_improvement": eval_uncal_metrics["ece"] - eval_cal_metrics["ece"],
        }
    }
    save_json(evaluation_results, results_dir / "evaluation_metrics.json")
    
    # 8. Generate and save evaluation subset predictions DataFrame
    logger.info("Generating predictions parquet for evaluation subset...")
    pred_df = eval_df.copy()
    pred_df["uncal_pred_label"] = uncal_eval_preds
    pred_df["uncal_pred_sentiment"] = [settings.label_to_sentiment[l] for l in uncal_eval_preds]
    pred_df["uncal_confidence"] = np.max(uncal_eval_probs, axis=1)
    pred_df["uncal_prob_negative"] = uncal_eval_probs[:, 0]
    pred_df["uncal_prob_neutral"] = uncal_eval_probs[:, 1]
    pred_df["uncal_prob_positive"] = uncal_eval_probs[:, 2]
    
    pred_df["cal_pred_label"] = cal_eval_preds
    pred_df["cal_pred_sentiment"] = [settings.label_to_sentiment[l] for l in cal_eval_preds]
    pred_df["cal_confidence"] = np.max(cal_eval_probs, axis=1)
    pred_df["cal_prob_negative"] = cal_eval_probs[:, 0]
    pred_df["cal_prob_neutral"] = cal_eval_probs[:, 1]
    pred_df["cal_prob_positive"] = cal_eval_probs[:, 2]
    
    predictions_path = results_dir / "predictions.parquet"
    save_dataframe(pred_df, predictions_path)
    
    # 9. Generate all 6 visualization plots
    logger.info("Generating publication-quality visualization plots...")
    plot_confusion_matrix(
        eval_uncal_metrics["confusion_matrix"],
        title="Uncalibrated Logistic Regression Confusion Matrix",
        output_path=results_dir / "confusion_matrix_uncalibrated.png"
    )
    plot_confusion_matrix(
        eval_cal_metrics["confusion_matrix"],
        title="Calibrated Logistic Regression (Sigmoid) Confusion Matrix",
        output_path=results_dir / "confusion_matrix_calibrated.png"
    )
    plot_reliability_diagram(
        eval_uncal_metrics["reliability_curves"],
        title="Uncalibrated Multiclass Reliability Diagram",
        output_path=results_dir / "calibration_uncalibrated.png"
    )
    plot_reliability_diagram(
        eval_cal_metrics["reliability_curves"],
        title="Calibrated Multiclass Reliability Diagram (Sigmoid)",
        output_path=results_dir / "calibration_calibrated.png"
    )
    plot_metrics_comparison(
        eval_uncal_metrics,
        eval_cal_metrics,
        output_path=results_dir / "metrics_comparison.png"
    )
    plot_confidence_distribution(
        eval_uncal_metrics["confidence_stats"],
        eval_cal_metrics["confidence_stats"],
        output_path=results_dir / "confidence_distribution.png"
    )
    
    # 10. Compile and save model metadata
    model_metadata = {
        "model_name": "LogisticRegression",
        "version": "1.0.0",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "random_seed": settings.random_seed,
        "tfidf_parameters": {
            "max_features": settings.tfidf_max_features,
            "ngram_range": list(settings.tfidf_ngram_range),
            "sublinear_tf": True,
            "vocabulary_size": len(vectorizer.vectorizer.vocabulary_),
        },
        "classifier_parameters": {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "class_weight": None,
            "classes": [0, 1, 2],
        },
        "calibration": {
            "method": "sigmoid",
            "calibration_dataset": "val_real.parquet",
            "calibration_samples": len(val_df),
        },
        "environment": {
            "python_version": sys.version,
            "sklearn_version": sklearn.__version__,
            "platform": platform.platform(),
        },
    }
    save_json(model_metadata, models_dir / "model_metadata.json")
    
    # 11. Compile and save experiment metadata
    total_duration = time.time() - pipeline_start_time
    def _safe_path_str(p: Path) -> str:
        try:
            return str(p.relative_to(settings.base_dir))
        except ValueError:
            return str(p)

    experiment_metadata = {
        "experiment_id": f"EXP-LR-BASELINE-{settings.random_seed}",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "random_seed": settings.random_seed,
        "dataset": {
            "name": settings.dataset_name,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "eval_subset_samples": len(eval_df),
            "class_distribution_eval": eval_df["label"].value_counts().to_dict(),
        },
        "timings_seconds": {
            "vectorizer_fitting": round(vectorizer_duration, 3),
            "classifier_training": round(training_duration, 3),
            "calibration_fitting": round(calibration_duration, 3),
            "evaluation_duration": round(evaluation_duration, 3),
            "total_pipeline_duration": round(total_duration, 3),
        },
        "evaluation_summary": {
            "uncalibrated": {
                "accuracy": eval_uncal_metrics["accuracy"],
                "macro_f1": eval_uncal_metrics["macro_f1"],
                "brier_score": eval_uncal_metrics["brier_score"],
                "log_loss": eval_uncal_metrics["log_loss"],
                "ece": eval_uncal_metrics["ece"],
            },
            "calibrated": {
                "accuracy": eval_cal_metrics["accuracy"],
                "macro_f1": eval_cal_metrics["macro_f1"],
                "brier_score": eval_cal_metrics["brier_score"],
                "log_loss": eval_cal_metrics["log_loss"],
                "ece": eval_cal_metrics["ece"],
            },
            "comparison": evaluation_results["comparison"],
        },
        "artifacts": {
            "tfidf_vectorizer": _safe_path_str(vec_artifact_path),
            "uncalibrated_model": _safe_path_str(uncal_model_path),
            "calibrated_model": _safe_path_str(cal_model_path),
            "predictions": _safe_path_str(predictions_path),
        }
    }
    save_json(experiment_metadata, results_dir / "experiment_metadata.json")
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY IN %.2f SECONDS", total_duration)
    logger.info("=" * 60)
    
    return {
        "model_metadata": model_metadata,
        "experiment_metadata": experiment_metadata,
        "evaluation_results": evaluation_results,
    }


if __name__ == "__main__":
    run_logistic_regression_pipeline()
