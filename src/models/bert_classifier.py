"""Hugging Face BERT classifier and inference service for 3-class sentiment analysis."""
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from config.settings import settings
from config.logging_config import setup_logger
from src.models.base import BaseSentimentModel
from src.models.common_interface import ModelPredictionResult

logger = setup_logger(__name__)


def get_pytorch_device() -> Tuple[str, str]:
    """Inspect and return the best available compute device for PyTorch.
    
    Returns:
        Tuple[str, str]: (device_type, device_description)
    """
    try:
        import torch
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            desc = f"GPU: {dev_name} ({vram_gb} GB VRAM)"
            logger.info("PyTorch CUDA available: %s", desc)
            return "cuda", desc
        else:
            desc = "CPU (CUDA not available in PyTorch build)"
            logger.info("PyTorch running on %s", desc)
            return "cpu", desc
    except OSError as e:
        # Windows Defender Application Control (WDAC) DLL loading block
        desc = f"BLOCKED_BY_OS_POLICY: {e}"
        logger.warning("PyTorch native execution blocked by OS Application Control: %s", e)
        return "blocked", desc
    except ImportError:
        desc = "PyTorch not installed"
        logger.warning("PyTorch is not installed in the active environment.")
        return "none", desc


class BertSentimentClassifier:
    """Hugging Face BERT sequence classification model for sentiment analysis."""

    def __init__(
        self,
        model_name: str = settings.bert_model_name,
        num_labels: int = 3,
        max_length: int = settings.bert_max_length,
        device: Optional[str] = None,
        random_seed: int = settings.random_seed,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.random_seed = random_seed
        
        detected_dev, self.device_description = get_pytorch_device()
        self.device = device or (detected_dev if detected_dev in ("cuda", "cpu") else "cpu")
        
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.is_trained: bool = False

    def load_pretrained(self) -> "BertSentimentClassifier":
        """Load pretrained base model and tokenizer from Hugging Face Hub."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            logger.info("Loading pretrained '%s' from Hugging Face Hub...", self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                id2label={i: name for i, name in settings.label_to_sentiment.items()},
                label2id=settings.sentiment_to_label,
            )
            
            if self.device != "blocked" and self.device != "none":
                self.model.to(self.device)
                
            logger.info("Pretrained BERT model '%s' loaded successfully.", self.model_name)
            return self
        except OSError as e:
            logger.error("Failed to load pretrained BERT due to OS/DLL policy: %s", str(e))
            raise RuntimeError(
                f"Local PyTorch execution is restricted by host Windows Application Control: {e}. "
                "Use the provided Google Colab notebook (notebooks/01_colab_bert_training.ipynb) "
                "to fine-tune BERT on free GPU and export checkpoints."
            ) from e

    def tokenize_texts(
        self,
        texts: List[str],
        padding: Union[bool, str] = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt"
    ) -> Dict[str, Any]:
        """Tokenize a list of review texts with padding and truncation."""
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
        sanitized = [str(t).strip() if t is not None else "" for t in texts]
        return self.tokenizer(
            sanitized,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors=return_tensors,
        )

    def predict_proba(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Predict class probability distributions [P(Neg), P(Neu), P(Pos)]."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("BERT model and tokenizer must be loaded before calling predict_proba.")
            
        import torch
        
        if isinstance(texts, str):
            texts = [texts]
            
        inputs = self.tokenize_texts(texts, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            
        # Ensure row sums equal 1.0 against floating point inaccuracies
        row_sums = np.sum(probs, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def predict(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Predict integer class labels (0=Negative, 1=Neutral, 2=Positive)."""
        probs = self.predict_proba(texts)
        return np.argmax(probs, axis=1)

    def save(self, output_dir: Union[str, Path]) -> Path:
        """Save model checkpoint, configuration, and tokenizer to disk."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Cannot save an uninitialized model.")
            
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        logger.info("Saved fine-tuned BERT checkpoint and tokenizer to %s", path)
        return path

    def load(self, model_dir: Union[str, Path]) -> "BertSentimentClassifier":
        """Load fine-tuned model and tokenizer from saved directory."""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        path = Path(model_dir)
        if not path.exists():
            raise FileNotFoundError(f"BERT model directory not found at {path}")
            
        logger.info("Loading fine-tuned BERT model from %s...", path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path))
        
        if self.device in ("cuda", "cpu"):
            self.model.to(self.device)
            
        self.is_trained = True
        logger.info("Fine-tuned BERT model loaded successfully from %s", path)
        return self


class BertSentimentPredictor:
    """Inference predictor service for BERT returning standardized ModelPredictionResult."""

    def __init__(
        self,
        classifier: BertSentimentClassifier,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.classifier = classifier
        self.metadata = metadata or {}
        self.model_version = self.metadata.get("version", "1.0.0")

    @classmethod
    def from_artifacts(
        cls,
        model_dir: Optional[Union[str, Path]] = None
    ) -> "BertSentimentPredictor":
        """Load predictor from saved model artifacts."""
        path = Path(model_dir or (settings.models_dir / "bert"))
        clf = BertSentimentClassifier()
        clf.load(path)
        from src.utils.file_io import load_json
        meta_file = path / "model_metadata.json"
        meta = load_json(meta_file) if meta_file.exists() else {}
        return cls(classifier=clf, metadata=meta)

    def predict_one(self, review: str) -> ModelPredictionResult:
        """Predict sentiment for a single review string."""
        start_time = time.time()
        sanitized = str(review).strip() if review else ""
        
        probs = self.classifier.predict_proba([sanitized])[0]
        pred_label = int(np.argmax(probs))
        pred_sentiment = settings.label_to_sentiment[pred_label]
        duration_ms = float((time.time() - start_time) * 1000.0)
        
        prob_dict = {
            settings.label_to_sentiment[k]: float(probs[k])
            for k in range(3)
        }
        
        return ModelPredictionResult(
            model_name="BERT_base_uncased",
            model_version=self.model_version,
            review=sanitized,
            predicted_label=pred_label,
            predicted_sentiment=pred_sentiment,
            confidence=float(np.max(probs)),
            class_probabilities=prob_dict,
            inference_duration_ms=duration_ms,
            is_calibrated=False,
            metadata={"device": self.classifier.device},
        )

    def predict_batch(self, reviews: List[str]) -> List[ModelPredictionResult]:
        """Predict sentiment for a list of review strings."""
        start_time = time.time()
        sanitized = [str(r).strip() if r is not None else "" for r in reviews]
        if not sanitized:
            return []
            
        probs = self.classifier.predict_proba(sanitized)
        pred_labels = np.argmax(probs, axis=1)
        total_duration_ms = float((time.time() - start_time) * 1000.0)
        per_sample_ms = total_duration_ms / max(1, len(sanitized))
        
        results = []
        for i, text in enumerate(sanitized):
            lbl = int(pred_labels[i])
            sent = settings.label_to_sentiment[lbl]
            prob_dict = {
                settings.label_to_sentiment[k]: float(probs[i, k])
                for k in range(3)
            }
            results.append(
                ModelPredictionResult(
                    model_name="BERT_base_uncased",
                    model_version=self.model_version,
                    review=text,
                    predicted_label=lbl,
                    predicted_sentiment=sent,
                    confidence=float(np.max(probs[i])),
                    class_probabilities=prob_dict,
                    inference_duration_ms=per_sample_ms,
                    is_calibrated=False,
                    metadata={"device": self.classifier.device},
                )
            )
        return results

