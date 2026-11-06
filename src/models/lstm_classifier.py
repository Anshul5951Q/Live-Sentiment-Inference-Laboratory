"""Keras Bidirectional LSTM sentiment classifier component."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from config.settings import settings
from config.logging_config import setup_logger
from src.models.base import BaseSentimentModel
from src.models.text_tokenizer import TextTokenizer

logger = setup_logger(__name__)


def setup_tensorflow_devices() -> str:
    """Inspect and configure TensorFlow compute devices (GPU / CPU)."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                device_str = f"GPU: {len(gpus)} device(s) configured with dynamic memory growth."
                logger.info(device_str)
                return device_str
            except RuntimeError as e:
                logger.warning("GPU memory growth could not be set: %s. Using default device placement.", str(e))
                return f"GPU available: {len(gpus)}"
        else:
            logger.info("No GPU detected by TensorFlow. Running on CPU.")
            return "CPU (no GPU detected)"
    except Exception as e:
        logger.warning("TensorFlow device configuration check failed: %s. Falling back to default CPU.", str(e))
        return "CPU fallback"


class KerasBiLSTMClassifier:
    """Bidirectional LSTM sentiment classifier implemented in TensorFlow/Keras."""

    def __init__(
        self,
        vocab_size: int = settings.lstm_vocab_size,
        max_length: int = settings.lstm_max_sequence_length,
        embedding_dim: int = settings.lstm_embedding_dim,
        lstm_units: int = settings.lstm_units,
        dense_units: int = 64,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        random_seed: int = settings.random_seed,
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        
        self.model: Optional[Any] = None
        self.is_trained: bool = False
        self.history_: Dict[str, List[float]] = {}
        self.device_info: str = setup_tensorflow_devices()

    def build_model(self) -> Any:
        """Construct the Keras Bidirectional LSTM architecture."""
        import tensorflow as tf
        from tensorflow.keras import layers, models, optimizers
        
        tf.random.set_seed(self.random_seed)
        
        inputs = layers.Input(shape=(self.max_length,), dtype=tf.int32, name="input_ids")
        
        # 1. Embedding Layer
        x = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name="embedding"
        )(inputs)
        
        # 2. Bidirectional LSTM Layer
        # Note: recurrent_dropout=0 enables cuDNN hardware acceleration
        x = layers.Bidirectional(
            layers.LSTM(
                units=self.lstm_units,
                dropout=self.dropout,
                recurrent_dropout=0.0,
                return_sequences=False,
                name="lstm"
            ),
            name="bidirectional_lstm"
        )(x)
        
        # 3. Dense & Dropout Layers
        x = layers.Dropout(rate=self.dropout, name="dropout_1")(x)
        x = layers.Dense(units=self.dense_units, activation="relu", name="dense_features")(x)
        x = layers.Dropout(rate=self.dropout, name="dropout_2")(x)
        
        # 4. 3-class Softmax Output
        outputs = layers.Dense(units=3, activation="softmax", name="sentiment_output")(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name="BiLSTM_Sentiment_Classifier")
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        self.model = model
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = settings.lstm_batch_size,
        epochs: int = settings.lstm_epochs,
        checkpoint_path: Optional[Union[str, Path]] = None,
        callbacks: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """Train the Bidirectional LSTM model with EarlyStopping and Checkpointing."""
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        
        if self.model is None:
            self.build_model()
            
        training_callbacks = callbacks or []
        
        # 1. Early Stopping
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
        training_callbacks.append(early_stopping)
        
        # 2. Model Checkpoint
        if checkpoint_path:
            chk_path = Path(checkpoint_path)
            chk_path.parent.mkdir(parents=True, exist_ok=True)
            model_checkpoint = ModelCheckpoint(
                filepath=str(chk_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            )
            training_callbacks.append(model_checkpoint)
            
        logger.info(
            "Starting LSTM training (epochs=%d, batch_size=%d, train_samples=%d, val_samples=%d)...",
            epochs, batch_size, len(X_train), len(X_val)
        )
        
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=training_callbacks,
            verbose=1
        )
        
        self.is_trained = True
        self.history_ = {
            k: [float(v) for v in vals]
            for k, vals in history.history.items()
        }
        
        logger.info("LSTM training completed. Best validation loss: %.4f", min(self.history_["val_loss"]))
        return self.history_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict integer class labels (0=Negative, 1=Neutral, 2=Positive)."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probability distributions [P(Neg), P(Neu), P(Pos)]."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before calling predict_proba.")
            
        probs = self.model.predict(X, verbose=0)
        # Numerical normalization check
        row_sums = np.sum(probs, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-4):
            raise ValueError(f"Probabilities do not sum to 1.0 (range: {row_sums.min():.4f} - {row_sums.max():.4f})")
            
        return probs

    def save(self, file_path: Union[str, Path]) -> Path:
        """Save Keras model to disk (.keras format)."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Cannot save an untrained model.")
            
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info("Saved Keras BiLSTM model to %s", path)
        return path

    def load(self, file_path: Union[str, Path]) -> "KerasBiLSTMClassifier":
        """Load Keras model from disk (.keras format)."""
        import tensorflow as tf
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Keras model file not found at {path}")
            
        self.model = tf.keras.models.load_model(str(path))
        self.is_trained = True
        logger.info("Loaded Keras BiLSTM model from %s", path)
        return self

