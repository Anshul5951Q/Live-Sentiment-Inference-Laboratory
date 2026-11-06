# SentimentLab AI: System Architecture

**Project:** Academic Mini-Project P4: Generative vs Discriminative Sentiment Analysis Laboratory  
**Document:** System Architecture & Technical Specification  
**Version:** 1.1 (Post-Review Refinements)  

---

## 1. System Overview & Clean Architecture Principles

SentimentLab AI is an experimental benchmarking platform comparing discriminative machine learning models (Logistic Regression, LSTM, BERT) against generative large language models (Google Gemini) for three-class sentiment analysis (Negative, Neutral, Positive).

The system follows Clean Architecture and Domain-Driven Design (DDD):
- **Core Domain Entities:** Standardized review records (`review`, `sentiment`, `label`, `source`).
- **Use Cases / Experiment Services:** Experiment runners for real-only training (Condition A) vs real + synthetic augmented training (Condition B).
- **Interface Adapters / Providers:** Abstracted model interfaces (`BaseSentimentModel`) and LLM interfaces (`BaseLLMProvider`).
- **Frameworks & UI:** Streamlit web application, PyTorch/TensorFlow runtimes, Hugging Face datasets.

```
       +-------------------------------------------------------+
       |                  Streamlit UI Layer                   |
       |  (Dashboard, Live Analyzer, Labs, Bias Audit, GenAI)  |
       +-------------------------------------------------------+
                                  |
                                  v
       +-------------------------------------------------------+
       |               Application Services Layer              |
       |  (ExperimentRunner, SyntheticGenerator, BiasTester)  |
       +-------------------------------------------------------+
                |                                    |
                v                                    v
       +-----------------------+          +--------------------+
       |  Discriminative Core  |          | Generative / LLM   |
       |  - LogisticRegression |          | - BaseLLMProvider  |
       |  - Keras BiLSTM       |          | - GeminiProvider   |
       |  - Hugging Face BERT  |          | - MockLLMProvider  |
       +-----------------------+          +--------------------+
                |                                    |
                +-------------------+----------------+
                                    |
                                    v
       +-------------------------------------------------------+
       |                 Data & Artifacts Layer                |
       |  (YelpDataLoader, Preprocessor, Splitter, Caches)     |
       +-------------------------------------------------------+
```

---

## 2. Refined Technical Decisions

### 2.1 Dataset & Evaluation Strategy
- **Dataset:** Hugging Face `yelp_review_full` (650,000 train, 50,000 test).
- **Class Mapping:**
  - 1–2 stars (or 0–1 in 0-indexed Yelp) $\to$ `0` (**Negative**)
  - 3 stars (or 2 in 0-indexed Yelp) $\to$ `1` (**Neutral**)
  - 4–5 stars (or 3–4 in 0-indexed Yelp) $\to$ `2` (**Positive**)
- **Experimental Corpus:**
  - **Balanced Real Subsample:** 24,000 samples (8,000 per class).
  - **Train Split:** 19,200 samples (6,400 per class, 80%).
  - **Validation Split:** 4,800 samples (1,600 per class, 20%).
- **Evaluation Strategy & Cost Control:**
  - Full untouched 50,000-sample test set preserved for offline/local model benchmarking.
  - **Fixed 3,000-Sample Evaluation Subset:** 1,000 Negative, 1,000 Neutral, 1,000 Positive.
  - Selected using deterministic seed (`seed=42`) with original sample indices persisted to `artifacts/data/eval_subset_metadata.json`.
  - Used uniformly across all models (including Gemini API evaluation) to prevent budget exhaustion while ensuring direct, statistically sound comparisons.
  - **Leakage Prevention:** Split stratification occurs before feature extraction; zero overlap is strictly enforced and verified via test assertions.

### 2.2 Logistic Regression & Multiclass Calibration (Implemented & Verified in Milestone 2)
- **Feature Extraction:** `TextTfidfVectorizer` (`src/models/tfidf_vectorizer.py`) implementing scikit-learn `TfidfVectorizer` with `max_features=10000`, `ngram_range=(1, 2)`, `sublinear_tf=True`, `min_df=2`, `max_df=0.95`. Strict leakage barrier: fitted exclusively on `train_real.parquet`.
- **Base Classifier:** `LogisticRegressionClassifier` (`src/models/logistic_regression.py`) implementing multi-class `LogisticRegression` with cross-entropy loss and L-BFGS optimizer ($C=1.0$, `max_iter=1000`).
- **Calibration Engine:** `MulticlassCalibrator` (`src/models/calibration.py`) wrapping the base model with `CalibratedClassifierCV` via `FrozenEstimator` (scikit-learn 1.4+ API compatible with fallback to `cv="prefit"`). Fitted strictly on `val_real.parquet` (4,800 samples).
- **Inference Service:** `SentimentPredictor` (`src/models/predictor.py`) offering single and batch review classification, confidence computation, and returning both uncalibrated and calibrated probability distributions.
- **Evaluation Engine:** `classification_metrics.py` computing accuracy, macro/weighted F1, confusion matrix, multiclass Brier score, log loss, ECE, confidence distribution, and reliability diagram coordinates.
- **Visualizer:** `visualizer.py` generating publication-quality confusion matrices, reliability curves, confidence distribution histograms, and metric comparisons saved to `artifacts/results/logistic_regression/`.

### 2.3 Keras Bidirectional LSTM Architecture (Implemented & Verified in Milestone 3)
- **Input Pipeline:** Custom integer-sequence `TextTokenizer` (`src/models/text_tokenizer.py`) fitted strictly on `train_real.parquet` (19,200 samples) with `vocab_size=20000`, sequence padding/truncation `max_length=128`, `<PAD>=0`, and `<OOV>=1`. Tokenizer mapping is serialized to `artifacts/models/lstm/tokenizer.json`.
- **Model Architecture:** `KerasBiLSTMClassifier` (`src/models/lstm_classifier.py`):
  1. `Input(shape=(128,), dtype=tf.int32, name="input_ids")`
  2. `Embedding(input_dim=20000, output_dim=128, mask_zero=True, name="embedding")`
  3. `Bidirectional(LSTM(units=64, dropout=0.3, recurrent_dropout=0.0, return_sequences=False), name="bidirectional_lstm")` *(recurrent_dropout=0.0 preserves cuDNN hardware acceleration compatibility)*
  4. `Dropout(rate=0.3, name="dropout_1")`
  5. `Dense(units=64, activation="relu", name="dense_features")`
  6. `Dropout(rate=0.3, name="dropout_2")`
  7. `Dense(units=3, activation="softmax", name="sentiment_output")`
- **Training Safeguards & Callbacks:**
  - `EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)`
  - `ModelCheckpoint(filepath="artifacts/models/lstm/lstm_model.keras", monitor="val_loss", save_best_only=True)`
  - Loss function: `sparse_categorical_crossentropy`, Optimizer: `Adam(learning_rate=0.001)`.
- **Artifacts Generated:** `lstm_model.keras` (32MB), `tokenizer.json`, `model_metadata.json`, `loss_curve.png`, `accuracy_curve.png`, `confusion_matrix.png`, `confidence_distribution.png`, `evaluation_metrics.json`, `validation_metrics.json`, and `predictions.parquet`.

### 2.4 Hugging Face BERT (`bert-base-uncased`)
- **Architecture:** Transformer sequence classification with 12 attention heads and 110M parameters.
- **Local vs Colab Execution:** Standalone Google Colab notebook (`notebooks/01_colab_bert_training.ipynb`) provided for free GPU fine-tuning. Local RTX 4050 (6GB VRAM) can run local fine-tuning using `fp16` mixed precision (batch size 16).
- **Deployment Safety:** In Streamlit Cloud, fine-tuning is disabled to prevent Out-Of-Memory (OOM) crashes; pre-trained weights and pre-computed evaluation matrices are loaded on demand.

### 2.5 Gemini Free-Tier Resilience & LLM Integration (Implemented in Milestone 3)
- **SDK:** Official Google Gen AI Python SDK (`google-genai`, version >= 3.0.0 / 0.1.1) utilizing HTTP REST (`httpx`) transport to prevent Windows DLL code-integrity conflicts with legacy gRPC C-extensions.
- **Configurability:** Model name configurable via `GEMINI_MODEL_NAME` (e.g. `gemini-1.5-flash`, `gemini-2.0-flash`).
- **Prompt Engineering & Structured JSON Output:** Strict JSON output schema (`sentiment`, `label`, `confidence`, `reason`) with system instructions forbidding markdown prose and enforcing sentiment polarity boundaries.
- **Free-Tier Rate-Limiting & Resilience:**
  - Exponential backoff with random jitter on HTTP 429 (`ResourceExhausted`) and 503 (`Unavailable`).
  - Configurable query delay (`rate_limit_delay=4.0s`) ensuring API calls do not breach the 15 requests per minute (RPM) free-tier ceiling.
  - Persistent JSON disk caching (`response_cache.json`) indexed by `hash(prompt)` to prevent redundant API queries.
  - State checkpointing (`checkpoint_{mode}.json`) allowing resumption from the exact failed review upon transient disconnections.
- **Unified Prediction Data Contract:** `ModelPredictionResult` (`src/models/common_interface.py`) providing a standardized output schema across Logistic Regression, BiLSTM, and Gemini (zero-shot and few-shot).
- **Evaluation Strategy on 300-Sample Benchmark:** Stratified, reproducible 300-sample evaluation benchmark (`artifacts/data/gemini_benchmark_300.parquet`: 100 Negative, 100 Neutral, 100 Positive) extracted from the fixed 3,000 evaluation subset for fair cross-model comparisons under quota constraints.
- **Probability vs Scalar Confidence:** Discriminative models output calibrated posterior distributions $P(Y=c|X)$ where $\sum_c P(Y=c|X) = 1.0$. Generative LLMs output self-reported certainty scalars elicited via structured prompting. The evaluation engine cleanly segregates these concepts, avoiding uncalibrated probability assumptions on LLMs.
- **Verification Gate:** Exactly 500 unique, validated reviews must be recorded before completion is declared.
- **Mock Provider:** `MockLLMProvider` used in automated tests; no real API quota consumed during testing.
- **Academic Substitution Transparency:** Clearly documented as a provider substitution for GPT due to free developer tier access. Synthetic review augmentation does not modify Gemini model weights.

### 2.6 Counterfactual Bias Testing
- **Templates:** 50+ identical templates across 4 major Indian metropolitan regions (Pune, Mumbai, Delhi, Chennai), demographic indicators, and neutral control entities.
- **Metrics:**
  - Prediction Consistency ($\%$ of template variations yielding identical sentiment).
  - Sentiment Probability Delta ($\Delta P(\text{Positive}), \Delta P(\text{Negative})$ across regional substitutions).
  - Class-wise error disparities.
- **Ethical Limitation Statement:** Documented that synthetic counterfactual tests cannot certify an AI system as universally unbiased.

### 2.7 Deployment Strategy & Lazy Loading
- **Memory Management on Streamlit Cloud:**
  - Lightweight Logistic Regression loaded on startup.
  - Deep learning models (LSTM, BERT) loaded into memory **lazily on demand** when accessed by the user, and unloaded if necessary.
  - Gemini calls executed only on explicit user interaction.
- **Multimodal Services:**
  - gTTS audio synthesis runs locally via CPU.
  - Hosted image generation runs via API with clear visual indicators if fallback mock mode is active.

---

## 3. Directory Structure

```
SentimentLab-AI/
├── config/                  # Settings and logging configuration
├── src/
│   ├── data/                # Data loader, preprocessor, splitter
│   ├── models/              # Logistic Regression, LSTM, BERT
│   ├── llm/                 # Provider interface, Gemini, Prompts, Generator
│   ├── training/            # Experiment A & B orchestrators
│   ├── evaluation/          # Metrics, calibration, visualizers
│   ├── responsible_ai/      # Bias tester, guardrails, model cards
│   ├── multimodal/          # Audio TTS, Image generation
│   └── utils/               # Seed, atomic file I/O, logging
├── ui/                      # Streamlit 10-page application
├── scripts/                 # CLI entrypoints
├── notebooks/               # Colab GPU notebook & EDA
├── artifacts/               # Caches, models, datasets, evaluation results
├── tests/                   # Pytest unit and integration suite
└── docs/                    # Academic documentation
```

