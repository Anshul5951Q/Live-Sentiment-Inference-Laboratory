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

### 2.2 Logistic Regression & Multiclass Calibration
- **Feature Extraction:** TF-IDF vectorizer (unigrams + bigrams, sublinear TF scaling).
- **Base Classifier:** Scikit-learn `LogisticRegression` (multinomial / cross-entropy loss).
- **Calibration Engine:** `CalibratedClassifierCV` using sigmoid or isotonic methods fitted strictly on training/validation folds.
- **Evaluation:** Reliability diagrams (calibration curves) before and after calibration, همراه with multiclass Brier score and Expected Calibration Error (ECE). The test set is never used for fitting calibration.

### 2.3 Keras Bidirectional LSTM Architecture
- **Input Pipeline:** Text tokenization fit strictly on training data; out-of-vocabulary (`<OOV>`) token handling; configurable sequence padding (`max_length=128`).
- **Layers:**
  1. `Embedding(input_dim=20000, output_dim=128, mask_zero=True)`
  2. `Bidirectional(LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))`
  3. `Dense(64, activation='relu')`
  4. `Dropout(0.3)`
  5. `Dense(3, activation='softmax')`
- **Training Safeguards:** `EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)` and checkpointing of the best model to `artifacts/models/lstm_model.keras`.

### 2.4 Hugging Face BERT (`bert-base-uncased`)
- **Architecture:** Transformer sequence classification with 12 attention heads and 110M parameters.
- **Local vs Colab Execution:** Standalone Google Colab notebook (`notebooks/01_colab_bert_training.ipynb`) provided for free GPU fine-tuning. Local RTX 4050 (6GB VRAM) can run local fine-tuning using `fp16` mixed precision (batch size 16).
- **Deployment Safety:** In Streamlit Cloud, fine-tuning is disabled to prevent Out-Of-Memory (OOM) crashes; pre-trained weights and pre-computed evaluation matrices are loaded on demand.

### 2.5 Gemini Free-Tier Resilience & Provider Abstraction
- **SDK:** Official `google-genai` SDK with fallback to `google-generativeai`.
- **Configurability:** Model name configurable via `GEMINI_MODEL_NAME` (e.g. `gemini-1.5-flash`, `gemini-2.0-flash`).
- **Adaptive Resilience:**
  - Dynamic rate-limit handling with exponential backoff and jitter.
  - Inspection of `Retry-After` HTTP headers on HTTP 429 (`ResourceExhausted`).
  - Max retry limits to prevent infinite loops.
- **Controlled Generation Batches:** Reviews generated in batches of 10–20 to minimize API calls while fitting within free-tier token ceilings.
- **State Checkpointing & Resumption:** Successful batches saved immediately to disk; interrupted runs resume from the last valid review count.
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
