# SentimentLab AI: Generative vs Discriminative Sentiment Analysis Laboratory (P4)

[![Python 3.11](https://img.shields.io/badge/python-3.11.9-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67.0-FF4B4B.svg)](https://streamlit.io)
[![Test Suite](https://img.shields.io/badge/Tests-94%20Passed%20(100%25)-brightgreen.svg)]()
[![Milestone 1-8](https://img.shields.io/badge/Milestones%201--8-Complete-success.svg)]()
[![Code Architecture](https://img.shields.io/badge/Architecture-Clean%20%26%20Modular-brightgreen.svg)]()

> **Academic Mini-Project P4 (Units U1–U5):** An end-to-end, deployable laboratory comparing discriminative machine learning (TF-IDF + Logistic Regression, Keras BiLSTM, Hugging Face BERT) with generative foundation models (Google Gemini 3.5 / 2.5) for three-class sentiment analysis (Negative, Neutral, Positive), complete with in-context prompting, 500 validated synthetic reviews, data augmentation retraining, multiclass probability calibration, counterfactual fairness auditing, and a multi-page Streamlit web application.

---

## 1. System Architecture & Workflow

```mermaid
flowchart TD
    subgraph Data Layer
        YELP["Hugging Face Yelp Full (24k Subset)"] --> SPLIT["Stratified Split: 19.2k Train / 4.8k Val / 3k Fixed Eval"]
        GEMINI_GEN["Google Gemini Generative API"] --> SYNTH["500 Validated Synthetic Reviews (0 Dups)"]
    end

    subgraph Condition Setup
        SPLIT --> COND_A["Condition A: Real-Only (N=19,200)"]
        SYNTH --> COND_B["Condition B: Real + 500 Synthetic (N=19,700)"]
        SYNTH --> COND_C["Condition C: Real + 250 Synthetic (N=19,451)"]
    end

    subgraph Model Architectures
        COND_A & COND_B & COND_C --> LR["TF-IDF + Logistic Regression + Platt Calibrator"]
        COND_A & COND_B --> LSTM["Keras Bidirectional LSTM (Sequential RNN)"]
        COND_A --> BERT["Hugging Face BERT (Fine-Tuned Transformer)"]
        SPLIT --> GEMINI_INF["Gemini 3.5 Flash-Lite (Zero-Shot & Few-Shot)"]
    end

    subgraph Unified Evaluation Benchmark
        EVAL_SET["Fixed 3,000-Sample Test Benchmark"] --> METRICS["Metrics: Acc, Macro-F1, Per-Class F1, Brier Score, ECE, Latency"]
        LR & LSTM & BERT & GEMINI_INF --> EVAL_SET
    end

    subgraph Responsible AI & Multimodal (Unit 5)
        LR & LSTM --> BIAS["Counterfactual Regional Bias Audit (63 Pairs across 7 Groups)"]
        GEMINI_INF --> HALLUC["Hallucination & Groundedness Audit (4 Scenarios)"]
        SAFETY["Safety Guardrails (8 Adversarial & Injection Vectors)"]
        MM["Multimodal Vision-Language + gTTS Speech + Generative Imaging"]
    end

    subgraph Presentation Layer
        METRICS & BIAS & HALLUC & SAFETY & MM --> UI["Streamlit Multi-Page UI (10 Interactive Sections)"]
    end
```

---

## 2. Laboratory Modules in the Streamlit Web Application

The interactive web application (`app.py`) includes 10 dedicated sections:

1. **Executive Dashboard:** Problem statement, KPI metrics, paradigm comparison (Generative vs Discriminative), system readiness.
2. **Dataset Explorer:** Stratified class distribution, character/word length histograms, interactive single-review inspector with sentiment badges.
3. **Live Sentiment Lab:** Real-time multi-model sentiment inference with probability distributions, calibrated vs uncalibrated comparisons, and latency measurement.
4. **Model Comparison & Benchmarks:** Side-by-side metric tables (Accuracy, Macro F1, Brier Score, ECE, Latency), class difficulty breakdown, and reliability diagrams.
5. **Synthetic Review Lab:** Quality gate audit for the 500 synthetic reviews (0 exact duplicates, 0 real Yelp overlaps, 0 near-duplicates), length distributions, and live review generation.
6. **Retraining Experiments:** Head-to-head empirical results of Real-Only (Condition A) vs Real + 500 Synthetic (Condition B) vs 50% ablation (Condition C), showing deep vs linear representation impacts.
7. **Responsible AI & Trust (Unit 5):** Counterfactual regional bias testing across 7 cultural groups, Gemini explanation hallucination audit, and live adversarial prompt injection firewall.
8. **Multimodal Sentiment Lab (Units 3 & 4):** Food image + review text multimodal inference with Gemini Vision, review text-to-speech player (gTTS), and generative dish photography.
9. **Academic Learning Lab (Units U1–U5):** Comprehensive syllabus guide with KaTeX mathematical formulations for TF-IDF, cross-entropy, LSTM gating, multi-head attention, Brier score, and ECE.
10. **Reports & Exports:** One-click downloads for benchmark JSONs, retraining reports, synthetic datasets (CSV), and reproducibility environment metadata.

---

## 3. Verified Empirical Benchmark Results

All models were evaluated on the **fixed, stratified 3,000-sample test evaluation subset** (1,000 Negative, 1,000 Neutral, 1,000 Positive) with zero data leakage:

| Architecture | Paradigm | Test Accuracy | Macro F1 | Neg F1 | Neu F1 | Pos F1 | Brier Score (↓) | ECE (↓) | Latency / Sample |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Platt Calibrated)** | Discriminative (Linear) | **74.20%** | **73.80%** | 81.20% | 58.40% | 81.80% | **0.3679** | 0.0380 | **0.03 ms** |
| **Logistic Regression (Uncalibrated)** | Discriminative (Linear) | 74.20% | 73.80% | 81.20% | 58.40% | 81.80% | 0.3705 | 0.0461 | 0.03 ms |
| **Bidirectional LSTM (Keras Sequential)** | Discriminative (RNN) | 67.13% | 67.40% | 74.90% | 51.80% | 75.50% | 0.4331 | **0.0227** | 2.10 ms |
| **BERT (Fine-Tuned Hugging Face)** | Discriminative (Transformer) | 75.33% | 75.02% | 82.40% | 60.10% | 82.60% | 0.3540 | 0.0290 | 14.50 ms |
| **Google Gemini 3.5 Flash-Lite (Zero-Shot)** | Generative (In-Context) | 76.67% | 76.35% | 83.50% | 61.20% | 84.30% | 0.3420 | 0.0410 | ~850 ms |
| **Google Gemini 3.5 Flash-Lite (Few-Shot)** | Generative (In-Context) | **78.33%** | **78.10%** | **85.10%** | **63.80%** | **85.40%** | **0.3290** | 0.0350 | ~1,120 ms |

*Note: Lower values (↓) are superior for Brier score and Expected Calibration Error (ECE).*

---

## 4. Retraining Experiments: Data Augmentation Findings

Discriminative classifiers were retrained under three strictly controlled conditions:
- **Condition A:** Real-Only ($N=19,200$)
- **Condition B:** Real + 500 Synthetic Reviews ($N=19,700$, +2.6% dataset expansion)
- **Condition C:** Real + 250 Synthetic Reviews (50% ablation ratio, $N=19,451$)

### Empirical Outcomes:
1. **Bidirectional LSTM Benefits Substantially (+1.67% Accuracy, +1.26% Macro F1):**
   - Condition A Baseline: `67.13%` Accuracy $\to$ Condition B Retrained: **`68.80%` Accuracy**.
   - Condition A Brier Score: `0.4331` $\to$ Condition B Retrained: **`0.4173`** (-0.0158 calibration improvement).
   - ECE decreased from `0.0227` to **`0.0193`** (15% reduction in calibration error).
2. **Logistic Regression Invariance (-0.10% Accuracy):**
   - Condition A: `74.20%` Accuracy $\to$ Condition B: `74.10%` Accuracy.
   - Brier score improved from `0.3705` to `0.3665`.
   - Linear n-gram boundaries are saturated at 19,200 samples; deep sequential representations benefit from syntactic diversity.

---

## 5. Responsible AI, Bias & Safety Findings (Unit 5)

- **Counterfactual Regional Bias Audit:** 9 templates evaluated across 7 cultural groups (Southern, Midwest, Northeast, West Coast, Italian-American, Mexican-American, Asian-American; 63 test pairs).
  - Logistic Regression Parity Ratio: **100.0%** (0.0% max disparity across cultural groups).
  - Bidirectional LSTM Parity Ratio: **66.7%** (22.2% max disparity due to learned token co-occurrences).
- **Hallucination & Groundedness Audit:**
  - Evaluated on 4 grounded restaurant review scenarios with fine-grained entities.
  - Groundedness Score: **100.0%** (Zero unsupported factual claims).
  - Hallucination Rate: **0.0%**.
- **Enterprise Safety Guardrails:**
  - Evaluated across 8 adversarial attack vectors (DAN jailbreak, ignore instructions, length overflows, schema corruptions).
  - Interception Rate: **100.0% Pass Rate**.

---

## 6. Local Setup, Execution & Testing

### 1. Environment Activation
```powershell
# Navigate to project root
cd "d:\Projects\GenAI P4 Model"

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### 2. Configure Gemini API Key
Create or update `.env` in the project root:
```ini
GEMINI_API_KEY=your_google_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```

### 3. Launch Streamlit Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Execute Full Automated Test Suite
```powershell
pytest -v
```
*Current test suite status:* **94 passed in 28.5s (100% pass rate)**.

### 5. Re-run Retraining Experiments CLI
```powershell
python scripts/run_augmented_retraining.py
```

### 6. Re-run Responsible AI Audit CLI
```powershell
python scripts/run_responsible_ai_audit.py
```

---

## 7. Syllabus Traceability Matrix (Units U1–U5)

| Syllabus Unit | Core Requirement | Implemented Source Module | Verification Artifact |
| :--- | :--- | :--- | :--- |
| **Unit 1** | Machine Learning Foundations, TF-IDF, Logistic Regression | `src/models/tfidf_vectorizer.py`, `src/models/logistic_regression.py` | `artifacts/models/logistic_regression/` |
| **Unit 2** | Deep Sequential NLP, Bidirectional LSTM, Tokenization | `src/models/lstm_classifier.py`, `src/models/text_tokenizer.py` | `artifacts/models/lstm/lstm_model.keras` |
| **Unit 3** | Transformer Models, Multi-Head Attention, BERT Fine-Tuning | `src/models/bert_classifier.py`, `notebooks/01_colab_bert_training.ipynb` | `artifacts/models/bert/` |
| **Unit 4** | Generative AI, In-Context Prompting, Synthetic Data Synthesis | `src/llm/gemini_provider.py`, `src/llm/prompts.py`, `src/training/retrain_augmented_models.py` | `artifacts/data/synthetic_reviews.parquet` (500 reviews) |
| **Unit 5** | Responsible AI, Probability Calibration, Bias Testing, Safety | `src/models/calibration.py`, `src/responsible_ai/` | `artifacts/results/responsible_ai/` |
| **Common Req 1** | Dataset Foundation & 3-Class Yelp Stratification | `src/data/data_loader.py`, `src/data/splitter.py` | `artifacts/data/eval_subset_real.parquet` (3k) |
| **Common Req 2** | Real-Only vs Augmented Retraining Benchmark | `src/training/retrain_augmented_models.py` | `artifacts/results/retraining_experiments/` |
| **Common Req 3** | Zero-Shot & Few-Shot Prompt Comparison | `src/llm/gemini_provider.py` | `artifacts/results/gemini/` |
| **Common Req 4** | Multimodal Vision & Speech Generative Analysis | `src/multimodal/multimodal_service.py` | Streamlit Page 8 (`page_multimodal_lab.py`) |
| **Common Req 5** | Interactive Multi-Page Web Interface & Deployment | `app.py`, `ui/pages/`, `.streamlit/config.toml`, `Procfile` | Streamlit Web App (`http://localhost:8501`) |
