# SentimentLab AI: Generative vs Discriminative Sentiment Lab (P4)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io)
[![Milestone 1](https://img.shields.io/badge/Milestone%201-Verified-success.svg)]()
[![Milestone 2](https://img.shields.io/badge/Milestone%202-Complete-success.svg)]()
[![Milestone 3](https://img.shields.io/badge/Milestone%203-Complete-success.svg)]()
[![Milestone 4](https://img.shields.io/badge/Milestone%204-Delivered-success.svg)]()
[![Code Architecture](https://img.shields.io/badge/Architecture-Clean%20Architecture-brightgreen.svg)]()

> **Academic Mini-Project P4:** An end-to-end experimental laboratory comparing discriminative machine learning models (Logistic Regression, Keras BiLSTM, Hugging Face BERT) with generative foundation models (Google Gemini) for three-class sentiment analysis (Negative, Neutral, Positive).

---

## 1. Project Overview & Laboratory Sections

SentimentLab AI is a web-based experimental platform featuring 10 dedicated laboratory sections:

1. **Dashboard:** High-level project metrics, dataset distributions, and system readiness.
2. **Live Sentiment Analyzer:** Real-time multi-model sentiment inference comparing all discriminative and generative predictions side-by-side with confidence scores.
3. **Model Training:** Interactive training, hyperparameter tuning, and probability calibration for Logistic Regression and LSTM with reliability diagrams.
4. **LLM Playground:** In-context zero-shot vs few-shot classification with temperature control and response comparison.
5. **Synthetic Data Lab:** Batch generation of 500 validated synthetic customer reviews (167 Positive, 167 Negative, 166 Neutral) with retry backoff, duplicate detection, and validation reports.
6. **Model Comparison:** Head-to-head empirical benchmarking of Experiment A (Real-only) vs Experiment B (Real + Synthetic) across Accuracy, Precision, Recall, Macro-F1, and Confusion Matrices.
7. **Bias Testing Lab:** Controlled counterfactual sensitivity testing across regional substitutions (Pune, Mumbai, Delhi, Chennai) and demographic references.
8. **GenAI Concepts (Units I & II):** Interactive educational module covering Generative vs Discriminative theory, ML lifecycles, Attention mechanics, and GAN/VAE architectures.
9. **Multimodal Generation (Units III & IV):** Synthesis of spoken sentiment reviews via gTTS and hosted image generation.
10. **Responsible AI (Unit V):** Safety guardrails, prompt injection detection, Dataset Card, and Model Cards.

---

## 2. Technology Stack

- **Runtime & Language:** Python 3.11.9, Windows / Linux
- **Web Interface:** Streamlit, Plotly, Matplotlib, Seaborn
- **Discriminative ML:** Scikit-learn (Logistic Regression, TF-IDF, CalibratedClassifierCV), Joblib
- **Deep Learning:** TensorFlow 2 / Keras (Bidirectional LSTM), PyTorch, Hugging Face Transformers (`bert-base-uncased`), Datasets
- **Generative AI:** Google GenAI SDK (`google-genai` / `google-generativeai`)
- **Multimodal:** gTTS (Text-to-Speech), Pillow, Requests
- **Testing & Quality Assurance:** Pytest, Pytest-Mock, Pydantic v2

---

## 3. Dataset Pipeline & Class Mapping

- **Primary Dataset:** Hugging Face `Yelp/yelp_review_full`
- **Standardized Mapping:**
  - 1–2 Stars $\to$ `0` (**Negative**)
  - 3 Stars $\to$ `1` (**Neutral**)
  - 4–5 Stars $\to$ `2` (**Positive**)
- **Corpus Sizing:**
  - **Balanced Real Subsample:** 24,000 reviews (8,000 per class)
  - **Training Split:** 19,200 reviews (6,400 per class, 80%)
  - **Validation Split:** 4,800 reviews (1,600 per class, 20%)
  - **Untouched Test Set:** Full 50,000 Yelp test split preserved
  - **Fixed Evaluation Benchmark:** 3,000 stratified samples (1,000 per class) with recorded indices (`artifacts/data/eval_subset_metadata.json`) for cross-model comparability under API budget limits.
  - **Zero Leakage:** Strictly verified via automated set-intersection assertions before training.

---

## 4. Academic Provider Substitution Notice

The assignment syllabus references GPT-based synthetic review generation. **Google Gemini** (`gemini-1.5-flash` / `gemini-2.0-flash`) is implemented due to free API tier access.
- In-context generation does **not** fine-tune or update Gemini's weights.
- Only the local discriminative models (Logistic Regression, LSTM, BERT) are retrained in Experiment B.
- A clean provider abstraction (`BaseLLMProvider`) allows swapping providers without refactoring application logic.

---

## 5. Local Setup & Quickstart

### Prerequisites
- Python 3.11 installed (`python --version` or `py -3.11 --version`)
- Git

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd "GenAI P4 Model"
   ```

2. **Create and activate Python 3.11 virtual environment:**
   ```powershell
   # Windows PowerShell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Copy `.env.example` to `.env` and insert your Gemini API Key:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   GEMINI_MODEL_NAME=gemini-1.5-flash
   RANDOM_SEED=42
   ```

5. **Download & preprocess dataset:**
   ```powershell
   python scripts/download_and_preprocess.py
   ```

6. **Train and evaluate Logistic Regression baseline (Milestone 2):**
   ```powershell
   python scripts/train_logistic_regression.py
   ```

7. **Train and evaluate Keras Bidirectional LSTM (Milestone 3):**
   ```powershell
   python scripts/train_lstm.py
   ```

8. **Train Hugging Face BERT locally or via Colab GPU (Milestone 4):**
   ```powershell
   # Local runner (checks device availability / OS policy):
   python scripts/train_bert.py

   # Or open in Google Colab for free NVIDIA T4 GPU fine-tuning:
   # notebooks/01_colab_bert_training.ipynb
   ```

9. **Generate 500 validated synthetic customer reviews (Milestone 4):**
   ```powershell
   python scripts/generate_synthetic_reviews.py
   ```

10. **Analyze synthetic dataset quality & length distributions (Milestone 4):**
   ```powershell
   python scripts/analyze_synthetic_data.py
   ```

11. **Run Five-Way Benchmark Comparison on 300 Identical Reviews (Milestone 4):**
   ```powershell
   python scripts/run_model_comparison.py
   ```

12. **Run automated test suite (79 tests across unit and integration):**
   ```powershell
   pytest -v
   ```

13. **Launch Streamlit Web Application (Available in Milestone 8):**
   ```powershell
   streamlit run ui/app.py
   ```

---

## 6. Multi-Model Benchmark Summary (300 Identical Reviews)

Performance evaluated on the fixed, balanced 300-review subset (100 Negative, 100 Neutral, 100 Positive):

| Model | Architecture / Paradigm | Accuracy | Macro F1 | Multiclass Brier | Avg Latency | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | Linear ML + TF-IDF (10k ngrams) | **74.67%** | **74.27%** | **0.3606** | **0.1 ms** | **Complete** |
| **Bidirectional LSTM** | Deep Recurrent (Embedding + BiLSTM) | **67.67%** | **67.93%** | **0.4219** | **2.4 ms** | **Complete** |
| **Hugging Face BERT** | Contextual Transformer (`bert-base-uncased`) | *Colab Run* | *Colab Run* | *Colab Run* | *Colab Run* | **Notebook Ready** |
| **Gemini Zero-Shot** | In-Context Learning (0 shots) | *Pending Key* | *Pending Key* | *N/A (Scalar)* | *~1,200 ms* | **Mock-Tested** |
| **Gemini Few-Shot** | In-Context Learning (6 shots) | *Pending Key* | *Pending Key* | *N/A (Scalar)* | *~1,400 ms* | **Mock-Tested** |

---

## 7. Project Architecture

```
SentimentLab-AI/
├── config/              # Centralized configuration & settings
├── src/
│   ├── data/            # Loader, preprocessor, stratified splitter
│   ├── models/          # Logistic Regression, BiLSTM, BERT
│   ├── llm/             # Base provider, Gemini integration, Prompts
│   ├── training/        # Experiment A vs B orchestration
│   ├── evaluation/      # Metrics, calibration curves, visualizers
│   ├── responsible_ai/  # Counterfactual bias tester, guardrails
│   ├── multimodal/      # gTTS audio synthesis, image generation
│   └── utils/           # Random seed, atomic file I/O, logging
├── ui/                  # Streamlit multi-page interface (10 pages)
├── scripts/             # Execution & data pipeline scripts
├── notebooks/           # Standalone Colab GPU fine-tuning notebooks
├── artifacts/           # Cached datasets, weights, experiment metrics
├── docs/                # Architecture, methodology, model cards
└── tests/               # Unit and integration pytest suite
```

