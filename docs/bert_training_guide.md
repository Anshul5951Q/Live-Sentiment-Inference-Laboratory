# SentimentLab AI: Hugging Face BERT Training & Deployment Guide

**Project:** Academic Mini-Project P4: Generative vs Discriminative Sentiment Analysis Laboratory  
**Document:** BERT Fine-Tuning Specification & Google Colab GPU Guide  
**Version:** 1.0  

---

## 1. BERT Architecture & Technical Overview

In SentimentLab AI, BERT (**Bidirectional Encoder Representations from Transformers**) represents the contextualized deep learning branch of our discriminative model comparison:

- **Base Checkpoint:** `bert-base-uncased` (110M parameters, 12 attention heads, 768 hidden dimension).
- **Tokenizer:** Pretrained WordPiece tokenizer (`AutoTokenizer.from_pretrained('bert-base-uncased')`). Pretrained vocabulary is strictly frozen and never modified.
- **Classification Head:** 
  $$\mathbf{h}_{[\text{CLS}]} \in \mathbb{R}^{768} \xrightarrow{\text{Dropout}(0.1)} \mathbf{W} \in \mathbb{R}^{3 \times 768} + \mathbf{b} \to \text{Softmax}(\mathbf{z}) \in \Delta^2$$
- **Target Classes:**
  - `0`: Negative (1–2 Yelp stars)
  - `1`: Neutral (3 Yelp stars)
  - `2`: Positive (4–5 Yelp stars)

---

## 2. Training Hardware Audit & Execution Policy

### 2.1 Hardware Discovery & Security Policy Isolation
On the local host system:
- **Physical GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) is physically installed.
- **OS Security Policy:** Windows Defender Application Control (WDAC) / AppLocker actively restricts loading unsigned C-extension DLLs (such as `torch_global_deps.dll`) inside non-sanctioned user directories (`[WinError 4551]`).
- **Ethical & Safety Boundary:** In adherence to core security guidelines, access control policies are never bypassed or spoofed.

### 2.2 Google Colab GPU Execution Protocol
To provide 100% reproducible GPU fine-tuning without security friction:
1. Open the standalone notebook: [`notebooks/01_colab_bert_training.ipynb`](file:///d:/Projects/GenAI%20P4%20Model/notebooks/01_colab_bert_training.ipynb).
2. Set runtime accelerator to **NVIDIA T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).
3. Run all cells to fine-tune `bert-base-uncased` across the 19,200 training Yelp reviews with validation early stopping.
4. Download the resulting `bert_artifacts.zip`.
5. Extract the exported checkpoint files into:
   - `artifacts/models/bert/` (`model.safetensors`, `config.json`, `tokenizer.json`, etc.)
   - `artifacts/results/bert/` (`evaluation_metrics.json`, `training_history.json`, plots).

---

## 3. Training Hyperparameters

- **Batch Size:** 16
- **Max Sequence Length:** 128 (or 256 for extended context)
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$)
- **Learning Rate:** $2 \times 10^{-5}$
- **Weight Decay:** 0.01
- **Learning Rate Schedule:** Linear decay with 10% warmup steps
- **Gradient Clipping:** Max norm 1.0
- **Epochs:** 3 with Early Stopping on validation loss / validation macro-F1

---

## 4. Ingestion & Local Predictor Usage

Once artifacts are placed in `artifacts/models/bert/`, load the inference predictor in Python:

```python
from src.models.bert_classifier import BertSentimentPredictor

predictor = BertSentimentPredictor.from_artifacts("artifacts/models/bert")
result = predictor.predict_one("The food was incredible but service was slightly slow.")
print(result.predicted_sentiment, result.confidence)
```
