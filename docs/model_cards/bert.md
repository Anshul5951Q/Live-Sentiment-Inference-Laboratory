# Model Card: Hugging Face BERT (`bert-base-uncased`)

## 1. Model Details
- **Model Name:** Hugging Face BERT for Sequence Classification (`BertSentimentClassifier`)
- **Base Pretrained Weights:** `bert-base-uncased` (Google Research, Devlin et al., 2018)
- **Version:** 1.0.0 (Milestone 4 & Milestone 4B)
- **Model Type:** Bidirectional Transformer Encoder with contextual self-attention.
- **Architecture:** 12 Transformer layers, 768 hidden dimensions, 12 attention heads, 110M parameters.
- **Classification Head:** Linear layer on top of pooled `[CLS]` embedding with $0.1$ dropout projecting to $\mathbb{R}^3$.
- **Tokenizer:** WordPiece tokenizer (frozen vocabulary of 30,522 subword tokens).
- **Framework:** PyTorch & Hugging Face Transformers.
- **License:** Apache 2.0.

## 2. Intended Use
- **Primary Use:** State-of-the-art discriminative sequence classification leveraging deep bidirectional self-attention to disambiguate subtle contextual sentiment, negations, and idioms.
- **Out-of-Scope Use:** Free-form review generation, zero-shot entity generation.

## 3. Training & Validation Strategy
- **Training Set:** 19,200 real Yelp customer reviews.
- **Validation Set:** 4,800 real Yelp customer reviews (monitored for early stopping).
- **Evaluation Set:** 3,000 isolated evaluation samples (and 300 benchmark subset).
- **Optimization:** AdamW ($lr=2 \times 10^{-5}$, weight decay $0.01$), linear warmup schedule (10% of total steps), batch size 16, max sequence length 128.
- **Hardware Profile:** NVIDIA T4 GPU runtime (Google Colab notebook `notebooks/01_colab_bert_training.ipynb`) with automated ingestion via `python scripts/train_bert.py --ingest bert_artifacts.zip`.

## 4. Strengths & Limitations
- **Contextual Superiority:** Handles polysemy, syntactic negation (e.g. "not bad at all" vs "not good"), and contrastive clauses ("great food but horrible service").
- **Compute Intensity:** Requires GPU acceleration for efficient fine-tuning; inference latency is higher than linear and shallow recurrent baselines (~15–25 ms / sample).

