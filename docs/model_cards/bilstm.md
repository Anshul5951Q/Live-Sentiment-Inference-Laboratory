# Model Card: Keras Bidirectional LSTM Sentiment Classifier

## 1. Model Details
- **Model Name:** Keras Bidirectional LSTM (`KerasBiLSTMClassifier`)
- **Version:** 1.0.0 (Milestone 3 & Milestone 5)
- **Model Type:** Deep recurrent neural network for sequence classification.
- **Architecture:**
  - Embedding Layer: $d_{\text{embed}} = 128$, vocabulary size $20,000$.
  - SpatialDropout1D: rate $0.2$.
  - Bidirectional LSTM: $64$ hidden units in each direction ($128$ concatenated).
  - Dense Hidden Layer: $64$ units with ReLU activation and $0.3$ Dropout.
  - Output Layer: $3$ units with Softmax activation.
- **Loss & Optimizer:** Sparse Categorical Cross-Entropy, Adam optimizer ($lr=10^{-3}$).
- **Framework:** TensorFlow 2 / Keras.
- **License:** Open Source (Apache 2.0).

## 2. Intended Use
- **Primary Use:** Recurrent sequential sentiment classification capturing word order, bidirectional context, and long-range syntactic dependencies.
- **Out-of-Scope Use:** Unbounded sequence lengths beyond 512 tokens (configured max length is 128), generative language modeling.

## 3. Training & Validation Data
- **Training Set (Base):** 19,200 real Yelp reviews.
- **Validation Set:** 4,800 real Yelp reviews with EarlyStopping (patience 3, restoring best validation loss).
- **Condition B (Augmented):** 19,700 reviews (19,200 real + 500 validated synthetic reviews).
- **Vocabulary:** 20,000 top unigrams with `<PAD>` at index 0 and `<OOV>` at index 1.

## 4. Evaluation Metrics (n=3,000 Real Evaluation Subset)
- **Real Only (Base):**
  - Accuracy: **67.13%**
  - Macro F1: **67.40%**
  - Multiclass Brier Score: **0.4331**
  - Expected Calibration Error (ECE): **0.0468**
  - Latency: **1.62 ms / review**
- **Real + 500 Synthetic (Condition B):**
  - Accuracy: **67.63%** ($\Delta +0.500\%$)
  - Macro F1: **67.91%** ($\Delta +0.506\%$)
  - Neutral Class F1: **62.88%** ($\Delta +0.812\%$)
  - Multiclass Brier Score: **0.4285** (improved confidence calibration)

## 5. Limitations & Bias
- **Compute Bound:** Recurrent steps cannot be parallelized across time steps as efficiently as self-attention in Transformers.
- **Context Truncation:** Reviews exceeding 128 tokens are truncated at the head.

