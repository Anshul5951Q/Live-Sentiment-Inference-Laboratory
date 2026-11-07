# Model Card: Logistic Regression Baseline & Calibrated Classifier

## 1. Model Details
- **Model Name:** TF-IDF + Multinomial Logistic Regression (`LogisticRegressionClassifier`)
- **Version:** 1.0.0 (Milestone 2 & Milestone 5)
- **Model Type:** Discriminative linear classification with multinomial cross-entropy loss.
- **Feature Extraction:** Sublinear TF-IDF vectorizer (10,000 max features, unigram + bigram range, sublinear TF scaling).
- **Calibration Engine:** Multiclass sigmoid calibrator (`MulticlassCalibrator`) using Platt scaling fitted strictly on validation split.
- **License:** Open Source (Apache 2.0 / MIT compatible).
- **Authors:** SentimentLab AI Academic Project Team (P4).

## 2. Intended Use
- **Primary Use:** Baseline discriminative three-class sentiment analysis (`0=Negative`, `1=Neutral`, `2=Positive`) on customer reviews.
- **Out-of-Scope Use:** Generation of natural language text, reasoning, cross-domain sentiment outside Yelp/restaurant domains without calibration.

## 3. Training & Validation Data
- **Training Set:** 19,200 real Yelp reviews (stratified: 6,400 Neg, 6,400 Neu, 6,400 Pos).
- **Validation Set:** 4,800 real Yelp reviews (stratified: 1,600 Neg, 1,600 Neu, 1,600 Pos).
- **Condition B (Augmented):** 19,700 reviews (19,200 real + 500 validated synthetic reviews from Google Gemini).
- **Preprocessing:** HTML tag stripping, URL removal, whitespace normalization, punctuation retention for emotional cues.

## 4. Evaluation Metrics (n=3,000 Real Evaluation Subset)
- **Real Only (Base):**
  - Accuracy: **74.20%**
  - Macro F1: **73.80%**
  - Multiclass Brier Score: **0.3679**
  - Expected Calibration Error (ECE): **0.0380**
  - Inference Latency: **0.11 ms / review**
- **Real + 500 Synthetic (Condition B):**
  - Accuracy: **74.37%** ($\Delta +0.167\%$)
  - Macro F1: **74.03%** ($\Delta +0.228\%$)
  - Neutral Class F1: **61.94%** ($\Delta +0.334\%$)
  - Multiclass Brier Score: **0.3662** (improved calibration)

## 5. Quantitative Limitations & Bias
- **Linearity Constraint:** Linear decision boundaries cannot model complex syntactic negation or word order nuances.
- **Vocabulary Sensitivity:** Relies on n-gram token overlap; cannot handle severe out-of-vocabulary slang.

