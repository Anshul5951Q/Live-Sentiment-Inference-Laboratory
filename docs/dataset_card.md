# Dataset Card: Yelp Review Full (SentimentLab AI Subset)

## 1. Dataset Summary
- **Source:** Hugging Face `Yelp/yelp_review_full`
- **Original Source:** Yelp Open Dataset & Yelp Academic Dataset
- **Task:** 3-Class Sentiment Classification (Negative, Neutral, Positive)
- **License:** Yelp Dataset License & Terms of Use (for academic and educational use only; raw corpus not redistributed)

---

## 2. Dataset Structure & Fields
Standardized schema across all experiment splits:
- `review` (*string*): Cleaned review text with HTML tags stripped and whitespace normalized.
- `label` (*int64*): Sentiment integer class index:
  - `0`: Negative (mapped from 1 and 2 stars)
  - `1`: Neutral (mapped from 3 stars)
  - `2`: Positive (mapped from 4 and 5 stars)
- `sentiment` (*string*): String representation corresponding to label (`"Negative"`, `"Neutral"`, `"Positive"`).
- `source` (*string*): Origin tag (`"real"` for Yelp dataset; `"synthetic"` for Gemini-generated reviews).

---

## 3. Data Splits & Sample Counts

| Split Name | File Name | Total Samples | Negative (0) | Neutral (1) | Positive (2) | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Real Train** | `train_real.parquet` | 19,200 | 6,400 (33.3%) | 6,400 (33.3%) | 6,400 (33.3%) | Training discriminative models (Condition A & B) |
| **Real Validation** | `val_real.parquet` | 4,800 | 1,600 (33.3%) | 1,600 (33.3%) | 1,600 (33.3%) | Early stopping, hyperparameter tuning & calibration |
| **Real Test (Full)** | `test_real.parquet` | 50,000 | ~20,000 | ~10,000 | ~20,000 | Offline untouched test benchmark |
| **Eval Subset (Fixed)**| `eval_subset_real.parquet`| 3,000 | 1,000 (33.3%) | 1,000 (33.3%) | 1,000 (33.3%) | Standardized evaluation set for all models (including Gemini API) |
| **Synthetic 500** | `synthetic_500.csv` | 500 | 167 (33.4%) | 166 (33.2%) | 167 (33.4%) | Training data augmentation (Condition B only) |

---

## 4. Leakage Prevention & Reproducibility
- **Deterministic Sampling:** All sampling and partitioning are seeded with `seed=42`.
- **Index Tracking:** The exact sample indices of the 3,000 evaluation subset are recorded in `artifacts/data/eval_subset_metadata.json`.
- **Zero Leakage:** Set-intersection checks between review texts in train, validation, and test sets are validated during dataset generation and enforced in the test suite (`test_splitter.py`).
