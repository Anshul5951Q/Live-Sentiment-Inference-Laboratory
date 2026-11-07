# SentimentLab AI: 3-Minute Video Demonstration Script

**Project:** SentimentLab AI – Generative vs Discriminative Sentiment Analysis Laboratory (P4)  
**Target Duration:** 3 Minutes (180 Seconds)  
**Presenter:** Student / Lead ML Engineer  
**Live Application URL:** `http://localhost:8501` (`streamlit run app.py`)  

---

## Slide & Screen Progression Breakdown

| Timestamp | Visual Screen / Streamlit Section | Presenter Narration (Word-for-Word Script) | Key Actions On-Screen |
| :--- | :--- | :--- | :--- |
| **0:00 - 0:30** | **Page 1: Executive Dashboard** | *"Hello everyone. Welcome to SentimentLab AI, our academic mini-project comparing discriminative machine learning with generative foundation models for three-class sentiment analysis across Units 1 to 5. Here on the Executive Dashboard, we track our 24,000 real Yelp review corpus, our fixed zero-leakage 3,000-sample test benchmark, and our 500 validated synthetic reviews generated using Google Gemini. We contrast discriminative models that directly learn posterior conditional boundaries $P(Y \mid X)$ with generative models that perform in-context reasoning."* | Hover over the 4 KPI metric cards, scroll past the paradigm comparison diagram. |
| **0:30 - 1:00** | **Page 3: Live Sentiment Lab** | *"In the Live Sentiment Lab, we can run real-time inference across all 5 architectures. Let's test a nuanced review: 'The pizza was acceptable, but twenty dollars for tap water and cold pasta is excessive.' When we run Logistic Regression with Platt calibration, we get an ultra-fast prediction in under 0.1 milliseconds with calibrated probabilities. In contrast, running Google Gemini 3.5 Flash-Lite produces rich in-context reasoning and extracts grounded entity rationales."* | Click the 'Ambiguous' test preset. Run inference with Calibrated Logistic Regression, then switch to Gemini 3.5. Show posterior probability bars and latency. |
| **1:00 - 1:30** | **Page 4: Model Benchmarks & Page 5: Synthetic Lab** | *"On Page 4, we evaluate all models head-to-head on the fixed 3,000-sample test set. While few-shot Gemini achieves the highest raw accuracy at 78.33%, fine-tuned BERT achieves 75.33%, and calibrated Logistic Regression delivers an astonishing 74.20% accuracy with 0.03 millisecond latency—making it 30,000 times faster than the LLM. On Page 5, we audit our 500 Gemini-synthesized reviews. Notice that our automated quality gates achieved exactly zero internal duplicates and zero overlaps with real Yelp reviews."* | Show the unified comparison table. Point out the neutral-class bottleneck (58% F1 vs 82% on polar classes). Switch to Page 5 and highlight the zero duplication audit metrics. |
| **1:30 - 2:15** | **Page 6: Retraining Experiments** | *"Now to our core research experiment in Milestone 5: Does synthetic data augmentation actually improve discriminative models? We retrained models under Condition A (Real-Only) versus Condition B (Real + 500 Synthetic). Look at this key empirical result: For linear TF-IDF Logistic Regression, the accuracy remained virtually invariant (-0.10%). But for the deep Bidirectional LSTM, synthetic data augmentation produced a massive +1.67% accuracy surge, boosted Macro F1 by +1.26%, and lowered the multiclass Brier score to 0.4173. The dense recurrent network capitalized on the syntactic fluency of synthetic reviews to regularize its decision boundaries."* | Display the Delta Metric Cards (+1.67% BiLSTM Acc). Show `real_vs_augmented_accuracy_f1.png` and explain the difference between linear subspace vs deep representation capacity. |
| **2:15 - 2:45** | **Page 7: Responsible AI & Page 8: Multimodal** | *"In Unit 5 Responsible AI, we executed counterfactual regional bias audits across 7 cultural groups. Logistic Regression exhibited 100% counterfactual parity, while BiLSTM showed 66.7% parity. Our Gemini explanation audit confirmed a 100% groundedness score with zero hallucinations, and our safety firewall blocked 100% of adversarial prompt injection attacks. Finally, on Page 8, our multimodal laboratory integrates Gemini Vision to jointly evaluate food photos and text reviews, with gTTS review speech synthesis and generative culinary dish imaging."* | Switch to Page 7, show the Demographic Parity chart and run a live attack test in the prompt injection simulator. Switch to Page 8 and play a synthesized review audio clip. |
| **2:45 - 3:00** | **Page 10: Reports & Conclusion** | *"All source code is tested with 94 passing automated tests, fully reproducible with fixed random seeds, and ready for deployment with complete model cards and traceable requirements. Thank you!"* | Show Page 10 downloads, pytest 94/94 green badge, and conclude. |

---

## Pre-Demonstration Checklist

1. Virtual environment active: `.\.venv\Scripts\Activate.ps1`
2. `.env` file present with valid `GEMINI_API_KEY`
3. Launch command: `streamlit run app.py`
4. Browser open at full-screen (1080p / 16:9 aspect ratio)
5. Clear browser zoom (set to 100%)

