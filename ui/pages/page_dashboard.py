"""Page 1: Executive Dashboard for SentimentLab AI."""
import streamlit as st
import pandas as pd
import json
from pathlib import Path
from config.settings import settings
from ui.components import render_header, render_metric_card


def render():
    render_header(
        title="SentimentLab AI - Executive Dashboard",
        subtitle="Generative vs Discriminative Sentiment Analysis Laboratory (P4 Academic Mini-Project)",
        unit_tags=["Units U1-U5", "Generative AI", "Discriminative NLP", "Data Augmentation", "Responsible AI"],
    )

    # Top high-level KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card(
            title="Real Yelp Corpus",
            value="24,000",
            subtitle="19,200 Train / 4,800 Val",
        )
    with col2:
        render_metric_card(
            title="Synthetic Reviews",
            value="500",
            delta="+2.6% Data Size",
            delta_positive=True,
            subtitle="Validated & Audited (0 Dups)",
        )
    with col3:
        render_metric_card(
            title="Evaluation Benchmark",
            value="3,000",
            subtitle="Stratified (1k Neg, 1k Neu, 1k Pos)",
        )
    with col4:
        render_metric_card(
            title="Models Benchmarked",
            value="5 Models",
            subtitle="LR, Calibrated LR, BiLSTM, BERT, Gemini",
        )

    st.markdown("---")

    # Core Academic Paradigm Comparison
    st.markdown("### 🔬 Paradigm Comparison: Discriminative vs Generative Approaches")
    
    col_disc, col_gen = st.columns(2)
    with col_disc:
        st.info(
            """
            #### ⚙️ Discriminative Sentiment Analysis
            - **Mathematical Foundation:** Direct modeling of conditional posterior distribution $P(Y \\mid X)$.
            - **Decision Boundary:** Optimizes hyperplanes or deep representations to maximize mutual information between text features and discrete sentiment labels $\\{0, 1, 2\\}$.
            - **Models Studied:**
              1. **TF-IDF + Logistic Regression:** High-dimensional sparse n-gram linear baseline with Platt/Isotonic multiclass calibration.
              2. **Bidirectional LSTM:** Sequential recurrence capturing bidirectional temporal dependencies and local syntax.
              3. **Fine-tuned BERT (Transformer):** Bidirectional masked self-attention pre-trained on large-scale corpora.
            - **Operational Characteristics:** Ultra-low inference latency (<1ms to 20ms), deterministic outputs, zero hallucination risk, requires labeled training corpus.
            """
        )

    with col_gen:
        st.success(
            """
            #### 🧠 Generative Sentiment Analysis
            - **Mathematical Foundation:** Autoregressive language modeling $P(X) = \\prod_{t=1}^T P(w_t \\mid w_{<t})$ and structured reasoning.
            - **Inference Mechanism:** In-context learning via Zero-shot and Few-shot prompting without parameter updates during test time.
            - **Models & Capabilities:**
              1. **Google Gemini (2.5 / 3.5 Flash):** Foundation LLM performing zero-shot, few-shot, and chain-of-thought sentiment classification.
              2. **Synthetic Data Synthesis:** Generates 500 domain-authentic customer reviews to augment low-resource/neutral training regimes.
              3. **Multimodal Reasoning:** Unified vision-language reasoning over restaurant photos and review text.
            - **Operational Characteristics:** High reasoning flexibility, zero labeled samples required for basic tasks, higher API latency and cost.
            """
        )

    st.markdown("---")

    # Experimental Lab Modules Grid
    st.markdown("### 🗂️ Experimental Laboratory Modules")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            """
            **1. Dataset Explorer**  
            Explore the Yelp 24k corpus, class balances, token length distributions, and inspect individual reviews.
            
            **2. Sentiment Lab**  
            Run real-time inference on custom customer reviews across all discriminative and generative models.
            """
        )
    with col_b:
        st.markdown(
            """
            **3. Model Comparison & Benchmarks**  
            Side-by-side evaluation metrics (Accuracy, Macro F1, Brier Score, ECE, Latency), confusion matrices, and calibration curves.
            
            **4. Synthetic Review Lab**  
            Audit the 500 Gemini-generated synthetic reviews, lexical diversity, and cosine similarity checks.
            """
        )
    with col_c:
        st.markdown(
            """
            **5. Retraining Experiments**  
            Examine the empirical impact of real-only vs real+synthetic training (Condition A vs B vs C).
            
            **6. Responsible AI & Multimodal**  
            Counterfactual regional bias audits, hallucination evaluations, safety guardrails, and Gemini vision analysis.
            """
        )

