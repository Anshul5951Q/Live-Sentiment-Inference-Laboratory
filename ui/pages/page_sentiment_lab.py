"""Page 3: Live Interactive Sentiment Lab."""
import streamlit as st
import numpy as np
import time
import plotly.graph_objects as go
from pathlib import Path
from config.settings import settings
from ui.components import render_header, render_metric_card, render_sentiment_badge
from src.llm.prompts import (
    format_zero_shot_prompt,
    format_few_shot_prompt,
    build_zero_shot_prompt,
    build_few_shot_prompt,
)


# Model loaders with caching
@st.cache_resource
def load_lr_models():
    """Load Logistic Regression models and vectorizer."""
    from src.models.tfidf_vectorizer import TextTfidfVectorizer
    from src.models.logistic_regression import LogisticRegressionClassifier
    from src.models.calibration import MulticlassCalibrator

    vec_path = settings.models_dir / "logistic_regression" / "tfidf_vectorizer.joblib"
    lr_path = settings.models_dir / "logistic_regression" / "logistic_regression.joblib"
    cal_path = settings.models_dir / "logistic_regression" / "calibrated_logistic_regression.joblib"

    if not vec_path.exists() or not lr_path.exists() or not cal_path.exists():
        raise FileNotFoundError(f"Logistic Regression artifacts missing in {settings.models_dir / 'logistic_regression'}")

    vec = TextTfidfVectorizer().load(vec_path)
    lr = LogisticRegressionClassifier().load(lr_path)
    cal = MulticlassCalibrator().load(cal_path)
    return vec, lr, cal


@st.cache_resource
def load_bilstm_models():
    """Load BiLSTM model and tokenizer."""
    from src.models.text_tokenizer import TextTokenizer
    from src.models.lstm_classifier import KerasBiLSTMClassifier

    tok_path = settings.models_dir / "lstm" / "tokenizer.json"
    clf_path = settings.models_dir / "lstm" / "lstm_model.keras"

    if not tok_path.exists() or not clf_path.exists():
        raise FileNotFoundError(f"BiLSTM artifacts missing in {settings.models_dir / 'lstm'}")

    tok = TextTokenizer().load(tok_path)
    clf = KerasBiLSTMClassifier().load(clf_path)
    return tok, clf


def render():
    render_header(
        title="Live Sentiment Inference Laboratory",
        subtitle="Real-time comparative sentiment classification across discriminative baselines and generative LLMs",
        unit_tags=["Units 2-4", "Real-Time Inference", "Confidence Calibration", "Few-Shot Prompting"],
    )

    # State management for preset inputs
    default_text = "The artisanal wood-fired pizza was absolutely magnificent! Fresh basil, silky mozzarella, and incredibly attentive service. Five stars all around!"
    if "current_review_text" not in st.session_state:
        st.session_state["current_review_text"] = default_text

    # Preset examples for fast testing
    st.markdown("##### 💡 Quick Test Presets:")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    if col_p1.button("🌟 Enthusiastic Positive"):
        st.session_state["current_review_text"] = "The artisanal wood-fired pizza was absolutely magnificent! Fresh basil, silky mozzarella, and incredibly attentive service. Five stars all around!"
        st.rerun()
    if col_p2.button("⚖️ Subtle Neutral"):
        st.session_state["current_review_text"] = "The food arrived within twenty minutes. Prices are typical for downtown dining. It was acceptable for a quick lunch, neither outstanding nor terrible."
        st.rerun()
    if col_p3.button("⚠️ Disappointed Negative"):
        st.session_state["current_review_text"] = "Waited over forty minutes for cold pasta and undercooked chicken. The manager was dismissive when we pointed out the issue. Will not return."
        st.rerun()
    if col_p4.button("🎭 Sarcastic Ambiguous"):
        st.session_state["current_review_text"] = "Oh wonderful, another place that charges twenty dollars for tap water and warm salad. What an unforgettable culinary experience."
        st.rerun()

    review_input = st.text_area(
        "Enter Customer Review Text:",
        value=st.session_state["current_review_text"],
        height=110,
        key="review_text_area",
    )

    col_model, col_btn = st.columns([3, 1])
    with col_model:
        model_choice = st.selectbox(
            "Select Architecture for Inference:",
            [
                "Logistic Regression (Platt Calibrated)",
                "Logistic Regression (Uncalibrated)",
                "Bidirectional LSTM (Keras Sequential)",
                "Fine-tuned BERT (Transformer)",
                "Google Gemini 3.5 Flash-Lite (Zero-Shot)",
                "Google Gemini 3.5 Flash-Lite (Few-Shot)",
            ],
        )
    with col_btn:
        st.write("")
        st.write("")
        run_btn = st.button("🚀 Classify Sentiment", type="primary", use_container_width=True)

    if run_btn:
        if not review_input.strip():
            st.warning("⚠️ Review text cannot be empty. Please enter or select a review to classify.")
            return

        with st.spinner(f"Running inference with {model_choice}..."):
            start_t = time.time()
            probs = None
            pred_label = None
            explanation = ""
            source_badge = ""
            error_msg = None

            try:
                if "Logistic Regression" in model_choice:
                    vec, lr, cal = load_lr_models()
                    X = vec.transform([review_input])
                    raw_p = lr.predict_proba(X)[0]
                    cal_p = cal.predict_proba(X)[0]
                    latency_ms = (time.time() - start_t) * 1000.0

                    if "Calibrated" in model_choice:
                        probs = cal_p
                        pred_label = int(np.argmax(cal_p))
                        explanation = "Probability calibrated using Multiclass Platt Scaling (Sigmoid) fitted on validation set."
                    else:
                        probs = raw_p
                        pred_label = int(np.argmax(raw_p))
                        explanation = "Raw uncalibrated multinomial logistic regression probabilities from TF-IDF features."
                    source_badge = "Local Scikit-learn Model"

                elif "Bidirectional LSTM" in model_choice:
                    tok, clf = load_bilstm_models()
                    seq = tok.transform([review_input])
                    raw_p = clf.predict_proba(seq)[0]
                    probs = raw_p / np.sum(raw_p)
                    pred_label = int(np.argmax(probs))
                    latency_ms = (time.time() - start_t) * 1000.0
                    explanation = "Sequential bidirectional LSTM processing forward and backward token context."
                    source_badge = "Local Keras BiLSTM Model"

                elif "BERT" in model_choice:
                    bert_path = settings.models_dir / "bert" / "final_model"
                    bert_root = settings.models_dir / "bert"
                    
                    if (bert_path / "config.json").exists() or (bert_root / "config.json").exists():
                        target_dir = bert_path if (bert_path / "config.json").exists() else bert_root
                        from transformers import AutoTokenizer, AutoModelForSequenceClassification
                        import torch
                        tokenizer = AutoTokenizer.from_pretrained(str(target_dir))
                        model = AutoModelForSequenceClassification.from_pretrained(str(target_dir))
                        inputs = tokenizer(review_input, return_tensors="pt", truncation=True, max_length=128)
                        with torch.no_grad():
                            logits = model(**inputs).logits
                            probs = torch.softmax(logits, dim=1).numpy()[0]
                        pred_label = int(np.argmax(probs))
                        explanation = f"Fine-tuned BERT transformer weights loaded from {target_dir.name}."
                        source_badge = "Local Hugging Face BERT"
                    else:
                        # Informative fallback using validated discriminative baseline
                        vec, lr, cal = load_lr_models()
                        X = vec.transform([review_input])
                        probs = cal.predict_proba(X)[0]
                        pred_label = int(np.argmax(probs))
                        explanation = "Notice: Local BERT fine-tuned weights not found in artifacts/models/bert/. Displaying high-fidelity discriminative proxy. (Train on GPU via notebooks/01_colab_bert_training.ipynb to download full checkpoint)."
                        source_badge = "Discriminative Proxy (BERT pending Colab download)"
                    latency_ms = (time.time() - start_t) * 1000.0

                elif "Gemini" in model_choice:
                    from src.llm.gemini_provider import GeminiProvider
                    provider = GeminiProvider()
                    is_few_shot = "Few-Shot" in model_choice
                    mode = "few_shot" if is_few_shot else "zero_shot"

                    if not provider.api_key:
                        st.error("🔑 GEMINI_API_KEY is not configured in your environment or .env file. Please configure your key in .env to execute live Gemini calls.")
                        return

                    res = provider.classify_sentiment(review_input, mode=mode)
                    latency_ms = res.inference_duration_ms if res.inference_duration_ms > 0 else (time.time() - start_t) * 1000.0
                    
                    if res.error_status:
                        error_msg = res.error_status
                        st.error(f"⚠️ Gemini API Error: {res.error_status}")
                        if "RESOURCE_EXHAUSTED" in res.error_status or "429" in res.error_status:
                            st.warning("Daily free-tier request quota reached for current model. You may switch GEMINI_MODEL in .env or wait for quota reset.")
                        return

                    pred_label = res.predicted_label
                    conf = max(0.01, min(0.99, res.confidence))
                    explanation = res.reason or "Classified via Google Gemini in-context reasoning."
                    
                    # Create normalized 3-class distribution for visualization
                    # Place reported confidence on predicted class, divide remainder
                    rem = (1.0 - conf) / 2.0
                    probs = np.array([rem, rem, rem])
                    probs[pred_label] = conf
                    
                    src = res.metadata.get("source", "live_api")
                    source_badge = f"Google Gemini ({provider.model_name} | {src.upper()})"

            except Exception as ex:
                st.error(f"❌ Execution failed for {model_choice}: {str(ex)}")
                return

            # Display Results
            st.markdown("---")
            st.markdown("### 🎯 Classification Results")

            res_col1, res_col2, res_col3 = st.columns([1, 1, 2])
            with res_col1:
                render_metric_card(
                    title="Predicted Class",
                    value=["Negative", "Neutral", "Positive"][pred_label],
                    subtitle=f"Class Index: {pred_label}",
                )
            with res_col2:
                render_metric_card(
                    title="Model Confidence",
                    value=f"{probs[pred_label] * 100:.1f}%",
                    subtitle=f"Latency: {latency_ms:.1f} ms",
                )
            with res_col3:
                # Softmax Probability Distribution Bar
                fig = go.Figure(
                    go.Bar(
                        x=["Negative (0)", "Neutral (1)", "Positive (2)"],
                        y=probs,
                        marker_color=["#ef4444", "#f59e0b", "#10b981"],
                        text=[f"{p*100:.1f}%" for p in probs],
                        textposition="auto",
                    )
                )
                fig.update_layout(
                    template="plotly_dark",
                    title="Probability Distribution P(Y|X)",
                    yaxis=dict(range=[0, 1.05]),
                    height=240,
                    margin=dict(l=20, r=20, t=35, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)

            if source_badge:
                st.caption(f"**Execution Engine:** `{source_badge}` | **Latency:** `{latency_ms:.2f} ms`")

            if explanation:
                st.info(f"**Model Rationale / Details:** {explanation}")
