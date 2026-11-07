"""Page 4: Comprehensive Model Comparison & Benchmarks."""
import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from config.settings import settings
from ui.components import render_header, render_metric_card


def render():
    render_header(
        title="Model Benchmark & Evaluation Matrix",
        subtitle="Comprehensive cross-paradigm evaluation on the fixed 3,000-sample test set and 300 Gemini benchmark",
        unit_tags=["Units 2-4", "Metric Synthesis", "Brier Score", "ECE", "Latency Trade-offs"],
    )

    # High-level Summary Metrics Table
    summary_data = [
        {
            "Architecture": "Logistic Regression (Calibrated)",
            "Paradigm": "Discriminative (Linear)",
            "Eval N": "3,000",
            "Accuracy": 0.7420,
            "Macro F1": 0.7380,
            "Neg F1": 0.8120,
            "Neu F1": 0.5840,
            "Pos F1": 0.8180,
            "Brier Score": 0.3679,
            "ECE": 0.0380,
            "Latency (ms)": 0.03,
            "Size": "10 MB",
        },
        {
            "Architecture": "Logistic Regression (Uncalibrated)",
            "Paradigm": "Discriminative (Linear)",
            "Eval N": "3,000",
            "Accuracy": 0.7420,
            "Macro F1": 0.7380,
            "Neg F1": 0.8120,
            "Neu F1": 0.5840,
            "Pos F1": 0.8180,
            "Brier Score": 0.3705,
            "ECE": 0.0461,
            "Latency (ms)": 0.03,
            "Size": "10 MB",
        },
        {
            "Architecture": "Bidirectional LSTM (Keras)",
            "Paradigm": "Discriminative (RNN)",
            "Eval N": "3,000",
            "Accuracy": 0.6713,
            "Macro F1": 0.6740,
            "Neg F1": 0.7490,
            "Neu F1": 0.5180,
            "Pos F1": 0.7550,
            "Brier Score": 0.4331,
            "ECE": 0.0227,
            "Latency (ms)": 2.10,
            "Size": "32 MB",
        },
        {
            "Architecture": "BERT (Fine-Tuned Hugging Face)",
            "Paradigm": "Discriminative (Transformer)",
            "Eval N": "3,000",
            "Accuracy": 0.7533,
            "Macro F1": 0.7502,
            "Neg F1": 0.8240,
            "Neu F1": 0.6010,
            "Pos F1": 0.8260,
            "Brier Score": 0.3540,
            "ECE": 0.0290,
            "Latency (ms)": 14.50,
            "Size": "438 MB",
        },
        {
            "Architecture": "Google Gemini 3.5 (Zero-Shot)",
            "Paradigm": "Generative (In-Context)",
            "Eval N": "300",
            "Accuracy": 0.7667,
            "Macro F1": 0.7635,
            "Neg F1": 0.8350,
            "Neu F1": 0.6120,
            "Pos F1": 0.8430,
            "Brier Score": 0.3420,
            "ECE": 0.0410,
            "Latency (ms)": 850.0,
            "Size": "API Endpoint",
        },
        {
            "Architecture": "Google Gemini 3.5 (Few-Shot)",
            "Paradigm": "Generative (In-Context)",
            "Eval N": "300",
            "Accuracy": 0.7833,
            "Macro F1": 0.7810,
            "Neg F1": 0.8510,
            "Neu F1": 0.6380,
            "Pos F1": 0.8540,
            "Brier Score": 0.3290,
            "ECE": 0.0350,
            "Latency (ms)": 1120.0,
            "Size": "API Endpoint",
        },
    ]
    df_metrics = pd.DataFrame(summary_data)

    st.markdown("### 📋 Unified Model Benchmark Comparison Table")
    st.dataframe(
        df_metrics.style.format({
            "Accuracy": "{:.2%}",
            "Macro F1": "{:.2%}",
            "Neg F1": "{:.2%}",
            "Neu F1": "{:.2%}",
            "Pos F1": "{:.2%}",
            "Brier Score": "{:.4f}",
            "ECE": "{:.4f}",
            "Latency (ms)": "{:.2f}",
        }),
        use_container_width=True,
    )

    st.markdown("---")

    # Visual Comparison Charts
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🏆 Accuracy & Macro F1 Across Models")
        fig_acc = px.bar(
            df_metrics,
            x="Architecture",
            y=["Accuracy", "Macro F1"],
            barmode="group",
            labels={"value": "Score", "variable": "Metric"},
            color_discrete_map={"Accuracy": "#3b82f6", "Macro F1": "#8b5cf6"},
        )
        fig_acc.update_layout(template="plotly_dark", yaxis=dict(range=[0.5, 0.85]), height=360, margin=dict(l=20, r=20, t=20, b=80))
        st.plotly_chart(fig_acc, use_container_width=True)

    with col2:
        st.markdown("#### ⚡ Latency vs Accuracy Trade-off (Log Scale)")
        fig_scatter = px.scatter(
            df_metrics,
            x="Latency (ms)",
            y="Accuracy",
            color="Paradigm",
            size=[16, 16, 18, 22, 24, 26],
            text="Architecture",
            log_x=True,
            color_discrete_map={
                "Discriminative (Linear)": "#10b981",
                "Discriminative (RNN)": "#f59e0b",
                "Discriminative (Transformer)": "#3b82f6",
                "Generative (In-Context)": "#ec4899",
            },
        )
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(template="plotly_dark", height=360, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # Per-Class F1 Breakdown & Calibration Analysis
    col_class, col_cal = st.columns(2)
    with col_class:
        st.markdown("#### 🎯 Per-Class F1 Score Breakdown (Class Difficulty)")
        st.caption("Notice the substantial performance gap between polar classes (Neg/Pos ~82%) and Neutral (~58%)")
        class_f1_data = []
        for row in summary_data:
            class_f1_data.extend([
                {"Model": row["Architecture"][:18], "Sentiment": "Negative (0)", "F1": row["Neg F1"]},
                {"Model": row["Architecture"][:18], "Sentiment": "Neutral (1)", "F1": row["Neu F1"]},
                {"Model": row["Architecture"][:18], "Sentiment": "Positive (2)", "F1": row["Pos F1"]},
            ])
        df_class_f1 = pd.DataFrame(class_f1_data)
        fig_class = px.bar(
            df_class_f1,
            x="Model",
            y="F1",
            color="Sentiment",
            barmode="group",
            color_discrete_map={
                "Negative (0)": "#ef4444",
                "Neutral (1)": "#f59e0b",
                "Positive (2)": "#10b981",
            },
        )
        fig_class.update_layout(template="plotly_dark", yaxis=dict(range=[0.4, 0.95]), height=360, margin=dict(l=20, r=20, t=20, b=60))
        st.plotly_chart(fig_class, use_container_width=True)

    with col_cal:
        st.markdown("#### 📐 Multiclass Probability Calibration Impact")
        st.caption("Multiclass Platt calibration reduces Expected Calibration Error (ECE) and lowers Brier score")
        st.markdown(
            """
            - **Logistic Regression (Raw vs Calibrated):**
              - Raw Brier Score: `0.3705` ➔ Calibrated Brier Score: `0.3679` (Improved)
              - Raw ECE: `0.0461` ➔ Calibrated ECE: `0.0380` (17.5% Calibration Error Reduction)
            - **Bidirectional LSTM:**
              - Raw Brier Score: `0.4331` ➔ Retrained Aug Brier: `0.4173`
              - Raw ECE: `0.0227` ➔ Retrained Aug ECE: `0.0193`
            - **Academic Takeaway:** Deep networks (LSTM) benefit from temperature/data augmentation calibration, whereas linear TF-IDF benefits from sigmoid/Platt isotonic calibration.
            """
        )
        # Calibration curve visualization
        bins = [0.1, 0.3, 0.5, 0.7, 0.9]
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect Calibration", line=dict(dash="dash", color="#94a3b8")))
        fig_cal.add_trace(go.Scatter(x=bins, y=[0.14, 0.35, 0.56, 0.73, 0.88], mode="lines+markers", name="Uncalibrated LR", line=dict(color="#f87171")))
        fig_cal.add_trace(go.Scatter(x=bins, y=[0.11, 0.31, 0.51, 0.71, 0.91], mode="lines+markers", name="Platt Calibrated LR", line=dict(color="#34d399")))
        fig_cal.update_layout(
            template="plotly_dark",
            title="Reliability Diagram (Confidence vs Empirical Accuracy)",
            xaxis_title="Confidence Bin",
            yaxis_title="Empirical Accuracy",
            height=260,
            margin=dict(l=20, r=20, t=35, b=20),
        )
        st.plotly_chart(fig_cal, use_container_width=True)

