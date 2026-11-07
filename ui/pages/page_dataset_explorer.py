"""Page 2: Dataset Explorer for SentimentLab AI."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from config.settings import settings
from ui.components import render_header, render_metric_card, render_sentiment_badge


@st.cache_data
def load_split_data(split_name: str) -> pd.DataFrame:
    """Load parquet split from artifacts directory."""
    path_map = {
        "Train (Real, N=19,200)": settings.data_dir / "train_real.parquet",
        "Validation (Real, N=4,800)": settings.data_dir / "val_real.parquet",
        "Evaluation Benchmark (N=3,000)": settings.data_dir / "eval_subset_real.parquet",
        "Gemini Benchmark (N=300)": settings.data_dir / "gemini_benchmark_300.parquet",
        "Synthetic Reviews (N=500)": settings.data_dir / "synthetic_reviews.parquet",
    }
    target_path = path_map.get(split_name)
    if target_path and target_path.exists():
        df = pd.read_parquet(target_path)
        if "char_length" not in df.columns:
            df["char_length"] = df["review"].str.len()
        if "word_count" not in df.columns:
            df["word_count"] = df["review"].str.split().str.len()
        return df
    return pd.DataFrame()


def render():
    render_header(
        title="Dataset Explorer & Corpus Analytics",
        subtitle="Yelp Review Full three-class stratification, validation splits, and generative synthetic corpus",
        unit_tags=["Unit 1: Data Foundations", "Stratified Sampling", "Three-Class Taxonomy", "EDA"],
    )

    # Dataset Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Real Yelp Train", "19,200", subtitle="6,400 / class (Balanced)")
    with col2:
        render_metric_card("Real Yelp Val", "4,800", subtitle="1,600 / class (Balanced)")
    with col3:
        render_metric_card("Fixed Eval Subset", "3,000", subtitle="1,000 / class (Zero Leakage)")
    with col4:
        render_metric_card("Synthetic Reviews", "500", subtitle="Gemini-generated (Balanced)")

    st.markdown("---")

    # Split Selector and Filters
    col_sel, col_class, col_len = st.columns([2, 1, 1])
    with col_sel:
        selected_split = st.selectbox(
            "Select Dataset Split to Inspect:",
            [
                "Train (Real, N=19,200)",
                "Validation (Real, N=4,800)",
                "Evaluation Benchmark (N=3,000)",
                "Gemini Benchmark (N=300)",
                "Synthetic Reviews (N=500)",
            ],
        )
    with col_class:
        class_filter = st.selectbox("Filter Sentiment:", ["All Classes", "Negative (0)", "Neutral (1)", "Positive (2)"])
    with col_len:
        min_words = st.slider("Min Word Count:", 0, 150, 0)

    df = load_split_data(selected_split)
    if df.empty:
        st.info(f"ℹ️ Split data for '{selected_split}' is not bundled in this lightweight repository to stay within cloud storage limits. To generate full splits locally, execute: `python -m src.data.loader`.")
        return

    # Filter dataframe
    filtered_df = df.copy()
    if class_filter == "Negative (0)":
        filtered_df = filtered_df[filtered_df["label"] == 0]
    elif class_filter == "Neutral (1)":
        filtered_df = filtered_df[filtered_df["label"] == 1]
    elif class_filter == "Positive (2)":
        filtered_df = filtered_df[filtered_df["label"] == 2]

    filtered_df = filtered_df[filtered_df["word_count"] >= min_words]

    # Visualizations Row
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("#### 📊 Sentiment Class Distribution")
        label_counts = df["label"].value_counts().sort_index()
        label_names = {0: "Negative (0)", 1: "Neutral (1)", 2: "Positive (2)"}
        fig_bar = px.bar(
            x=[label_names[k] for k in label_counts.index],
            y=label_counts.values,
            labels={"x": "Sentiment Class", "y": "Number of Reviews"},
            color=[label_names[k] for k in label_counts.index],
            color_discrete_map={
                "Negative (0)": "#ef4444",
                "Neutral (1)": "#f59e0b",
                "Positive (2)": "#10b981",
            },
        )
        fig_bar.update_layout(template="plotly_dark", showlegend=False, height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_chart2:
        st.markdown("#### 📏 Review Word Count Distribution")
        fig_hist = px.histogram(
            df,
            x="word_count",
            color="label",
            nbins=40,
            labels={"word_count": "Word Count", "label": "Sentiment"},
            color_discrete_map={0: "#ef4444", 1: "#f59e0b", 2: "#10b981"},
        )
        fig_hist.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    # Interactive Table & Review Inspector
    st.markdown("---")
    st.markdown(f"#### 🔍 Sample Review Explorer ({len(filtered_df):,} matching reviews)")
    
    # Show interactive dataframe
    display_cols = ["label", "word_count", "char_length", "review"]
    st.dataframe(
        filtered_df[display_cols].head(100),
        use_container_width=True,
        height=320,
    )

    # Individual review detailed inspection
    st.markdown("##### 📖 Single Review Inspector")
    if not filtered_df.empty:
        sample_idx = st.slider("Select sample index:", 0, min(50, len(filtered_df) - 1), 0)
        sample_row = filtered_df.iloc[sample_idx]
        
        col_badge, col_meta = st.columns([1, 3])
        with col_badge:
            st.markdown(render_sentiment_badge(sample_row["label"]), unsafe_allow_html=True)
            st.caption(f"Words: {sample_row['word_count']} | Chars: {sample_row['char_length']}")
        with col_meta:
            st.info(f"❝ {sample_row['review']} ❞")

