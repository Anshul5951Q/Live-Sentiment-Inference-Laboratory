"""Page 10: Reports, Reproducibility Metadata & Exports."""
import streamlit as st
import json
import pandas as pd
from pathlib import Path
from config.settings import settings
from ui.components import render_header


def render():
    render_header(
        title="Reports, Reproducibility & Data Exports",
        subtitle="Download validated artifacts, benchmark reports, synthetic datasets, and audit logs",
        unit_tags=["Reproducibility", "Data Export", "Artifact Auditing", "Academic Transparency"],
    )

    st.markdown("### 📥 Download Verified Project Artifacts")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("#### 📊 Evaluation & Retraining Reports")
        
        # Retraining Report
        retrain_path = settings.results_dir / "retraining_experiments" / "augmented_experiment_report.json"
        if retrain_path.exists():
            with open(retrain_path, "rb") as f:
                st.download_button(
                    label="📄 Download Retraining Matrix (JSON)",
                    data=f.read(),
                    file_name="augmented_experiment_report.json",
                    mime="application/json",
                    use_container_width=True,
                )

        # Responsible AI Report
        res_ai_path = settings.results_dir / "responsible_ai" / "bias_audit_logistic_regression.json"
        if res_ai_path.exists():
            with open(res_ai_path, "rb") as f:
                st.download_button(
                    label="⚖️ Download Bias Audit Results (JSON)",
                    data=f.read(),
                    file_name="bias_audit_logistic_regression.json",
                    mime="application/json",
                    use_container_width=True,
                )

    with col_d2:
        st.markdown("#### 🧪 Synthetic Datasets")
        synth_path = settings.data_dir / "synthetic_reviews.parquet"
        if synth_path.exists():
            df_synth = pd.read_parquet(synth_path)
            csv_data = df_synth.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📁 Download 500 Synthetic Reviews (CSV)",
                data=csv_data,
                file_name="synthetic_reviews_500.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Quality Audit Report
        audit_path = settings.results_dir / "synthetic_analysis" / "synthetic_quality_report.json"
        if audit_path.exists():
            with open(audit_path, "rb") as f:
                st.download_button(
                    label="🛡️ Download Synthetic Quality Audit (JSON)",
                    data=f.read(),
                    file_name="synthetic_quality_report.json",
                    mime="application/json",
                    use_container_width=True,
                )

    st.markdown("---")

    # Reproducibility Metadata
    st.markdown("### 🔬 Reproducibility Specifications & Environment")
    meta = {
        "Project Name": "SentimentLab AI (P4 Generative vs Discriminative Sentiment Lab)",
        "Curriculum Units": "U1, U2, U3, U4, U5",
        "Python Runtime": "Python 3.11.9 (Windows 11 AMD64)",
        "Random Seed": settings.random_seed,
        "Total Real Yelp Reviews": 24000,
        "Training Samples (Real)": 19200,
        "Validation Samples (Real)": 4800,
        "Fixed Evaluation Subset": 3000,
        "Gemini Benchmark Subset": 300,
        "Validated Synthetic Reviews": 500,
        "Condition B Retrained Size": 19700,
        "Condition C Retrained Size": 19451,
        "Gemini Models": "gemini-3.5-flash-lite & gemini-2.5-flash",
        "Primary Metric": "Macro-Averaged F1 Score & Accuracy",
        "Calibration Metric": "Multiclass Brier Score & Expected Calibration Error (ECE)",
    }
    st.json(meta)

