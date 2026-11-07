"""SentimentLab AI - Main Streamlit Application Entrypoint.

Generative vs Discriminative Sentiment Analysis Laboratory (P4 Academic Mini-Project).
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import settings
settings.ensure_directories()

from ui.components import load_custom_css
from ui.pages import (
    page_dashboard,
    page_dataset_explorer,
    page_sentiment_lab,
    page_model_comparison,
    page_synthetic_lab,
    page_retraining_experiments,
    page_responsible_ai,
    page_multimodal_lab,
    page_academic_lab,
    page_reports,
)

# 1. Page Configuration
st.set_page_config(
    page_title="SentimentLab AI - Generative vs Discriminative",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject Custom Theme & CSS
load_custom_css()

# 3. Sidebar Branding & Navigation
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 12px 0 20px 0;">
            <h2 style="color: #c4b5fd; margin: 0; font-size: 1.6rem; font-weight: 800;">🧪 SentimentLab AI</h2>
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 4px 0 0 0;">Generative vs Discriminative Sentiment Lab</p>
            <span style="display: inline-block; background: rgba(139, 92, 246, 0.2); color: #c4b5fd; border: 1px solid rgba(139, 92, 246, 0.4); padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: 600; margin-top: 6px;">P4 Academic Project | Units U1–U5</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("---")
    
    page_options = [
        "📊 1. Executive Dashboard",
        "🗂️ 2. Dataset Explorer",
        "⚡ 3. Live Sentiment Lab",
        "📈 4. Model Comparison & Benchmarks",
        "🧪 5. Synthetic Review Lab",
        "🔄 6. Retraining Experiments",
        "⚖️ 7. Responsible AI & Trust",
        "👁️ 8. Multimodal Sentiment Lab",
        "🎓 9. Academic Learning Lab (U1-U5)",
        "📥 10. Reports & Data Exports",
    ]
    
    selected_page = st.radio("Navigate Laboratory:", page_options, index=0)
    
    st.markdown("---")
    st.markdown("##### ⚙️ System Status")
    st.caption("• **Runtime:** Python 3.11.9 (.venv)")
    st.caption("• **Models:** LR, BiLSTM, BERT, Gemini 3.5")
    st.caption("• **Corpus:** 24k Real + 500 Synthetic")
    st.caption("• **Fixed Benchmark:** 3,000 Stratified")

# 4. Router
if "1. Executive Dashboard" in selected_page:
    page_dashboard.render()
elif "2. Dataset Explorer" in selected_page:
    page_dataset_explorer.render()
elif "3. Live Sentiment Lab" in selected_page:
    page_sentiment_lab.render()
elif "4. Model Comparison" in selected_page:
    page_model_comparison.render()
elif "5. Synthetic Review Lab" in selected_page:
    page_synthetic_lab.render()
elif "6. Retraining Experiments" in selected_page:
    page_retraining_experiments.render()
elif "7. Responsible AI" in selected_page:
    page_responsible_ai.render()
elif "8. Multimodal" in selected_page:
    page_multimodal_lab.render()
elif "9. Academic Learning Lab" in selected_page:
    page_academic_lab.render()
elif "10. Reports" in selected_page:
    page_reports.render()

