"""Reusable UI components and styles for SentimentLab AI."""
import streamlit as st
from typing import Optional, Dict, Any


def load_custom_css():
    """Inject polished modern CSS styling for academic presentation."""
    st.markdown(
        """
        <style>
        /* Main Container Styling */
        .main {
            background-color: #0e1117;
        }
        
        /* Metric Card Container */
        .metric-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 18px 22px;
            margin-bottom: 14px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
        }
        .metric-title {
            color: #94a3b8;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }
        .metric-value {
            color: #f8fafc;
            font-size: 1.85rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .metric-sub {
            color: #64748b;
            font-size: 0.8rem;
            margin-top: 4px;
        }
        .metric-delta-pos {
            color: #10b981;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .metric-delta-neg {
            color: #ef4444;
            font-weight: 600;
            font-size: 0.85rem;
        }

        /* Sentiment Badges */
        .badge-positive {
            display: inline-block;
            background: rgba(16, 185, 129, 0.15);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.35);
            padding: 4px 14px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.95rem;
        }
        .badge-neutral {
            display: inline-block;
            background: rgba(245, 158, 11, 0.15);
            color: #f59e0b;
            border: 1px solid rgba(245, 158, 11, 0.35);
            padding: 4px 14px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.95rem;
        }
        .badge-negative {
            display: inline-block;
            background: rgba(239, 68, 68, 0.15);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.35);
            padding: 4px 14px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.95rem;
        }

        /* Banner Header */
        .header-banner {
            background: linear-gradient(90deg, #1e1b4b 0%, #311042 50%, #1e1b4b 100%);
            border: 1px solid rgba(139, 92, 246, 0.3);
            border-radius: 14px;
            padding: 24px 28px;
            margin-bottom: 24px;
        }
        .header-title {
            color: #ffffff;
            font-size: 2.1rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .header-subtitle {
            color: #cbd5e1;
            font-size: 1.05rem;
            margin-top: 6px;
            font-weight: 400;
        }
        .tag-pill {
            display: inline-block;
            background: rgba(139, 92, 246, 0.25);
            color: #c4b5fd;
            border: 1px solid rgba(139, 92, 246, 0.45);
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 6px;
            margin-top: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(
    title: str,
    subtitle: str,
    unit_tags: list[str] = None
):
    """Render a unified top banner across all pages."""
    tags_html = ""
    if unit_tags:
        tags_html = "".join([f'<span class="tag-pill">{tag}</span>' for tag in unit_tags])
        
    st.markdown(
        f"""
        <div class="header-banner">
            <h1 class="header-title">🧪 {title}</h1>
            <div class="header-subtitle">{subtitle}</div>
            <div style="margin-top: 8px;">{tags_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    title: str,
    value: str,
    delta: Optional[str] = None,
    delta_positive: bool = True,
    subtitle: Optional[str] = None,
):
    """Render a modern glassmorphism metric card."""
    delta_html = ""
    if delta:
        delta_class = "metric-delta-pos" if delta_positive else "metric-delta-neg"
        arrow = "▲" if delta_positive else "▼"
        delta_html = f'<div class="{delta_class}">{arrow} {delta}</div>'
        
    sub_html = f'<div class="metric-sub">{subtitle}</div>' if subtitle else ""
    
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sentiment_badge(label: int | str) -> str:
    """Return styled HTML string for sentiment label."""
    if label in (2, "2", "Positive", "positive"):
        return '<span class="badge-positive">● Positive</span>'
    elif label in (1, "1", "Neutral", "neutral"):
        return '<span class="badge-neutral">● Neutral</span>'
    else:
        return '<span class="badge-negative">● Negative</span>'

