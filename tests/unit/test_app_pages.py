"""Automated UI regression tests for all 10 Streamlit pages using Streamlit AppTest (Phase 5)."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent.parent / "app.py")


class TestStreamlitAppPages:
    """Validate that every page renders without unhandled exceptions or crashes."""

    def test_all_10_pages_render_without_exception(self):
        """Iterate through every section in the sidebar and verify clean rendering."""
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=10)
        assert len(at.exception) == 0

        radio = at.sidebar.radio[0]
        assert len(radio.options) == 10

        for page_name in radio.options:
            radio.set_value(page_name).run(timeout=10)
            assert len(at.exception) == 0, f"Exception on page '{page_name}': {at.exception}"

    def test_sentiment_lab_page_elements(self):
        """Verify the Live Sentiment Lab possesses all inputs, models, and buttons."""
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=10)
        
        radio = at.sidebar.radio[0]
        # Navigate to Page 3
        radio.set_value("⚡ 3. Live Sentiment Lab").run(timeout=10)
        assert len(at.exception) == 0

        # Check for model selectbox
        assert len(at.selectbox) >= 1
        model_select = at.selectbox[0]
        assert "Logistic Regression (Platt Calibrated)" in model_select.options
        assert "Bidirectional LSTM (Keras Sequential)" in model_select.options
        assert "Google Gemini 3.5 Flash-Lite (Zero-Shot)" in model_select.options

        # Check for text area and classify button
        assert len(at.text_area) >= 1
        assert len(at.button) >= 4  # Preset buttons + Classify button

