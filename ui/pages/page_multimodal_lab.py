"""Page 8: Multimodal Vision, Speech & Generative Photography Lab (Units 3 & 4)."""
import streamlit as st
import tempfile
from pathlib import Path
from PIL import Image
import io
from config.settings import settings
from ui.components import render_header, render_metric_card, render_sentiment_badge
from src.multimodal.multimodal_service import MultimodalSentimentService


@st.cache_resource
def get_multimodal_service():
    return MultimodalSentimentService()


def render():
    render_header(
        title="Multimodal Sentiment & Generative Media Laboratory",
        subtitle="Visual food analysis with Gemini Flash Vision, Review Speech Synthesis (gTTS), and Generative Dish Photography",
        unit_tags=["Units 3 & 4", "Multimodal Vision-Language", "Text-to-Speech (gTTS)", "Generative Photography", "Cross-Modal Consistency"],
    )

    mm_service = get_multimodal_service()

    tab_vision, tab_tts, tab_dish = st.tabs([
        "👁️ 1. Multimodal Vision Sentiment",
        "🎙️ 2. Review Audio Synthesis (gTTS)",
        "🎨 3. Generative Dish Photography",
    ])

    # TAB 1: VISION SENTIMENT
    with tab_vision:
        st.markdown("### 📸 Unified Visual + Textual Sentiment Analysis")
        st.markdown(
            """
            In modern hospitality and e-commerce platforms, customer reviews frequently include photographs.
            A review may say *\"Food was okay\"*, but the attached photo reveals an overcooked, burned steak.
            Google Gemini Vision evaluates **both modalities simultaneously** to detect cross-modal sentiment consistency.
            """
        )

        col_img, col_txt = st.columns([1, 1])
        with col_img:
            uploaded_file = st.file_uploader("Upload Dish / Restaurant Photo (JPG, PNG):", type=["jpg", "jpeg", "png"])
            # Fallback preset sample image if none uploaded
            sample_choice = st.selectbox("Or choose a pre-loaded culinary sample:", [
                "Custom Upload",
                "Sample A: Artisanal Truffle Pasta (High Quality)",
                "Sample B: Burnt Undercooked Pizza (Defective Quality)",
            ])
            
            image_to_analyze = None
            if uploaded_file is not None:
                image_to_analyze = Image.open(uploaded_file)
                st.image(image_to_analyze, caption="Uploaded Image", use_column_width=True)
            elif sample_choice != "Custom Upload":
                # Create a simple synthetic PIL image swatch for testing
                if "Sample A" in sample_choice:
                    image_to_analyze = Image.new("RGB", (300, 300), color=(245, 230, 210))
                    st.image(image_to_analyze, caption="Preset: Artisanal Pasta", use_column_width=True)
                else:
                    image_to_analyze = Image.new("RGB", (300, 300), color=(60, 40, 30))
                    st.image(image_to_analyze, caption="Preset: Overcooked Pizza", use_column_width=True)

        with col_txt:
            review_text = st.text_area(
                "Accompanying Customer Text Review:",
                value="The pasta arrived piping hot with rich black truffle shavings. Plating was pristine!",
                height=130,
            )
            analyze_btn = st.button("🔍 Run Multimodal Vision Inference", type="primary", use_container_width=True)

        if analyze_btn and image_to_analyze:
            with st.spinner("Analyzing cross-modal visual and textual cues via Gemini 2.5/3.5 Vision..."):
                result = mm_service.analyze_image_and_text(image=image_to_analyze, text=review_text)
                
                st.markdown("---")
                st.markdown("#### 🎯 Multimodal Classification Output")
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    render_metric_card("Multimodal Sentiment", result.get("multimodal_sentiment", "Positive"))
                with m_col2:
                    render_metric_card("Confidence", f"{result.get('confidence', 0.95)*100:.1f}%")
                with m_col3:
                    render_metric_card("Visual Consistency", "High Agreement" if result.get("visual_consistency") else "Moderate")

                st.info(f"**Visual & Text Rationale:** {result.get('rationale', '')}")
                if result.get("visual_aspects"):
                    st.write("**Identified Visual Aspects:**", ", ".join(result.get("visual_aspects", [])))

    # TAB 2: SPEECH SYNTHESIS (gTTS)
    with tab_tts:
        st.markdown("### 🎙️ Review Text-to-Speech (gTTS) Player")
        st.markdown("Synthesize natural-sounding audio for any customer review to enable voice-assisted accessibility and podcast-style review feeds.")
        
        tts_text = st.text_area(
            "Review Text for Audio Synthesis:",
            value="The service at Bistro Lumière was attentive, but the sea bass lacked seasoning and was somewhat dry for forty dollars.",
            height=100,
        )
        col_accent, col_play = st.columns([1, 1])
        with col_accent:
            tld = st.selectbox("Select Voice Accent (Top-Level Domain):", ["com (US English)", "co.uk (British English)", "ca (Canadian)", "com.au (Australian)"])
            selected_tld = tld.split()[0]
        with col_play:
            st.write("")
            st.write("")
            synth_audio_btn = st.button("🔊 Synthesize Speech MP3", use_container_width=True)

        if synth_audio_btn and tts_text.strip():
            with st.spinner("Generating speech audio with gTTS..."):
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                    audio_path = tmp_file.name
                
                audio_path = mm_service.synthesize_speech(tts_text, output_path=audio_path, tld=selected_tld)
                st.success("✅ Speech synthesized successfully:")
                st.audio(audio_path, format="audio/mp3")

    # TAB 3: GENERATIVE DISH PHOTOGRAPHY
    with tab_dish:
        st.markdown("### 🎨 Generative Culinary Photography (Review-to-Image)")
        st.markdown(
            """
            Transform the text of a customer review into a **photorealistic culinary visualization** 
            of the described dish using diffusion-based generative imaging.
            """
        )

        dish_prompt = st.text_input(
            "Culinary Visual Prompt:",
            value="A plate of artisanal hand-rolled gnocchi in rich sage brown butter sauce with microgreens, Michelin star presentation, dramatic studio lighting",
        )
        gen_dish_btn = st.button("✨ Generate Dish Photograph", type="primary")

        if gen_dish_btn and dish_prompt:
            with st.spinner("Synthesizing generative culinary photograph..."):
                img_path = mm_service.generate_dish_image(dish_prompt)
                if Path(img_path).exists():
                    st.image(img_path, caption=f"Generative Visualization: \"{dish_prompt}\"", use_column_width=True)
                else:
                    st.warning("Image service currently in local mode.")

