"""Multimodal review intelligence and generative media service (Units 3 & 4).

Features:
1. Multimodal Review Analysis: Joint visual + text sentiment understanding using Gemini 2.5 Flash Vision.
2. Audio Review Synthesis: Text-to-speech audio rendering using gTTS for auditory review playback.
3. Generative Media Synthesis: AI-powered dish / restaurant imagery generation from textual review prompts.
"""
import base64
import io
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from config.settings import settings
from config.logging_config import setup_logger
from src.llm.gemini_provider import GeminiProvider
from src.utils.file_io import save_json

logger = setup_logger(__name__)


class MultimodalSentimentService:
    """Provides multimodal perception and generative audio/image capabilities for customer reviews."""

    def __init__(
        self,
        gemini_provider: Optional[GeminiProvider] = None,
        media_output_dir: Optional[Path] = None,
    ):
        self.provider = gemini_provider or GeminiProvider()
        self.output_dir = media_output_dir or (settings.results_dir / "multimodal")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_image_and_review(
        self,
        image_bytes: bytes,
        review_text: str,
        mime_type: str = "image/jpeg",
    ) -> Dict[str, Any]:
        """Analyze a customer food photo alongside their text review using Gemini Vision.

        Args:
            image_bytes: Raw binary bytes of the food/restaurant image.
            review_text: The user's accompanying written text.
            mime_type: MIME format of the image (e.g. image/jpeg, image/png).

        Returns:
            Structured dictionary with visual findings, joint sentiment, confidence, and explanation.
        """
        logger.info("Executing multimodal sentiment analysis (Image size: %d bytes)...", len(image_bytes))
        prompt = f"""You are an expert multimodal restaurant quality and sentiment analyst.
Analyze both the provided customer photograph and the accompanying written review text:
Customer Review: "{review_text}"

Respond strictly in valid JSON matching this exact structure:
{{
  "visual_description": "Concise factual description of what is visible in the photo (presentation, portion, dish quality)",
  "text_sentiment": "Negative, Neutral, or Positive",
  "visual_sentiment": "Negative, Neutral, or Positive (based on food appearance/plating/freshness)",
  "overall_joint_sentiment": "Negative, Neutral, or Positive",
  "confidence": 0.95,
  "explanation": "Synthesized rationale reconciling both visual and textual evidence"
}}
"""
        if getattr(self.provider, "_client", None) is None:
            logger.warning("Gemini client not initialized. Operating in fallback multimodal mode.")
            return {
                "visual_description": "Customer uploaded food presentation photo.",
                "text_sentiment": "Positive",
                "visual_sentiment": "Positive",
                "overall_joint_sentiment": "Positive",
                "confidence": 0.85,
                "explanation": "Combined visual presentation and positive text comment.",
                "source": "offline_fallback",
            }

        try:
            from google.genai import types
            response = self.provider._client.models.generate_content(
                model=self.provider.model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    prompt,
                ],
                config={"temperature": 0.2}
            )
            cleaned = self.provider._clean_json_text(response.text)
            parsed = json.loads(cleaned)
            parsed["source"] = "gemini_multimodal_vision"
            return parsed
        except Exception as e:
            logger.error("Multimodal analysis failed: %s", str(e))
            return {
                "visual_description": "Photo of food item.",
                "text_sentiment": "Neutral",
                "visual_sentiment": "Neutral",
                "overall_joint_sentiment": "Neutral",
                "confidence": 0.5,
                "explanation": f"Multimodal analysis encountered API notice: {str(e)[:100]}",
                "source": "error_fallback",
            }

    def synthesize_review_speech(
        self,
        review_text: str,
        filename: str = "review_audio.mp3",
        output_path: Optional[Union[str, Path]] = None,
        lang: str = "en",
        tld: str = "com",
    ) -> Path:
        """Synthesize natural spoken audio of a customer review using gTTS.

        Args:
            review_text: Text of the review to vocalize.
            filename: Destination audio filename in media output dir.
            output_path: Optional explicit output path.
            lang: Language code (default 'en').
            tld: Top level domain for regional accent (e.g. 'com', 'co.uk').

        Returns:
            Path to the saved MP3 audio file.
        """
        from gtts import gTTS
        if output_path:
            out_path = Path(output_path)
        else:
            out_path = self.output_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Synthesizing audio review (%d characters) -> %s", len(review_text), out_path)
        
        # Limit text length for TTS to prevent excessive processing
        truncated_text = review_text[:600]
        tts = gTTS(text=truncated_text, lang=lang, tld=tld, slow=False)
        tts.save(str(out_path))
        return out_path

    # Functional alias
    synthesize_speech = synthesize_review_speech

    def generate_dish_image(
        self,
        prompt: str,
        filename: str = "generated_dish.jpg",
        width: int = 512,
        height: int = 384,
    ) -> Path:
        """Generate a realistic culinary or restaurant photo matching a review prompt (Unit 4).

        Args:
            prompt: Text prompt describing the dish or restaurant setting.
            filename: Output filename for the generated image.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            Path to the saved image file on disk.
        """
        out_path = self.output_dir / filename
        logger.info("Generating generative image from prompt: '%s'...", prompt[:60])
        
        # Enhanced culinary styling prompt
        enhanced_prompt = f"professional food photography, appetizing, high resolution, {prompt}, restaurant quality, natural soft lighting"
        encoded_prompt = urllib.parse.quote(enhanced_prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true&seed=42"
        
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            image_data = resp.read()
            
        with open(out_path, "wb") as f:
            f.write(image_data)
            
        logger.info("Saved generated image to %s (%d bytes)", out_path, len(image_data))
        return out_path

