"""Unit tests for multimodal sentiment analysis, speech synthesis, and media generation."""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import io
from PIL import Image

from src.multimodal.multimodal_service import MultimodalSentimentService


class TestMultimodalService:
    """Test multimodal vision analysis, gTTS synthesis, and image generation."""

    def test_synthesize_review_speech(self, tmp_path):
        service = MultimodalSentimentService(media_output_dir=tmp_path)
        text = "The pizza crust was crisp and the sauce was delightfully flavorful."
        audio_path = service.synthesize_review_speech(text, filename="test_audio.mp3")
        assert audio_path.exists()
        assert audio_path.stat().st_size > 5000  # valid MP3 binary

    def test_analyze_image_and_review_mocked(self, tmp_path):
        mock_provider = MagicMock()
        mock_client = MagicMock()
        mock_provider._client = mock_client
        mock_provider.model_name = "gemini-2.5-flash"
        mock_provider._clean_json_text.return_value = '{"visual_description": "Grilled burger", "text_sentiment": "Positive", "visual_sentiment": "Positive", "overall_joint_sentiment": "Positive", "confidence": 0.95, "explanation": "Delicious presentation"}'
        
        service = MultimodalSentimentService(gemini_provider=mock_provider, media_output_dir=tmp_path)
        img = Image.new("RGB", (64, 64), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        
        result = service.analyze_image_and_review(buf.getvalue(), "Amazing burger!")
        assert result["overall_joint_sentiment"] == "Positive"
        assert result["confidence"] == 0.95
        assert "Grilled burger" in result["visual_description"]

    def test_generate_dish_image_mocked(self, tmp_path):
        service = MultimodalSentimentService(media_output_dir=tmp_path)
        # Mock urllib request to return fake image bytes without internet in unit tests
        mock_bytes = b"\xff\xd8\xff\xe0fake_jpeg_data"
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_bytes
            mock_urlopen.return_value.__enter__.return_value = mock_resp
            
            img_path = service.generate_dish_image("artisanal pasta with truffle oil", filename="test_pasta.jpg")
            assert img_path.exists()
            assert img_path.read_bytes() == mock_bytes

