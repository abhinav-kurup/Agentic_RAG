import logging
import os
import tempfile
from typing import Optional

from faster_whisper import WhisperModel

from core.config import Config

logger = logging.getLogger(__name__)


class STTEngine:
    """Speech-to-text for Streamlit audio uploads (browser-recorded audio)."""

    def __init__(
        self,
        model_size: str = None,
        device: str = None,
        compute_type: str = None,
        language: Optional[str] = None,
    ):
        model_size = model_size or Config.WHISPER_MODEL
        device = device or Config.WHISPER_DEVICE
        compute_type = compute_type or Config.WHISPER_COMPUTE_TYPE
        self.language = (
            language if language is not None else Config.WHISPER_LANGUAGE
        )
        logger.info("Loading Whisper model '%s' on %s", model_size, device)
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

    def transcribe_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> str:
        """Transcribe raw audio bytes from st.audio_input or file upload."""
        if not audio_bytes:
            return ""

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        try:
            return self.transcribe_file(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def transcribe_file(self, path: str) -> str:
        kwargs = {
            "beam_size": 5,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 500},
        }
        if self.language:
            kwargs["language"] = self.language

        segments, _ = self.model.transcribe(path, **kwargs)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.info("STT produced %d characters", len(text))
        return text
