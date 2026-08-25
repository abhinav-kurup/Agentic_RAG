import io
import logging
import os
import tempfile
from typing import Optional, Union

from faster_whisper import WhisperModel

from core.config import Config
from core.device import whisper_device_and_compute

logger = logging.getLogger(__name__)


class STTEngine:
    """Speech-to-text for browser-recorded audio uploads."""

    def __init__(
        self,
        model_size: str = None,
        device: str = None,
        compute_type: str = None,
        language: Optional[str] = None,
    ):
        model_size = model_size or Config.WHISPER_MODEL
        if device is None or compute_type is None:
            auto_device, auto_compute = whisper_device_and_compute()
            device = device or auto_device
            compute_type = compute_type or auto_compute
        self.language = (
            language if language is not None else Config.WHISPER_LANGUAGE
        )
        logger.info("Loading Whisper model '%s' on %s (%s)", model_size, device, compute_type)
        try:
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
        except Exception:
            if device != "cpu":
                logger.exception("Whisper GPU load failed; falling back to CPU")
                self.model = WhisperModel(
                    model_size,
                    device="cpu",
                    compute_type="int8",
                )
            else:
                raise

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        suffix: str = ".wav",
        initial_prompt: Optional[str] = None,
    ) -> str:
        if not audio_bytes:
            return ""

        try:
            return self._transcribe_input(
                io.BytesIO(audio_bytes),
                initial_prompt=initial_prompt,
            )
        except Exception as e:
            logger.warning("In-memory STT failed (%s); falling back to tempfile", e)

        with tempfile.NamedTemporaryFile(suffix=suffix or ".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            return self._transcribe_input(tmp_path, initial_prompt=initial_prompt)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def transcribe_file(self, path: str, initial_prompt: Optional[str] = None) -> str:
        return self._transcribe_input(path, initial_prompt=initial_prompt)

    def _transcribe_input(
        self,
        audio: Union[str, io.BytesIO],
        initial_prompt: Optional[str] = None,
    ) -> str:
        kwargs = {
            "beam_size": 5,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 250},
            "condition_on_previous_text": False,
        }
        if self.language:
            kwargs["language"] = self.language
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt[:400]

        segments, info = self.model.transcribe(audio, **kwargs)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if not text:
            logger.info(
                "STT empty (no_speech_prob=%s)",
                getattr(info, "no_speech_prob", None),
            )
            return ""
        logger.info("STT produced %d characters: %s", len(text), text[:120])
        return text
