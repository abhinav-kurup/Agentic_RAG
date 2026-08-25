import io
import logging
import re
import threading
import wave
from pathlib import Path
from typing import List, Optional
from urllib.request import urlopen

import numpy as np

from core.config import Config
from core.device import cuda_available

logger = logging.getLogger(__name__)

_SILENCE_SECONDS = 0.2
_MAX_UTTERANCE_CHARS = 400
_VOICE_RE = re.compile(
    r"^(?P<lang_family>[^-]+)_(?P<lang_region>[^-]+)-(?P<voice_name>[^-]+)-(?P<voice_quality>.+)$"
)
_VOICE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "{lang_family}/{lang_code}/{voice_name}/{voice_quality}/"
    "{lang_code}-{voice_name}-{voice_quality}{ext}?download=true"
)


def to_speakable_text(text: str) -> str:
    """Turn a RAG answer into plain speech without mangling ratios or paths."""
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"^\s*[-•*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\b(\d+)\s*/\s*(\d+)\b", r"\1 of \2", text)

    def replace_currency(m):
        dollars = m.group(1).replace(",", "")
        cents = m.group(2)
        result = f"{dollars} dollar{'s' if dollars != '1' else ''}"
        if cents:
            result += f" and {cents} cent{'s' if cents != '1' else ''}"
        return result

    text = re.sub(r"\$(\d[\d,]*)(?:\.(\d{2}))?", replace_currency, text)
    text = re.sub(r"(\d+)\.(\d+)\s*%", r"\1 point \2 percent", text)
    text = re.sub(r"(\d+)\s*%", r"\1 percent", text)
    text = re.sub(r"\b(\d+)\.(\d+)\b", r"\1 point \2", text)
    text = re.sub(r"(\d),(\d{3})(?=\D|$)", r"\1\2", text)
    text = re.sub(r"(\d),(\d{3})(?=\D|$)", r"\1\2", text)

    try:
        from num2words import num2words

        def int_to_words(m):
            try:
                return num2words(int(m.group()), lang="en")
            except Exception:
                return m.group()

        text = re.sub(r"\b\d+\b", int_to_words, text)
    except ImportError:
        pass

    text = re.sub(r"[_\[\]{}<>|\\^~`]", " ", text)
    text = re.sub(r"\n+", ". ", text)
    text = re.sub(r"\.{2,}", ". ", text)
    text = re.sub(r" +", " ", text).strip()
    return text


def split_utterances(text: str) -> List[str]:
    """One Piper call per sentence. Never pack multiple sentences together."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    utterances: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= _MAX_UTTERANCE_CHARS:
            utterances.append(part)
        else:
            utterances.extend(_split_long(part))
    return utterances


def _split_long(text: str) -> List[str]:
    pieces = re.split(r"(?<=[,;:])\s+", text)
    chunks: List[str] = []
    current = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= _MAX_UTTERANCE_CHARS:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(piece) <= _MAX_UTTERANCE_CHARS:
            current = piece
        else:
            chunks.extend(_split_by_words(piece))
            current = ""
    if current:
        chunks.append(current)
    return chunks


def _split_by_words(text: str) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        if len(candidate) <= _MAX_UTTERANCE_CHARS:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return chunks


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urlopen(url) as response, open(tmp, "wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)


def ensure_piper_voice(voice: str, download_dir: Path) -> Path:
    model_path = download_dir / f"{voice}.onnx"
    config_path = download_dir / f"{voice}.onnx.json"
    if model_path.exists() and model_path.stat().st_size > 0 and config_path.exists():
        return model_path

    download_dir.mkdir(parents=True, exist_ok=True)
    try:
        from piper.download_voices import download_voice

        download_voice(voice, download_dir)
        return model_path
    except Exception as e:
        logger.warning("piper.download_voices failed (%s); using direct download", e)

    match = _VOICE_RE.match(voice.strip())
    if not match:
        raise ValueError(
            f"Voice '{voice}' must look like 'en_US-lessac-medium'"
        )
    lang_family = match.group("lang_family")
    lang_code = f"{lang_family}_{match.group('lang_region')}"
    args = {
        "lang_family": lang_family,
        "lang_code": lang_code,
        "voice_name": match.group("voice_name"),
        "voice_quality": match.group("voice_quality"),
    }
    if not model_path.exists() or model_path.stat().st_size == 0:
        logger.info("Downloading Piper voice %s", voice)
        _download_file(_VOICE_URL.format(ext=".onnx", **args), model_path)
    if not config_path.exists() or config_path.stat().st_size == 0:
        _download_file(_VOICE_URL.format(ext=".onnx.json", **args), config_path)
    return model_path


class TTSEngine:
    """Piper TTS. One sentence per synthesis call, silence between utterances."""

    def __init__(self, voice_name: Optional[str] = None):
        voice_name = voice_name or Config.PIPER_VOICE
        download_dir = Path(Config.PIPER_VOICE_DIR)
        logger.info("Loading Piper voice '%s'", voice_name)
        model_path = ensure_piper_voice(voice_name, download_dir)

        from piper import PiperVoice

        use_cuda = cuda_available()
        try:
            self.voice = PiperVoice.load(str(model_path), use_cuda=use_cuda)
        except Exception:
            if use_cuda:
                logger.exception("Piper CUDA load failed; using CPU")
                self.voice = PiperVoice.load(str(model_path))
                use_cuda = False
            else:
                raise
        cfg = getattr(self.voice, "config", None)
        sample_rate = getattr(cfg, "sample_rate", None)
        if sample_rate is None and isinstance(cfg, dict):
            sample_rate = (cfg.get("audio") or {}).get("sample_rate")
        self.sample_rate = int(sample_rate or 22050)
        self._lock = threading.Lock()
        logger.info("Piper ready (%s Hz, cuda=%s)", self.sample_rate, use_cuda)

    def synthesize_to_wav_bytes(self, text: str) -> bytes:
        text = (text or "").strip()
        if not text:
            return b""
        if len(text) > Config.TTS_MAX_CHARS:
            text = text[: Config.TTS_MAX_CHARS]

        speakable = to_speakable_text(text)
        utterances = split_utterances(speakable)
        if not utterances:
            return b""

        with self._lock:
            pieces: List[np.ndarray] = []
            sample_rate = self.sample_rate
            for i, utterance in enumerate(utterances):
                audio, sr = self._synth_utterance(utterance)
                if sr:
                    sample_rate = sr
                if audio.size == 0:
                    logger.error("Piper produced no audio for utterance %d: %s", i + 1, utterance[:80])
                    continue
                pieces.append(audio)
                logger.debug("Utterance %d/%d: %d samples", i + 1, len(utterances), audio.size)

        if not pieces:
            return b""

        silence = np.zeros(int(sample_rate * _SILENCE_SECONDS), dtype=np.int16)
        joined: List[np.ndarray] = []
        for i, piece in enumerate(pieces):
            if i:
                joined.append(silence)
            joined.append(piece)
        audio = np.concatenate(joined)
        wav_bytes = _pcm16_to_wav_bytes(audio, sample_rate)
        logger.info("TTS synthesized %d bytes (%d utterances)", len(wav_bytes), len(pieces))
        return wav_bytes

    def _synth_utterance(self, text: str) -> tuple:
        audio, sr = self._synth_once(text)
        if audio.size:
            return audio, sr
        logger.warning("Retrying Piper utterance: %s", text[:80])
        return self._synth_once(text)

    def _synth_once(self, text: str) -> tuple:
        buf = io.BytesIO()
        try:
            with wave.open(buf, "wb") as wav_file:
                if hasattr(self.voice, "synthesize_wav"):
                    self.voice.synthesize_wav(text, wav_file)
                else:
                    self.voice.synthesize(text, wav_file)
        except Exception:
            logger.exception("Piper synthesis failed: %s", text[:80])
            return np.array([], dtype=np.int16), self.sample_rate

        buf.seek(0)
        try:
            with wave.open(buf, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).copy()
            return audio, sample_rate
        except Exception:
            logger.exception("Failed to read Piper WAV for: %s", text[:80])
            return np.array([], dtype=np.int16), self.sample_rate


def _pcm16_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.astype(np.int16).tobytes())
    return buf.getvalue()
