import io
import re
import logging

import soundfile as sf
import torch
from datasets import load_dataset
from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
)

from core.config import Config

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
SPEECHT5_MAX_TOKENS = 600
# Stay well under the hard cap; 400 gives ~33% headroom.
SAFE_TOKEN_LIMIT = 300

# Pessimistic chars-per-token estimate for SpeechT5's SentencePiece tokenizer.
# 3 chars/token means we only skip the processor for clearly short strings.
_CHARS_PER_TOKEN = 3
_CHAR_LIMIT = SAFE_TOKEN_LIMIT * _CHARS_PER_TOKEN  # ~1200 chars

# Crossfade length in samples at 16 kHz.
# Keep this small so it never consumes a meaningful portion of a short chunk.
# 160 samples = 10 ms — audibly seamless but safe even for very short sentences.
_FADE_SAMPLES = 160

# Silence inserted between chunks (natural sentence pause).
# 3200 samples = 200 ms.
_SILENCE_SAMPLES = 3200


def normalize_for_tts(text: str) -> str:
    """
    Convert numbers, symbols, and markdown into speakable plain English.
    Must run BEFORE passing text to the SpeechT5 processor.
    """
    # --- Strip markdown formatting ---
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    text = re.sub(r'^\s*[-•*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'https?://\S+', '', text)

    # --- Currency ---
    def replace_currency(m):
        dollars = m.group(1).replace(',', '')
        cents = m.group(2)
        result = f"{dollars} dollar{'s' if dollars != '1' else ''}"
        if cents:
            result += f" and {cents} cent{'s' if cents != '1' else ''}"
        return result

    text = re.sub(r'\$(\d[\d,]*)(?:\.(\d{2}))?', replace_currency, text)

    # --- Percentages and decimals (must run before comma stripping) ---
    text = re.sub(r'(\d+)\.(\d+)\s*%', r'\1 point \2 percent', text)
    text = re.sub(r'(\d+)\s*%', r'\1 percent', text)
    text = re.sub(r'\b(\d+)\.(\d+)\b', r'\1 point \2', text)

    # --- Strip thousand-separator commas (two passes handles up to 9-digit numbers) ---
    text = re.sub(r'(\d),(\d{3})(?=\D|$)', r'\1\2', text)
    text = re.sub(r'(\d),(\d{3})(?=\D|$)', r'\1\2', text)

    # --- Integers to words ---
    try:
        from num2words import num2words

        def int_to_words(m):
            try:
                return num2words(int(m.group()), lang='en')
            except Exception:
                return m.group()

        text = re.sub(r'\b\d+\b', int_to_words, text)

    except ImportError:
        digit_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three',
            '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
            '8': 'eight', '9': 'nine',
        }

        def expand_number(m):
            return ' '.join(digit_words.get(d, d) for d in m.group())

        text = re.sub(r'\b\d+\b', expand_number, text)

    # --- Math / special symbols ---
    text = text.replace('+', ' plus ')
    text = text.replace('=', ' equals ')
    text = text.replace('/', ' divided by ')
    text = text.replace('*', ' times ')
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')
    text = re.sub(r'[_\[\]{}<>|\\^~`]', ' ', text)

    # --- Whitespace cleanup ---
    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r' +', ' ', text).strip()

    return text


class TTSEngine:
    """Text-to-speech engine; returns WAV bytes for st.audio in the browser."""

    def __init__(self, speaker_index: int = None):
        speaker_index = (
            speaker_index
            if speaker_index is not None
            else Config.TTS_SPEAKER_INDEX
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        logger.info("TTS using device: %s", self.device)
        logger.info("Loading TTS models (first run may download ~200 MB)...")

        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

        self.model = (
            SpeechT5ForTextToSpeech
            .from_pretrained("microsoft/speecht5_tts", use_safetensors=True)
            .to(self.device)
            .eval()
        )

        self.vocoder = (
            SpeechT5HifiGan
            .from_pretrained("microsoft/speecht5_hifigan", use_safetensors=True)
            .to(self.device)
            .eval()
        )

        embeddings_dataset = load_dataset(
            "Matthijs/cmu-arctic-xvectors",
            split="validation",
        )
        self.speaker_embedding = (
            torch.tensor(embeddings_dataset[speaker_index]["xvector"])
            .unsqueeze(0)
            .to(self.device)
        )
        logger.info("TTS models ready (speaker index %s)", speaker_index)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, text: str) -> torch.Tensor:
        text = normalize_for_tts(text)
        logger.debug("TTS normalized text (first 300 chars): %s", text[:300])

        chunks = self._chunk_text(text)
        if not chunks:
            return torch.tensor([])

        # Log every chunk so skipping problems are immediately visible in logs
        for i, chunk in enumerate(chunks):
            token_count = self._count_tokens(chunk)
            logger.debug("Chunk %d/%d (%d tokens): %s", i + 1, len(chunks), token_count, chunk[:120])

        audio_chunks = []
        for i, chunk in enumerate(chunks):
            # Validate token count BEFORE sending to model — truncation=False
            # means the processor will raise instead of silently dropping words.
            token_count = self._count_tokens(chunk)
            if token_count > SPEECHT5_MAX_TOKENS:
                # Should never happen given _fits(), but log and skip rather
                # than letting the model produce corrupted/truncated audio.
                logger.error(
                    "Chunk %d exceeds hard token limit (%d > %d); skipping: %s",
                    i + 1, token_count, SPEECHT5_MAX_TOKENS, chunk[:80],
                )
                continue

            inputs = self.processor(
                text=chunk,
                return_tensors="pt",
                truncation=False,       # CHANGED: never silently drop words
            )
            input_ids = inputs["input_ids"].to(self.device)

            with torch.no_grad():
                speech = self.model.generate_speech(
                    input_ids,
                    self.speaker_embedding,
                    vocoder=self.vocoder,
                )

            audio_chunks.append(speech.cpu())
            logger.debug("Chunk %d synthesized: %d samples", i + 1, len(speech))

        if not audio_chunks:
            return torch.tensor([])

        return self._smooth_concat(audio_chunks)

    def synthesize_to_wav_bytes(self, text: str) -> bytes:
        """WAV bytes suitable for st.audio(format='audio/wav')."""
        text = (text or "").strip()
        if not text:
            return b""

        audio = self.synthesize(text)
        if audio.numel() == 0:
            return b""

        buffer = io.BytesIO()
        sf.write(buffer, audio.numpy(), samplerate=SAMPLE_RATE, format="WAV")
        logger.info("TTS synthesized %d bytes of audio", buffer.tell())
        return buffer.getvalue()

    # ------------------------------------------------------------------
    # Chunking helpers
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into chunks guaranteed to stay under SAFE_TOKEN_LIMIT.

        Strategy
        --------
        1. Split on sentence boundaries (. ! ?).
        2. Greedily accumulate sentences into a chunk as long as they fit.
        3. When a single sentence is too long on its own, fall back to
           word-level splitting for that sentence only.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            candidate = f"{current} {sentence}".strip() if current else sentence

            if self._fits(candidate):
                current = candidate
            else:
                if current:
                    chunks.append(current)
                    current = ""

                if self._fits(sentence):
                    current = sentence
                else:
                    # Sentence too long alone — split at word level
                    word_chunks = self._split_by_words(sentence)
                    if len(word_chunks) > 1:
                        chunks.extend(word_chunks[:-1])
                    current = word_chunks[-1] if word_chunks else ""

        if current.strip():
            chunks.append(current.strip())

        logger.debug("TTS split into %d chunk(s)", len(chunks))
        return chunks

    def _split_by_words(self, text: str) -> list[str]:
        """Split text at word boundaries so every chunk fits the token limit."""
        words = text.split()
        chunks: list[str] = []
        current = ""

        for word in words:
            candidate = f"{current} {word}".strip() if current else word
            if self._fits(candidate):
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = word

        if current:
            chunks.append(current)

        return chunks

    def _fits(self, text: str) -> bool:
        """
        Two-stage token check.

        Stage 1 (cheap): only skip the processor when text is clearly
        short — under 50% of the char limit (pessimistic cutoff).

        Stage 2 (exact): processor token count for anything longer.
        """
        if len(text) < _CHAR_LIMIT * 0.5:
            return True
        if len(text) > _CHAR_LIMIT * 2.0:
            return False
        return self._count_tokens(text) <= SAFE_TOKEN_LIMIT

    def _count_tokens(self, text: str) -> int:
        """Exact token count via the processor."""
        return self.processor(text=text, return_tensors="pt")["input_ids"].shape[1]

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    def _smooth_concat(self, chunks: list[torch.Tensor]) -> torch.Tensor:
        """
        Concatenate audio chunks with a short crossfade and silence gap.

        Key fix for word skipping
        -------------------------
        The previous fade=400 samples was applied even to very short chunks,
        where min(fade, len(chunk)) could consume most of the chunk's audio.
        _FADE_SAMPLES is now 160 (10 ms) — inaudible as a click but small
        enough to never eat meaningful speech from a short sentence.

        We also guard against degenerate chunks shorter than 2x the fade
        length; those get a plain concatenation with just silence, no fade.
        """
        if not chunks:
            return torch.tensor([])
        if len(chunks) == 1:
            return chunks[0]

        silence = torch.zeros(_SILENCE_SAMPLES)
        output = chunks[0]

        for next_chunk in chunks[1:]:
            # Only crossfade if both tensors are long enough for it to make
            # sense; otherwise just join with silence to avoid corrupting audio.
            min_len = min(len(output), len(next_chunk))
            if min_len < _FADE_SAMPLES * 2:
                output = torch.cat([output, silence, next_chunk])
                continue

            fade_out = torch.linspace(1.0, 0.0, _FADE_SAMPLES)
            fade_in  = torch.linspace(0.0, 1.0, _FADE_SAMPLES)

            overlap = (
                output[-_FADE_SAMPLES:] * fade_out
                + next_chunk[:_FADE_SAMPLES] * fade_in
            )

            output = torch.cat([
                output[:-_FADE_SAMPLES],
                overlap,
                silence,
                next_chunk[_FADE_SAMPLES:],
            ])

        return output