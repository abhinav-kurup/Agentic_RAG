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
# SpeechT5's hard token limit from tokenizer_config.json
SPEECHT5_MAX_TOKENS = 600
# We use 550 as safe ceiling — leaves room for special tokens
SAFE_TOKEN_LIMIT = 500


# ---------------------------------------------------------------------------
# Bug Fix 1: Number & Symbol Normalizer
# SpeechT5 was trained on natural speech. Digits, symbols, markdown
# map to <unk> tokens which the model silently skips.
# This converts everything to speakable English words BEFORE tokenizing.
# ---------------------------------------------------------------------------

def normalize_for_tts(text: str) -> str:
    """
    Convert numbers, symbols, and markdown into speakable plain English.
    Must run BEFORE passing text to the SpeechT5 processor.
    """

    # --- Markdown / formatting ---
    # Remove bold/italic markers: **word** → word, *word* → word
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    # Remove bullet points at line start
    text = re.sub(r'^\s*[-•*]\s+', '', text, flags=re.MULTILINE)
    # Remove headers: ### Title → Title
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove brackets and URLs
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # [text](url) → text
    text = re.sub(r'https?://\S+', '', text)                 # bare URLs

    # --- Currency (must come before plain number rules) ---
    # $1,234.56 → "1234 dollars and 56 cents"
    def replace_currency(m):
        dollars, cents = (m.group(1).replace(',', ''), m.group(2) if m.group(2) else None)
        result = f"{dollars} dollar{'s' if dollars != '1' else ''}"
        if cents:
            result += f" and {cents} cent{'s' if cents != '1' else ''}"
        return result
    text = re.sub(r'\$(\d[\d,]*)(?:\.(\d{2}))?', replace_currency, text)

    # --- Percentages ---
    # 98.5% → "98 point 5 percent"
    text = re.sub(r'(\d+)\.(\d+)\s*%', r'\1 point \2 percent', text)
    text = re.sub(r'(\d+)\s*%', r'\1 percent', text)

    # --- Decimal numbers (must come before plain integers) ---
    # 398.5 → "398 point 5"
    text = re.sub(r'\b(\d+)\.(\d+)\b', r'\1 point \2', text)

    # --- Large numbers with commas ---
    # 1,000,000 → "1000000" (then num2words handles it below)
    text = re.sub(r'(\d),(\d{3})', r'\1\2', text)
    # Run multiple times for numbers like 1,000,000
    text = re.sub(r'(\d),(\d{3})', r'\1\2', text)

    # --- Convert remaining integers to words ---
    try:
        from num2words import num2words

        def int_to_words(m):
            try:
                return num2words(int(m.group()), lang='en')
            except Exception:
                return m.group()

        text = re.sub(r'\b\d+\b', int_to_words, text)

    except ImportError:
        # num2words not installed — at least remove bare digits that would
        # be skipped as <unk>. Spell out single digits manually.
        digit_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three',
            '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
            '8': 'eight', '9': 'nine'
        }
        # Multi-digit: just space out digits so model can attempt them
        def expand_number(m):
            n = m.group()
            return ' '.join(digit_words.get(d, d) for d in n)

        text = re.sub(r'\b\d+\b', expand_number, text)

    # --- Math operators ---
    text = text.replace('+', ' plus ')
    text = text.replace('=', ' equals ')
    text = text.replace('/', ' divided by ')
    text = text.replace('*', ' times ')
    text = text.replace('(', ' ')
    text = text.replace(')', ' ')

    # --- Remaining symbols that map to <unk> ---
    text = re.sub(r'[_\[\]{}<>|\\^~`]', ' ', text)

    # --- Collapse whitespace ---
    text = re.sub(r'\n+', '. ', text)        # newlines → pause
    text = re.sub(r' +', ' ', text).strip()

    return text


class TTSEngine:
    """Text-to-speech; returns WAV bytes for st.audio in the browser."""

    def __init__(self, speaker_index: int = None):
        speaker_index = (
            speaker_index
            if speaker_index is not None
            else Config.TTS_SPEAKER_INDEX
        )
        logger.info("Loading TTS models (first run may download ~200MB)...")
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

        self.model = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts",
            use_safetensors=True,
        )

        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan",
            use_safetensors=True,
        )

        embeddings_dataset = load_dataset(
            "Matthijs/cmu-arctic-xvectors",
            split="validation",
        )
        self.speaker_embedding = torch.tensor(
            embeddings_dataset[speaker_index]["xvector"]
        ).unsqueeze(0)
        logger.info("TTS models ready (speaker index %s)", speaker_index)

    def synthesize(self, text: str) -> torch.Tensor:
        # Bug Fix 1: normalize numbers/symbols BEFORE chunking
        text = normalize_for_tts(text)
        logger.debug("TTS normalized text: %s", text[:200])

        # Bug Fix 2: chunk on token count with safe ceiling
        chunks = self._chunk_text(text)
        if not chunks:
            return torch.tensor([])

        audio_chunks = []
        for chunk in chunks:
            # Bug Fix 2: pass truncation=True as safety net so the processor
            # never silently passes an oversized tensor to the model
            inputs = self.processor(
                text=chunk,
                return_tensors="pt",
                truncation=True,
                max_length=SAFE_TOKEN_LIMIT,
            )

            with torch.no_grad():
                speech = self.model.generate_speech(
                    inputs["input_ids"],
                    self.speaker_embedding,
                    vocoder=self.vocoder,
                )

            audio_chunks.append(speech)

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

    def _chunk_text(self, text: str) -> list[str]:
        """
        Bug Fix 2: Split text into chunks that are guaranteed to be under
        SAFE_TOKEN_LIMIT tokens. Splits at sentence boundaries first,
        then falls back to word-level splitting if a single sentence
        is still too long.
        """
        # Split at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current = ""

        for sentence in sentences:
            candidate = (current + " " + sentence).strip() if current else sentence.strip()

            token_count = self._count_tokens(candidate)

            if token_count <= SAFE_TOKEN_LIMIT:
                current = candidate
            else:
                # Flush current chunk
                if current:
                    chunks.append(current.strip())

                # Check if this single sentence itself is too long
                if self._count_tokens(sentence) > SAFE_TOKEN_LIMIT:
                    # Fall back to word-level splitting for this sentence
                    word_chunks = self._split_by_words(sentence)
                    # All but last become complete chunks
                    chunks.extend(word_chunks[:-1])
                    current = word_chunks[-1] if word_chunks else ""
                else:
                    current = sentence.strip()

        if current.strip():
            chunks.append(current.strip())

        logger.debug("TTS split into %d chunk(s)", len(chunks))
        return chunks

    def _split_by_words(self, text: str) -> list[str]:
        """Split a single oversized string by words to fit token limit."""
        words = text.split()
        chunks = []
        current = ""

        for word in words:
            candidate = (current + " " + word).strip() if current else word
            if self._count_tokens(candidate) <= SAFE_TOKEN_LIMIT:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                current = word

        if current:
            chunks.append(current.strip())

        return chunks

    def _count_tokens(self, text: str) -> int:
        """Return the token count for a string using the actual processor."""
        return self.processor(
            text=text,
            return_tensors="pt",
        )["input_ids"].shape[1]

    def _smooth_concat(self, chunks: list[torch.Tensor], fade: int = 400) -> torch.Tensor:
        if len(chunks) == 1:
            return chunks[0]

        output = chunks[0]

        for next_chunk in chunks[1:]:
            fade_actual = min(fade, len(output), len(next_chunk))

            fade_out = torch.linspace(1, 0, fade_actual)
            fade_in = torch.linspace(0, 1, fade_actual)

            overlap = (
                output[-fade_actual:] * fade_out
                + next_chunk[:fade_actual] * fade_in
            )

            output = torch.cat([
                output[:-fade_actual],
                overlap,
                next_chunk[fade_actual:],
            ])

        return output