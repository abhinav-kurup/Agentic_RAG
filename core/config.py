import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_DB_DIR", "data/chroma")
    MODEL_NAME = os.getenv("LLM_MODEL", "qwen2.5:3b")
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    USE_CROSS_ENCODER = os.getenv("USE_CROSS_ENCODER", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    CROSS_ENCODER_MODEL = os.getenv(
        "CROSS_ENCODER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    ENABLE_VOICE = os.getenv("ENABLE_VOICE", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    ENABLE_TTS = os.getenv("ENABLE_TTS", "true").lower() in ("1", "true", "yes")
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
    WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
    WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en") or None
    TTS_SPEAKER_INDEX = int(os.getenv("TTS_SPEAKER_INDEX", "7306"))
