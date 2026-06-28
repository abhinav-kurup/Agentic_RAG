import os
from dotenv import load_dotenv

load_dotenv(override=True)


class Config:
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_DB_DIR", "data/chroma")
    MODEL_NAME = os.getenv("LLM_MODEL", "qwen2.5:3b")
    PLANNER_MODEL = os.getenv("PLANNER_MODEL", "groq/llama-3.3-70b-versatile")
    RETRIEVAL_MODEL = os.getenv("RETRIEVAL_MODEL", "groq/llama-3.1-8b-instant")
    EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "gemini/gemini-2.0-flash-lite")
    ANALYSIS_MODEL = os.getenv("ANALYSIS_MODEL", "gemini/gemini-2.0-flash")
    RAGAS_EVAL_MODEL = os.getenv("RAGAS_EVAL_MODEL", "groq/llama-3.3-70b-versatile")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    USE_CROSS_ENCODER = os.getenv("USE_CROSS_ENCODER", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    CROSS_ENCODER_MODEL = os.getenv(
        "CROSS_ENCODER_MODEL",
        "BAAI/bge-reranker-v2-m3",
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
