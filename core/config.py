import os
from dotenv import load_dotenv

load_dotenv(override=True)


class Config:
    PARSER_TYPE = os.getenv("PARSER_TYPE", "pymupdf")  # 'pymupdf' or 'llama_parse'
    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "documind_collection")
    VISION_MODEL = os.getenv("VISION_MODEL", "gemini/gemini-2.0-flash-lite")
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

    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    DOCUMIND_API_URL = os.getenv("DOCUMIND_API_URL", "http://localhost:8000")
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
    DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "data/documents")
    AUDIO_DIR = os.getenv("AUDIO_DIR", "data/audio")
    DOCUMENT_REGISTRY_PATH = os.getenv(
        "DOCUMENT_REGISTRY_PATH", "data/documents_registry.json"
    )

    # Ingestion worker queue (Redis + RQ)
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6380/0")
    INGEST_QUEUE_NAME = os.getenv("INGEST_QUEUE_NAME", "documind:ingest")
    JOBS_DIR = os.getenv("JOBS_DIR", "data/jobs")
    INGEST_JOB_TIMEOUT_SECONDS = int(os.getenv("INGEST_JOB_TIMEOUT_SECONDS", "3600"))

    # Layout parsing — disable Gemini vision / image blocks when false
    ENABLE_IMAGE_PROCESSING = os.getenv("ENABLE_IMAGE_PROCESSING", "false").lower() in (
        "1",
        "true",
        "yes",
    )
