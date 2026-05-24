import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_DB_DIR", "data/chroma")
    MODEL_NAME = os.getenv("LLM_MODEL", "qwen2.5:3b")
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    USE_CROSS_ENCODER = os.getenv("USE_CROSS_ENCODER", "true").lower() in ("1", "true", "yes")
