import logging

from fastapi import APIRouter, Request

from core.config import Config
from backend.api.schemas import FeaturesResponse, HealthResponse
from backend.queue.redis_client import check_redis

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    qdrant_status = "unknown"
    try:
        vector_store = request.app.state.vector_store
        count = vector_store._qdrant_chunk_count()
        qdrant_status = f"connected ({count} chunks)"
    except Exception as e:
        logger.error("Qdrant health check failed: %s", e)
        qdrant_status = f"error: {e}"

    redis_ok, redis_status = check_redis()

    overall = "ok" if qdrant_status.startswith("connected") and redis_ok else "degraded"
    return HealthResponse(
        status=overall,
        qdrant=f"{qdrant_status}; redis={redis_status}",
    )


@router.get("/config/features", response_model=FeaturesResponse)
async def features():
    return FeaturesResponse(
        enable_voice=Config.ENABLE_VOICE,
        enable_tts=Config.ENABLE_TTS,
        qdrant_url=Config.QDRANT_URL,
        embedding_model=Config.EMBEDDING_MODEL,
    )
