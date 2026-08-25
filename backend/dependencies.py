import logging
from typing import Optional

from fastapi import Request

from core.config import Config
from core.orchestrator import Orchestrator
from vectorstore import VectorStoreManager
from audit.logger import AuditLogger
from backend.queue.job_store import FileJobStore, get_job_store
from backend.services.document_registry import DocumentRegistry

logger = logging.getLogger(__name__)


class AppState:
    vector_store: Optional[VectorStoreManager] = None
    orchestrator: Optional[Orchestrator] = None
    audit_logger: Optional[AuditLogger] = None
    document_registry: Optional[DocumentRegistry] = None
    stt_engine = None
    tts_engine = None


def get_vector_store(request: Request) -> VectorStoreManager:
    return request.app.state.vector_store


def get_orchestrator(request: Request) -> Orchestrator:
    return request.app.state.orchestrator


def get_audit_logger(request: Request) -> AuditLogger:
    return request.app.state.audit_logger


def get_job_store_dep(_request: Request) -> FileJobStore:
    return get_job_store()


def get_stt_engine(request: Request):
    return request.app.state.stt_engine


def get_tts_engine(request: Request):
    return request.app.state.tts_engine


def get_document_registry(request: Request) -> DocumentRegistry:
    return request.app.state.document_registry


async def init_app_state(app_state: AppState) -> None:
    logger.info("Initializing API services (ingestion runs in separate worker)...")
    app_state.document_registry = DocumentRegistry()
    app_state.vector_store = VectorStoreManager()
    app_state.orchestrator = Orchestrator(vector_store=app_state.vector_store)
    app_state.audit_logger = AuditLogger()

    if Config.USE_CROSS_ENCODER:
        if Config.COHERE_API_KEY:
            logger.info("Rerank: Cohere %s", Config.COHERE_RERANK_MODEL)
        else:
            logger.warning(
                "Rerank enabled but COHERE_API_KEY is not set; hybrid scores will be used."
            )

    if Config.ENABLE_VOICE:
        from agents.stt import STTEngine

        logger.info("Loading STT engine...")
        app_state.stt_engine = STTEngine()

    if Config.ENABLE_TTS:
        from agents.tts import TTSEngine

        logger.info("Loading TTS engine...")
        app_state.tts_engine = TTSEngine()

    logger.info("API services initialized.")

