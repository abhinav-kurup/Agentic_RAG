import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import Config
from backend.dependencies import AppState, init_app_state
from backend.api.routes import health, documents, chat, voice, audit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(Config.DOCUMENTS_DIR, exist_ok=True)
    os.makedirs(Config.AUDIO_DIR, exist_ok=True)
    os.makedirs(Config.JOBS_DIR, exist_ok=True)
    os.makedirs("data/logs", exist_ok=True)

    state = AppState()
    await init_app_state(state)

    app.state.vector_store = state.vector_store
    app.state.orchestrator = state.orchestrator
    app.state.audit_logger = state.audit_logger
    app.state.document_registry = state.document_registry
    app.state.stt_engine = state.stt_engine
    app.state.tts_engine = state.tts_engine

    logger.info("DocuMind API ready on %s:%s", Config.API_HOST, Config.API_PORT)
    logger.info("Ingestion jobs require Redis + worker: python workers/run_worker.py")
    yield


app = FastAPI(
    title="DocuMind API",
    description="FastAPI backend for DocuMind RAG platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(voice.router)
app.include_router(audit.router)
