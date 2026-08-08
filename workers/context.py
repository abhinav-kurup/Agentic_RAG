import logging
import threading
from typing import Optional

from document_processing.loader import PDFLoader
from document_processing.chunking import DocumentChunker
from vectorstore import VectorStoreManager
from backend.services.document_registry import DocumentRegistry

logger = logging.getLogger(__name__)


class WorkerContext:
    """
    Lazy singletons for the ingest worker process.
    Heavy models (embedders) load once on first job, not at import time.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._vector_store: Optional[VectorStoreManager] = None
        self._loader: Optional[PDFLoader] = None
        self._chunker: Optional[DocumentChunker] = None
        self._registry: Optional[DocumentRegistry] = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            logger.info("Worker: initializing ingestion dependencies...")
            self._registry = DocumentRegistry()
            self._vector_store = VectorStoreManager()
            self._loader = PDFLoader()
            self._chunker = DocumentChunker()
            self._initialized = True
            logger.info("Worker: ingestion dependencies ready.")

    @property
    def vector_store(self) -> VectorStoreManager:
        self._ensure_initialized()
        return self._vector_store

    @property
    def loader(self) -> PDFLoader:
        self._ensure_initialized()
        return self._loader

    @property
    def chunker(self) -> DocumentChunker:
        self._ensure_initialized()
        return self._chunker

    @property
    def registry(self) -> DocumentRegistry:
        self._ensure_initialized()
        return self._registry


_context: Optional[WorkerContext] = None
_context_lock = threading.Lock()


def get_worker_context() -> WorkerContext:
    global _context
    with _context_lock:
        if _context is None:
            _context = WorkerContext()
        return _context
