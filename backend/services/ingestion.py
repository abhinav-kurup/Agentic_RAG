import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

from langsmith import traceable

from document_processing.loader import PDFLoader
from document_processing.chunking import DocumentChunker
from vectorstore import VectorStoreManager
from backend.services.document_registry import DocumentRegistry
from backend.services.ingestion_progress import IngestionProgressReporter

logger = logging.getLogger(__name__)


@traceable(name="Ingestion Pipeline")
def ingest_files(
    file_paths: List[str],
    vector_store: VectorStoreManager,
    loader: PDFLoader,
    chunker: DocumentChunker,
    registry: DocumentRegistry,
    reporter: Optional[IngestionProgressReporter] = None,
) -> Dict[str, Any]:
    processed: List[str] = []
    replaced: List[str] = []
    errors: List[Dict[str, str]] = []
    total_chunks_added = 0
    total = len(file_paths)

    for index, file_path in enumerate(file_paths):
        filename = os.path.basename(file_path)
        file_lock = registry.file_lock(filename)

        if not file_lock.acquire(blocking=False):
            msg = "Another ingestion job is already processing this file."
            errors.append({"file": filename, "error": msg})
            if reporter:
                reporter.skip_file(filename, msg)
            continue

        try:
            if reporter:
                reporter.start_file(index, filename)
            result = _ingest_single_file(
                file_path=file_path,
                file_index=index,
                file_total=total,
                vector_store=vector_store,
                loader=loader,
                chunker=chunker,
                registry=registry,
                reporter=reporter,
            )
            processed.append(filename)
            total_chunks_added += result["chunk_count"]
            if result.get("replaced"):
                replaced.append(filename)
            if reporter:
                reporter.complete_file(
                    filename, result["chunk_count"], replaced=result.get("replaced", False)
                )
        except Exception as e:
            logger.exception("Error processing %s", filename)
            registry.mark_failed(filename, str(e))
            errors.append({"file": filename, "error": str(e)})
            if reporter:
                reporter.fail_file(filename, str(e))
        finally:
            file_lock.release()

    return {
        "processed": processed,
        "replaced": replaced,
        "errors": errors,
        "chunks_added": total_chunks_added,
    }


@traceable(name="Ingest Single PDF")
def _ingest_single_file(
    file_path: str,
    file_index: int,
    file_total: int,
    vector_store: VectorStoreManager,
    loader: PDFLoader,
    chunker: DocumentChunker,
    registry: DocumentRegistry,
    reporter: Optional[IngestionProgressReporter] = None,
) -> Dict[str, Any]:
    filename = os.path.basename(file_path)
    doc_id = hashlib.sha256(filename.encode()).hexdigest()[:32]
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else None
    replaced = False
    prefix = f"[{file_index + 1}/{file_total}]"

    if registry.exists(filename) or registry.get(filename):
        if reporter:
            reporter.file_stage(
                filename, "replacing",
                f"{prefix} Replacing existing index for {filename}...",
            )
        deleted = vector_store.delete_by_source(filename)
        registry.remove(filename)
        replaced = True
        if reporter:
            reporter.file_stage(
                filename, "replacing",
                f"{prefix} Removed {deleted} old chunks for {filename}.",
            )

    registry.mark_processing(filename, doc_id, file_size_bytes=file_size)

    if reporter:
        reporter.file_stage(
            filename, "parsing",
            f"{prefix} Running LlamaParse on {filename}...",
        )
    pages = loader.load(file_path)
    if reporter:
        reporter.file_stage(
            filename, "parsed",
            f"{prefix} Loaded {len(pages)} pages from {filename}.",
            pages=len(pages),
        )

    if reporter:
        reporter.file_stage(
            filename, "chunking",
            f"{prefix} Chunking {filename}...",
        )
    chunks = chunker.split_documents(pages, doc_id, source=filename)
    for c in chunks:
        c["metadata"]["source"] = filename

    type_counts: Dict[str, int] = {}
    for c in chunks:
        t = c.get("type", "text")
        type_counts[t] = type_counts.get(t, 0) + 1

    counts_str = ", ".join(f"{count} {t}" for t, count in type_counts.items())
    if reporter:
        reporter.file_stage(
            filename, "chunked",
            f"{prefix} Generated {len(chunks)} chunks ({counts_str}).",
            chunks=len(chunks),
            type_counts=type_counts,
        )

    if chunks:
        if reporter:
            reporter.file_stage(
                filename, "embedding",
                f"{prefix} Embedding & storing {len(chunks)} chunks in Qdrant...",
            )
        vector_store.add_chunks(chunks)

    registry.register(
        filename=filename,
        doc_id=doc_id,
        chunk_count=len(chunks),
        page_count=len(pages),
        file_size_bytes=file_size,
        type_counts=type_counts,
    )

    return {"chunk_count": len(chunks), "replaced": replaced}
