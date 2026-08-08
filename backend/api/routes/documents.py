import logging
import os
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from core.config import Config
from backend.api.schemas import (
    ClearDatabaseResponse,
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentUploadResponse,
    JobResponse,
)
from backend.dependencies import get_document_registry, get_vector_store
from backend.queue.enqueue import enqueue_ingestion_job
from backend.queue.job_store import get_job_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    os.makedirs(Config.DOCUMENTS_DIR, exist_ok=True)
    saved_paths = []

    for upload in files:
        if not upload.filename or not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file: {upload.filename}. Only PDF files are supported.",
            )
        dest = os.path.join(Config.DOCUMENTS_DIR, upload.filename)
        content = await upload.read()
        with open(dest, "wb") as f:
            f.write(content)
        saved_paths.append(dest)

    try:
        job_id = enqueue_ingestion_job(saved_paths)
    except ConnectionError as e:
        logger.error("Redis unavailable: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Job queue unavailable. Ensure Redis is running and REDIS_URL is correct.",
        ) from e
    except Exception as e:
        logger.exception("Failed to enqueue ingestion job")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to enqueue ingestion job: {e}",
        ) from e

    return DocumentUploadResponse(
        job_id=job_id,
        status="queued",
        message=f"Queued {len(saved_paths)} file(s). Poll GET /documents/jobs/{job_id}",
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_ingestion_job(job_id: str):
    job = get_job_store().get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobResponse(**job.to_dict())


@router.get("", response_model=DocumentListResponse)
async def list_documents(registry=Depends(get_document_registry)):
    return DocumentListResponse(documents=registry.list_all())


@router.delete("/{source}", response_model=DeleteDocumentResponse)
async def delete_document(
    source: str,
    vector_store=Depends(get_vector_store),
    registry=Depends(get_document_registry),
):
    deleted = vector_store.delete_by_source(source)
    registry.remove(source)
    return DeleteDocumentResponse(source=source, deleted_count=deleted)


@router.post("/clear", response_model=ClearDatabaseResponse)
async def clear_database(
    vector_store=Depends(get_vector_store),
    registry=Depends(get_document_registry),
):
    vector_store.clear_database()
    registry.clear()
    return ClearDatabaseResponse(message="Database cleared successfully.")
