import logging
import os
from typing import List

import redis
from langsmith import traceable

from backend.queue.job_store import JobStatus, get_job_store
from backend.queue.redis_client import get_ingest_queue
from backend.services.ingestion_progress import IngestionProgressReporter
from core.config import Config

logger = logging.getLogger(__name__)

INGEST_TASK_PATH = "workers.tasks.ingest.process_ingestion_job"


@traceable(name="Enqueue Ingestion Job")
def enqueue_ingestion_job(file_paths: List[str]) -> str:
    if not file_paths:
        raise ValueError("No files to enqueue.")

    job_store = get_job_store()
    job = job_store.create("ingestion", file_paths=file_paths)

    # Initialize structured progress so the UI can show file list immediately
    reporter = IngestionProgressReporter(job.job_id, file_paths)
    reporter.set_job_stage("queued", f"Queued {len(file_paths)} file(s) — waiting for worker")

    try:
        queue = get_ingest_queue()
        rq_job = queue.enqueue(
            INGEST_TASK_PATH,
            job.job_id,
            file_paths,
            job_timeout=Config.INGEST_JOB_TIMEOUT_SECONDS,
            failure_ttl=86400,
            result_ttl=86400,
            description=f"Ingest {len(file_paths)} PDF(s) — job {job.job_id}",
        )
    except redis.RedisError as e:
        job_store.update(job.job_id, status=JobStatus.FAILED, error=f"Redis error: {e}")
        raise ConnectionError(f"Redis unavailable: {e}") from e

    job_store.update(
        job.job_id,
        status=JobStatus.QUEUED,
        rq_job_id=rq_job.id,
        progress=f"Queued for worker ({len(file_paths)} file(s)).",
    )

    logger.info(
        "Enqueued ingestion job %s → RQ %s (%d files, queue depth=%d)",
        job.job_id,
        rq_job.id,
        len(file_paths),
        queue.count,
    )
    return job.job_id