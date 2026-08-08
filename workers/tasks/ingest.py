import logging
import os
from typing import List

from langsmith import traceable

from backend.queue.job_store import JobStatus, get_job_store
from backend.services.ingestion import ingest_files
from backend.services.ingestion_progress import IngestionProgressReporter
from workers.context import get_worker_context

logger = logging.getLogger(__name__)


@traceable(name="Ingestion Worker Job")
def process_ingestion_job(job_id: str, file_paths: List[str]) -> dict:
    job_store = get_job_store()
    ctx = get_worker_context()
    reporter = IngestionProgressReporter(job_id, file_paths)

    missing = [p for p in file_paths if not os.path.exists(p)]
    if missing:
        msg = f"PDF file(s) not found on disk: {missing}"
        logger.error("Job %s failed: %s", job_id, msg)
        job_store.update(job_id, status=JobStatus.FAILED, error=msg)
        raise FileNotFoundError(msg)

    job_store.update(job_id, status=JobStatus.PROCESSING)
    reporter.set_job_stage("processing", f"Processing {len(file_paths)} PDF(s)...")

    try:
        result = ingest_files(
            file_paths=file_paths,
            vector_store=ctx.vector_store,
            loader=ctx.loader,
            chunker=ctx.chunker,
            registry=ctx.registry,
            reporter=reporter,
        )

        if result.get("errors") and not result.get("processed"):
            error_summary = "; ".join(
                f"{e['file']}: {e['error']}" for e in result["errors"]
            )
            job_store.update(
                job_id,
                status=JobStatus.FAILED,
                result=result,
                error=error_summary,
            )
            raise RuntimeError(error_summary)

        reporter.complete_job()
        job_store.update(
            job_id,
            status=JobStatus.COMPLETED,
            result=result,
        )
        logger.info("Job %s completed: %s", job_id, result)
        return result

    except Exception as e:
        logger.exception("Job %s failed", job_id)
        job_store.update(
            job_id,
            status=JobStatus.FAILED,
            error=str(e),
            progress=f"Job failed: {e}",
        )
        raise
