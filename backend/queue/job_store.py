import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from core.config import Config

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    job_type: str
    status: JobStatus = JobStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    progress: List[str] = field(default_factory=list)
    progress_state: Optional[Dict[str, Any]] = None
    file_paths: List[str] = field(default_factory=list)
    rq_job_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "progress": self.progress,
            "progress_state": self.progress_state,
            "file_paths": self.file_paths,
            "rq_job_id": self.rq_job_id,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        status = data.get("status", JobStatus.PENDING.value)
        if isinstance(status, str):
            status = JobStatus(status)
        return cls(
            job_id=data["job_id"],
            job_type=data["job_type"],
            status=status,
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
            progress=data.get("progress") or [],
            progress_state=data.get("progress_state"),
            file_paths=data.get("file_paths") or [],
            rq_job_id=data.get("rq_job_id"),
            result=data.get("result"),
            error=data.get("error"),
        )


class FileJobStore:
    """
    Persistent job store — one JSON file per job under data/jobs/.
    Shared by API (create/enqueue) and ingest worker (update progress).
    Easy to inspect manually when debugging.
    """

    def __init__(self, jobs_dir: Optional[str] = None):
        self.jobs_dir = jobs_dir or Config.JOBS_DIR
        os.makedirs(self.jobs_dir, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, job_id: str) -> str:
        return os.path.join(self.jobs_dir, f"{job_id}.json")

    def _read_file(self, job_id: str) -> Optional[Job]:
        path = self._path(job_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return Job.from_dict(json.load(f))
        except (json.JSONDecodeError, OSError, KeyError, ValueError) as e:
            logger.error("Failed to read job file %s: %s", path, e)
            return None

    def _write_file(self, job: Job) -> None:
        path = self._path(job.job_id)
        temp = f"{path}.tmp"
        job.updated_at = datetime.now(timezone.utc).isoformat()
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(job.to_dict(), f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(temp, path)

    def create(self, job_type: str, file_paths: Optional[List[str]] = None) -> Job:
        job = Job(
            job_id=str(uuid.uuid4()),
            job_type=job_type,
            file_paths=file_paths or [],
        )
        with self._lock:
            self._write_file(job)
        logger.info("Created job %s (type=%s, files=%d)", job.job_id, job_type, len(job.file_paths))
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._read_file(job_id)

    def update(
        self,
        job_id: str,
        *,
        status: Optional[JobStatus] = None,
        progress: Optional[str] = None,
        progress_state: Optional[Dict[str, Any]] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        rq_job_id: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
    ) -> Optional[Job]:
        with self._lock:
            job = self._read_file(job_id)
            if not job:
                return None
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress.append(progress)
            if progress_state is not None:
                job.progress_state = progress_state
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            if rq_job_id is not None:
                job.rq_job_id = rq_job_id
            if file_paths is not None:
                job.file_paths = file_paths
            self._write_file(job)
            return job


# Process-wide singleton — API and worker share the same on-disk store.
_store: Optional[FileJobStore] = None
_store_lock = threading.Lock()


def get_job_store() -> FileJobStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = FileJobStore()
        return _store
