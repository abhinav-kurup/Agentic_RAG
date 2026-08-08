"""Backward-compatible re-exports — use backend.queue.job_store directly."""
from backend.queue.job_store import FileJobStore, Job, JobStatus, get_job_store

JobStore = FileJobStore

__all__ = ["Job", "JobStatus", "JobStore", "FileJobStore", "get_job_store"]
