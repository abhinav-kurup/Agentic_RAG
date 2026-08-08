import copy
import os
from typing import Any, Dict, List, Optional

from backend.queue.job_store import get_job_store

STAGE_FRACTION = {
    "pending": 0.0,
    "replacing": 0.05,
    "parsing": 0.15,
    "parsed": 0.50,
    "chunking": 0.55,
    "chunked": 0.65,
    "embedding": 0.70,
    "indexed": 1.0,
    "failed": 1.0,
    "skipped": 1.0,
}

STAGE_LABELS = {
    "pending": "Waiting",
    "replacing": "Replacing existing index",
    "parsing": "LlamaParse layout parsing",
    "parsed": "Layout parse complete",
    "chunking": "Chunking document",
    "chunked": "Chunks ready",
    "embedding": "Embedding & storing in Qdrant",
    "indexed": "Indexed",
    "failed": "Failed",
    "skipped": "Skipped",
    "queued": "Queued",
    "processing": "Processing",
    "completed": "Completed",
}


class IngestionProgressReporter:
    """Structured per-PDF progress written to the job store for the frontend."""

    def __init__(self, job_id: str, file_paths: List[str]):
        self.job_id = job_id
        self.filenames = [os.path.basename(p) for p in file_paths]
        self.total_files = len(self.filenames)
        self.store = get_job_store()

        existing = self.store.get(job_id)
        if existing and existing.progress_state:
            self._state = copy.deepcopy(existing.progress_state)
            self.completed_files = self._state.get("completed_files", 0)
            self.current_index = self._state.get("current_index", 0)
        else:
            self.completed_files = 0
            self.current_index = 0
            self._state = {
                "total_files": self.total_files,
                "completed_files": 0,
                "current_file": None,
                "current_index": 0,
                "stage": "queued",
                "stage_label": "Queued — waiting for worker",
                "percent": 0,
                "files": [
                    {"name": fn, "status": "pending", "stage": "pending", "chunks": None}
                    for fn in self.filenames
                ],
            }
            self._persist(message="Job created — waiting for worker.")

    def _file_entry(self, filename: str) -> Dict[str, Any]:
        for entry in self._state["files"]:
            if entry["name"] == filename:
                return entry
        return {}

    def _compute_percent(self) -> int:
        if self.total_files == 0:
            return 0
        current_frac = 0.0
        if self.current_index < self.total_files:
            fn = self.filenames[self.current_index]
            entry = self._file_entry(fn)
            stage = entry.get("stage", "pending")
            current_frac = STAGE_FRACTION.get(stage, 0.0)
        overall = (self.completed_files + current_frac) / self.total_files
        return min(100, int(round(overall * 100)))

    def _persist(self, message: Optional[str] = None) -> None:
        self._state["completed_files"] = self.completed_files
        self._state["current_index"] = self.current_index
        self._state["percent"] = self._compute_percent()
        if self.current_index < self.total_files:
            self._state["current_file"] = self.filenames[self.current_index]
        self.store.update(
            self.job_id,
            progress_state=self._state,
            progress=message,
        )

    def set_job_stage(self, stage: str, message: str) -> None:
        self._state["stage"] = stage
        self._state["stage_label"] = message
        self._persist(message=message)

    def start_file(self, index: int, filename: str) -> None:
        self.current_index = index
        entry = self._file_entry(filename)
        entry["status"] = "processing"
        entry["stage"] = "replacing"
        self._state["stage"] = "processing"
        self._state["stage_label"] = f"Processing {filename} ({index + 1}/{self.total_files})"
        self._persist(message=f"[{index + 1}/{self.total_files}] Starting {filename}...")

    def file_stage(self, filename: str, stage: str, message: str, **extra) -> None:
        entry = self._file_entry(filename)
        entry["stage"] = stage
        entry["status"] = "processing"
        for k, v in extra.items():
            entry[k] = v
        self._state["stage"] = stage
        self._state["stage_label"] = STAGE_LABELS.get(stage, stage)
        self._persist(message=message)

    def complete_file(self, filename: str, chunk_count: int, replaced: bool = False) -> None:
        entry = self._file_entry(filename)
        entry["status"] = "done"
        entry["stage"] = "indexed"
        entry["chunks"] = chunk_count
        entry["replaced"] = replaced
        self.completed_files += 1
        action = "Re-indexed" if replaced else "Indexed"
        self._persist(message=f"{action} {filename} ({chunk_count} chunks).")

    def fail_file(self, filename: str, error: str) -> None:
        entry = self._file_entry(filename)
        entry["status"] = "failed"
        entry["stage"] = "failed"
        entry["error"] = error
        self.completed_files += 1
        self._persist(message=f"Failed {filename}: {error}")

    def skip_file(self, filename: str, reason: str) -> None:
        entry = self._file_entry(filename)
        entry["status"] = "skipped"
        entry["stage"] = "skipped"
        entry["error"] = reason
        self.completed_files += 1
        self._persist(message=f"Skipped {filename}: {reason}")

    def complete_job(self) -> None:
        self._state["stage"] = "completed"
        self._state["stage_label"] = "All documents processed"
        self._state["percent"] = 100
        self._state["current_file"] = None
        self._persist(message="Ingestion job completed.")
