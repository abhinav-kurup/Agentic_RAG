import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.config import Config

logger = logging.getLogger(__name__)

STATUS_INDEXED = "indexed"
STATUS_PROCESSING = "processing"
STATUS_FAILED = "failed"


class DocumentRegistry:
    """Thread-safe document-level index stored as JSON on disk."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or Config.DOCUMENT_REGISTRY_PATH
        self._lock = threading.Lock()
        self._file_locks: Dict[str, threading.Lock] = {}
        self._file_locks_guard = threading.Lock()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not os.path.exists(self.path):
            self._write({})

    def _read(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to read document registry: %s", e)
            return {}

    def _write(self, data: Dict[str, Dict[str, Any]]) -> None:
        temp_path = f"{self.path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(temp_path, self.path)

    @staticmethod
    def _normalize_filename(filename: str) -> str:
        return os.path.basename(filename)

    def file_lock(self, filename: str) -> threading.Lock:
        """Per-filename lock to prevent concurrent ingest of the same document."""
        key = self._normalize_filename(filename)
        with self._file_locks_guard:
            if key not in self._file_locks:
                self._file_locks[key] = threading.Lock()
            return self._file_locks[key]

    def exists(self, filename: str) -> bool:
        key = self._normalize_filename(filename)
        with self._lock:
            entry = self._read().get(key)
            return entry is not None and entry.get("status") == STATUS_INDEXED

    def get(self, filename: str) -> Optional[Dict[str, Any]]:
        key = self._normalize_filename(filename)
        with self._lock:
            return self._read().get(key)

    def list_all(self) -> List[str]:
        with self._lock:
            data = self._read()
            return sorted(
                k for k, v in data.items() if v.get("status") == STATUS_INDEXED
            )

    def list_entries(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self._read())

    def mark_processing(
        self,
        filename: str,
        doc_id: str,
        file_size_bytes: Optional[int] = None,
    ) -> None:
        key = self._normalize_filename(filename)
        with self._lock:
            data = self._read()
            data[key] = {
                "doc_id": doc_id,
                "filename": key,
                "status": STATUS_PROCESSING,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
                "file_size_bytes": file_size_bytes,
            }
            self._write(data)

    def register(
        self,
        filename: str,
        doc_id: str,
        chunk_count: int,
        page_count: int,
        file_size_bytes: Optional[int] = None,
        type_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        key = self._normalize_filename(filename)
        with self._lock:
            data = self._read()
            data[key] = {
                "doc_id": doc_id,
                "filename": key,
                "status": STATUS_INDEXED,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
                "chunk_count": chunk_count,
                "page_count": page_count,
                "file_size_bytes": file_size_bytes,
                "type_counts": type_counts or {},
            }
            self._write(data)
        logger.info("Registered document '%s' (%d chunks).", key, chunk_count)

    def mark_failed(self, filename: str, error: str) -> None:
        key = self._normalize_filename(filename)
        with self._lock:
            data = self._read()
            entry = data.get(key, {"filename": key})
            entry.update(
                {
                    "status": STATUS_FAILED,
                    "error": error,
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            data[key] = entry
            self._write(data)

    def remove(self, filename: str) -> bool:
        key = self._normalize_filename(filename)
        with self._lock:
            data = self._read()
            if key not in data:
                return False
            del data[key]
            self._write(data)
            return True

    def clear(self) -> None:
        with self._lock:
            self._write({})
        logger.info("Document registry cleared.")
