import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from core.config import Config

logger = logging.getLogger(__name__)

BASE_URL = Config.DOCUMIND_API_URL.rstrip("/")
DEFAULT_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class APIError(Exception):
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


def _handle_response(response: httpx.Response) -> Any:
    if response.is_success:
        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()
        return response.content
    detail = response.text
    try:
        detail = response.json().get("detail", detail)
    except Exception:
        pass
    raise APIError(str(detail), response.status_code)


def health_check() -> Dict[str, Any]:
    with httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT) as client:
        return _handle_response(client.get("/health"))


def get_features() -> Dict[str, Any]:
    with httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT) as client:
        return _handle_response(client.get("/config/features"))


def list_documents() -> List[str]:
    with httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT) as client:
        data = _handle_response(client.get("/documents"))
        return data.get("documents", [])


def upload_documents(files: List[tuple]) -> Dict[str, Any]:
    """files: list of (filename, file_bytes) tuples."""
    multipart = [("files", (name, data, "application/pdf")) for name, data in files]
    with httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT) as client:
        return _handle_response(client.post("/documents/upload", files=multipart))


def get_document_job(job_id: str) -> Dict[str, Any]:
    with httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT) as client:
        return _handle_response(client.get(f"/documents/jobs/{job_id}"))


def get_job(job_id: str) -> Dict[str, Any]:
    with httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT) as client:
        # Try documents job first, then voice job
        for prefix in ("/documents/jobs", "/voice/jobs"):
            try:
                resp = client.get(f"{prefix}/{job_id}")
                if resp.status_code == 404:
                    continue
                return _handle_response(resp)
            except APIError:
                continue
        raise APIError("Job not found", 404)


def clear_database() -> Dict[str, Any]:
    with httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT) as client:
        return _handle_response(client.post("/documents/clear"))


def chat_query(query: str, session_id: str, query_id: Optional[str] = None) -> Dict[str, Any]:
    payload = {"query": query, "session_id": session_id}
    if query_id:
        payload["query_id"] = query_id
    with httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT) as client:
        return _handle_response(client.post("/chat/query", json=payload))


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    files = {"file": (filename, audio_bytes, "audio/wav")}
    with httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT) as client:
        data = _handle_response(client.post("/voice/transcribe", files=files))
        return data.get("transcript", "")


def start_tts(text: str, query_id: Optional[str] = None) -> Dict[str, Any]:
    payload = {"text": text}
    if query_id:
        payload["query_id"] = query_id
    with httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT) as client:
        return _handle_response(client.post("/voice/synthesize/async", json=payload))


def get_tts_job(job_id: str) -> Dict[str, Any]:
    with httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT) as client:
        return _handle_response(client.get(f"/voice/jobs/{job_id}"))


def download_tts_audio(job_id: str) -> bytes:
    with httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT) as client:
        resp = client.get(f"/voice/jobs/{job_id}/audio")
        return _handle_response(resp)


def get_audit_logs() -> List[Dict[str, Any]]:
    with httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT) as client:
        data = _handle_response(client.get("/audit/logs"))
        return data.get("logs", [])


def get_metrics() -> Dict[str, Any]:
    with httpx.Client(base_url=BASE_URL, timeout=DEFAULT_TIMEOUT) as client:
        return _handle_response(client.get("/metrics"))


def poll_job(job_id: str, timeout_seconds: int = 300, interval: float = 1.5) -> Dict[str, Any]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        job = get_job(job_id)
        if job.get("status") in ("completed", "failed"):
            return job
        time.sleep(interval)
    raise APIError("Job timed out", 408)
