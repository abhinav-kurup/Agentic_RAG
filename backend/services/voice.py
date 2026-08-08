import logging
import os
import threading
from typing import Optional

from core.config import Config
from backend.services.jobs import JobStatus, JobStore

logger = logging.getLogger(__name__)


def transcribe_audio(stt_engine, audio_bytes: bytes) -> str:
    if not audio_bytes:
        return ""
    return stt_engine.transcribe_bytes(audio_bytes)


def synthesize_audio(tts_engine, text: str) -> bytes:
    return tts_engine.synthesize_to_wav_bytes(text)


def start_tts_job(
    job_store: JobStore,
    tts_engine,
    text: str,
    query_id: Optional[str] = None,
) -> str:
    job = job_store.create("tts")
    os.makedirs(Config.AUDIO_DIR, exist_ok=True)
    audio_path = os.path.join(Config.AUDIO_DIR, f"{job.job_id}.wav")

    def _run():
        job_store.update(job.job_id, status=JobStatus.PROCESSING)
        try:
            audio_bytes = synthesize_audio(tts_engine, text)
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
            job_store.update(
                job.job_id,
                status=JobStatus.COMPLETED,
                result={
                    "query_id": query_id,
                    "audio_path": audio_path,
                    "size_bytes": len(audio_bytes),
                },
            )
        except Exception as e:
            logger.exception("TTS job failed")
            job_store.update(job.job_id, status=JobStatus.FAILED, error=str(e))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return job.job_id
