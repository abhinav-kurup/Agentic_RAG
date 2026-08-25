import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

from agents.voice_intent import get_voice_router
from core.config import Config
from backend.services.jobs import JobStatus, JobStore

logger = logging.getLogger(__name__)

ASK_ACK = "Your answer is on the screen."
CLARIFY_ACK = "Could you repeat that?"


async def interpret_audio(
    stt_engine,
    orchestrator,
    audio_bytes: bytes,
    suffix: str,
    session_id: str,
    awaiting_confirm: bool,
    initial_prompt: Optional[str] = None,
    has_last_answer: bool = False,
) -> dict:
    transcript = await asyncio.to_thread(
        transcribe_audio,
        stt_engine,
        audio_bytes,
        initial_prompt,
        suffix,
    )
    transcript = (transcript or "").strip()
    heard = transcript if transcript else "(empty — STT heard nothing)"
    print(
        f"\n[voice] heard: {heard}\n"
        f"[voice] audio bytes: {len(audio_bytes)}\n",
        flush=True,
    )
    ctx = orchestrator.session_context(session_id)
    routed = await get_voice_router().aroute(
        transcript,
        history_text=ctx.get("history_text") or "",
        has_last_answer=bool(has_last_answer or ctx.get("has_answer")),
    )
    intent = routed.get("intent") or "ask"
    reconstructed = transcript
    spoken_ack = ""
    print(f"[voice] intent: {intent}", flush=True)
    logger.info("Voice intent=%s transcript=%r", intent, transcript[:160])

    if intent == "ask":
        reconstructed = (routed.get("query") or "").strip()
        if not reconstructed:
            reconstructed = await orchestrator.reconstruct_query(transcript, session_id)
        spoken_ack = ASK_ACK
    elif intent == "clarify":
        spoken_ack = CLARIFY_ACK
    elif intent == "unheard":
        spoken_ack = ""

    return {
        "transcript": transcript,
        "reconstructed_query": reconstructed,
        "intent": intent,
        "spoken_ack": spoken_ack,
    }


def transcribe_audio(
    stt_engine,
    audio_bytes: bytes,
    initial_prompt: Optional[str] = None,
    suffix: str = ".wav",
) -> str:
    if not audio_bytes:
        return ""
    return stt_engine.transcribe_bytes(
        audio_bytes,
        suffix=suffix,
        initial_prompt=initial_prompt,
    )


def synthesize_audio(tts_engine, text: str) -> bytes:
    return tts_engine.synthesize_to_wav_bytes(text)


def cleanup_old_audio(max_age_seconds: Optional[int] = None) -> None:
    max_age = Config.AUDIO_TTL_SECONDS if max_age_seconds is None else max_age_seconds
    audio_dir = Config.AUDIO_DIR
    if not os.path.isdir(audio_dir):
        return
    now = time.time()
    for name in os.listdir(audio_dir):
        path = os.path.join(audio_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            if now - os.path.getmtime(path) > max_age:
                os.unlink(path)
        except OSError as e:
            logger.warning("Failed to delete old audio %s: %s", path, e)


def resolved_tts_audio_path(job_id: str, audio_path: str) -> Optional[str]:
    """Return the path only if it is the WAV for this job under AUDIO_DIR."""
    if not audio_path:
        return None
    audio_dir = Path(Config.AUDIO_DIR).resolve()
    expected = (audio_dir / f"{job_id}.wav").resolve()
    actual = Path(audio_path).resolve()
    if actual != expected or not actual.is_file():
        return None
    try:
        actual.relative_to(audio_dir)
    except ValueError:
        return None
    return str(actual)


def start_tts_job(
    job_store: JobStore,
    tts_engine,
    text: str,
    query_id: Optional[str] = None,
) -> str:
    cleanup_old_audio()
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
