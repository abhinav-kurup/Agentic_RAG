import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from core.config import Config
from backend.api.schemas import (
    JobResponse,
    TranscribeResponse,
    TTSJobResponse,
    TTSRequest,
    VoiceInterpretResponse,
)
from backend.dependencies import (
    get_document_registry,
    get_job_store_dep,
    get_orchestrator,
    get_stt_engine,
    get_tts_engine,
)
from backend.services.document_registry import DocumentRegistry
from backend.services.voice import (
    interpret_audio,
    resolved_tts_audio_path,
    start_tts_job,
    synthesize_audio,
    transcribe_audio,
)

router = APIRouter(prefix="/voice", tags=["voice"])


def _stt_prompt(registry: DocumentRegistry) -> str:
    try:
        names = registry.list_all()[:12]
    except Exception:
        return ""
    return ", ".join(Path(name).stem for name in names)[:400]


def _truthy(value: str) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    stt_engine=Depends(get_stt_engine),
    registry: DocumentRegistry = Depends(get_document_registry),
):
    if not Config.ENABLE_VOICE:
        raise HTTPException(status_code=503, detail="Voice input is disabled.")
    if stt_engine is None:
        raise HTTPException(status_code=503, detail="STT engine not loaded.")

    audio_bytes = await file.read(Config.AUDIO_MAX_BYTES + 1)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    if len(audio_bytes) > Config.AUDIO_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large.")

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    prompt = _stt_prompt(registry) or None

    try:
        transcript = await asyncio.to_thread(
            transcribe_audio,
            stt_engine,
            audio_bytes,
            prompt,
            suffix,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return TranscribeResponse(transcript=transcript)


@router.post("/interpret", response_model=VoiceInterpretResponse)
async def interpret(
    file: UploadFile = File(...),
    session_id: str = Form(""),
    awaiting_confirm: str = Form("false"),
    has_last_answer: str = Form("false"),
    stt_engine=Depends(get_stt_engine),
    orchestrator=Depends(get_orchestrator),
    registry: DocumentRegistry = Depends(get_document_registry),
):
    if not Config.ENABLE_VOICE:
        raise HTTPException(status_code=503, detail="Voice input is disabled.")
    if stt_engine is None:
        raise HTTPException(status_code=503, detail="STT engine not loaded.")

    audio_bytes = await file.read(Config.AUDIO_MAX_BYTES + 1)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")
    if len(audio_bytes) > Config.AUDIO_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Audio file too large.")

    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    prompt = _stt_prompt(registry) or None
    thread_id = (session_id or "").strip() or "default_session"

    try:
        result = await interpret_audio(
            stt_engine=stt_engine,
            orchestrator=orchestrator,
            audio_bytes=audio_bytes,
            suffix=suffix,
            session_id=thread_id,
            awaiting_confirm=_truthy(awaiting_confirm),
            initial_prompt=prompt,
            has_last_answer=_truthy(has_last_answer),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return VoiceInterpretResponse(**result)


@router.post("/synthesize")
async def synthesize(
    body: TTSRequest,
    tts_engine=Depends(get_tts_engine),
):
    if not Config.ENABLE_TTS:
        raise HTTPException(status_code=503, detail="TTS is disabled.")
    if tts_engine is None:
        raise HTTPException(status_code=503, detail="TTS engine not loaded.")

    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if len(text) > Config.TTS_MAX_CHARS:
        raise HTTPException(status_code=400, detail="Text too long to synthesize.")

    try:
        wav = await asyncio.to_thread(synthesize_audio, tts_engine, text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return Response(content=wav, media_type="audio/wav")


@router.post("/synthesize/async", response_model=TTSJobResponse)
async def synthesize_async(
    body: TTSRequest,
    tts_engine=Depends(get_tts_engine),
    job_store=Depends(get_job_store_dep),
):
    if not Config.ENABLE_TTS:
        raise HTTPException(status_code=503, detail="TTS is disabled.")
    if tts_engine is None:
        raise HTTPException(status_code=503, detail="TTS engine not loaded.")

    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if len(text) > Config.TTS_MAX_CHARS:
        raise HTTPException(status_code=400, detail="Text too long to synthesize.")

    job_id = start_tts_job(job_store, tts_engine, text, query_id=body.query_id)
    return TTSJobResponse(job_id=job_id, status="processing")


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_tts_job(job_id: str, job_store=Depends(get_job_store_dep)):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobResponse(**job.to_dict())


@router.get("/jobs/{job_id}/audio")
async def download_tts_audio(job_id: str, job_store=Depends(get_job_store_dep)):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status.value != "completed" or not job.result:
        raise HTTPException(status_code=404, detail="Audio not ready.")

    audio_path = resolved_tts_audio_path(job_id, job.result.get("audio_path"))
    if not audio_path:
        raise HTTPException(status_code=404, detail="Audio file missing.")

    return FileResponse(audio_path, media_type="audio/wav", filename=f"{job_id}.wav")
