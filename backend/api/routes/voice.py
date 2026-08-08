from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from core.config import Config
from backend.api.schemas import JobResponse, TranscribeResponse, TTSJobResponse, TTSRequest
from backend.dependencies import get_job_store_dep, get_stt_engine, get_tts_engine
from backend.services.voice import start_tts_job, transcribe_audio

router = APIRouter(prefix="/voice", tags=["voice"])


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    stt_engine=Depends(get_stt_engine),
):
    if not Config.ENABLE_VOICE:
        raise HTTPException(status_code=503, detail="Voice input is disabled.")
    if stt_engine is None:
        raise HTTPException(status_code=503, detail="STT engine not loaded.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    try:
        transcript = transcribe_audio(stt_engine, audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return TranscribeResponse(transcript=transcript)


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

    audio_path = job.result.get("audio_path")
    if not audio_path:
        raise HTTPException(status_code=404, detail="Audio file missing.")

    return FileResponse(audio_path, media_type="audio/wav", filename=f"{job_id}.wav")
