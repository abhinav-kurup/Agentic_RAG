from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    version: str = "1.0.0"


class FeaturesResponse(BaseModel):
    enable_voice: bool
    enable_tts: bool
    qdrant_url: str
    embedding_model: str


class ChatQueryRequest(BaseModel):
    query: str
    session_id: str
    query_id: Optional[str] = None


class ChatQueryResponse(BaseModel):
    query_id: str
    session_id: str
    response: str
    citations: List[str] = Field(default_factory=list)
    route: Optional[str] = None
    confidence: Optional[float] = None
    audit_log: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    job_id: str
    status: str
    message: str


class DocumentListResponse(BaseModel):
    documents: List[str]


class JobResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    created_at: str
    updated_at: str
    progress: List[str] = Field(default_factory=list)
    progress_state: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TranscribeResponse(BaseModel):
    transcript: str


class VoiceInterpretResponse(BaseModel):
    transcript: str
    reconstructed_query: str
    intent: str
    spoken_ack: str = ""


class TTSRequest(BaseModel):
    text: str
    query_id: Optional[str] = None


class TTSJobResponse(BaseModel):
    job_id: str
    status: str


class ClearDatabaseResponse(BaseModel):
    message: str


class DeleteDocumentResponse(BaseModel):
    source: str
    deleted_count: int
