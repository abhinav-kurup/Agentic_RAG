from fastapi import APIRouter, Depends, HTTPException

from backend.api.schemas import ChatQueryRequest, ChatQueryResponse
from backend.dependencies import get_audit_logger, get_orchestrator
from backend.services.query import run_query

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/query", response_model=ChatQueryResponse)
async def chat_query(
    body: ChatQueryRequest,
    orchestrator=Depends(get_orchestrator),
    audit_logger=Depends(get_audit_logger),
):
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    session_id = (body.session_id or "").strip() or "default_session"

    try:
        result = await run_query(
            orchestrator=orchestrator,
            audit_logger=audit_logger,
            query=query,
            session_id=session_id,
            query_id=body.query_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return ChatQueryResponse(**result)
