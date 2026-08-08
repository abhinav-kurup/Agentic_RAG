import json
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from backend.dependencies import get_audit_logger

router = APIRouter(tags=["audit"])


@router.get("/audit/logs")
async def get_audit_logs(audit_logger=Depends(get_audit_logger)):
    logs = audit_logger.get_logs()
    logs.sort(
        key=lambda l: l.get("timestamp") or l.get("created_at") or "",
        reverse=True,
    )
    return {"logs": logs, "count": len(logs)}


@router.get("/metrics")
async def get_metrics():
    metrics_path = os.path.join("evaluation", "results", "my_metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=404,
            detail=f"Metrics file not found at {metrics_path}. Run evaluations first.",
        )
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content=data)
