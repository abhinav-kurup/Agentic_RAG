import logging
import uuid
from typing import Any, Dict, Optional

from core.orchestrator import Orchestrator
from audit.logger import AuditLogger

logger = logging.getLogger(__name__)


async def run_query(
    orchestrator: Orchestrator,
    audit_logger: AuditLogger,
    query: str,
    session_id: str,
    query_id: Optional[str] = None,
) -> Dict[str, Any]:
    query_id = query_id or str(uuid.uuid4())

    try:
        result_state = await orchestrator.arun(
            query,
            query_id=query_id,
            session_id=session_id,
        )
    except Exception:
        logger.exception("Orchestrator failed for query: %s", query[:120])
        raise

    response = result_state.get("final_response") or (
        "I apologize, but I could not generate an answer. "
        "Please check the Audit Logs for more details."
    )

    audit_logger.log_query(query_id, result_state)

    confidence = result_state.get("confidence")
    if confidence is None:
        for step in reversed(result_state.get("audit_log", [])):
            if isinstance(step, dict) and step.get("confidence") is not None:
                confidence = step["confidence"]
                break

    return {
        "query_id": query_id,
        "session_id": session_id,
        "response": response,
        "citations": result_state.get("citations") or [],
        "route": result_state.get("route"),
        "confidence": confidence,
        "audit_log": result_state.get("audit_log", []),
        "retrieved_docs": _serialize_docs(result_state.get("retrieved_docs", [])),
    }


def _serialize_docs(docs) -> list:
    serialized = []
    for doc in docs or []:
        if isinstance(doc, dict):
            serialized.append(
                {
                    "content": doc.get("content") or doc.get("text") or "",
                    "metadata": doc.get("metadata", {}),
                    "score": doc.get("score"),
                }
            )
    return serialized
