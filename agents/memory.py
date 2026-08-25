"""Three-layer memory for the agentic RAG loop.

1. Short-term: LangGraph checkpointer (messages, per session_id).
2. Working: evidence_store — deduped chunks the agent has actually fetched.
3. Long-term: Qdrant (untouched here). Summary compresses old turns.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from core.config import Config
from core.llm import get_llm

logger = logging.getLogger(__name__)


def chunk_id(content: str) -> str:
    return hashlib.md5((content or "").strip().encode("utf-8")).hexdigest()


def format_evidence_inventory(evidence: List[Dict[str, Any]], limit: int = 12) -> str:
    if not evidence:
        return "(empty — you must search_docs before answering)"
    lines = []
    for i, doc in enumerate(evidence[:limit], start=1):
        meta = doc.get("metadata") or {}
        source = meta.get("source", "unknown")
        page = meta.get("page_number", "?")
        score = doc.get("score")
        score_s = f"{float(score):.3f}" if isinstance(score, (int, float)) else "-"
        preview = (doc.get("content") or "").replace("\n", " ").strip()[:180]
        lines.append(f"{i}. {source} p.{page} score={score_s} :: {preview}")
    if len(evidence) > limit:
        lines.append(f"... {len(evidence) - limit} more chunks in working memory")
    return "\n".join(lines)


def merge_evidence(
    existing: List[Dict[str, Any]],
    incoming: List[Dict[str, Any]],
    max_chunks: int,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for doc in list(existing or []) + list(incoming or []):
        content = (doc.get("content") or "").strip()
        if not content:
            continue
        key = chunk_id(content)
        prev = merged.get(key)
        if prev is None or float(doc.get("score") or 0) > float(prev.get("score") or 0):
            item = dict(doc)
            item["content"] = content
            item["chunk_id"] = key
            if "metadata" not in item or not isinstance(item["metadata"], dict):
                item["metadata"] = {}
            merged[key] = item
    ranked = sorted(
        merged.values(),
        key=lambda d: float(d.get("score") or 0.0),
        reverse=True,
    )
    return ranked[:max_chunks]


def trim_messages(messages: list, window: int) -> list:
    if not messages:
        return []
    if len(messages) <= window:
        return list(messages)
    return list(messages[-window:])


async def maybe_summarize(state: Dict[str, Any]) -> Optional[str]:
    """Compress older turns when the transcript grows past the window."""
    messages = list(state.get("messages") or [])
    window = Config.MEMORY_MESSAGE_WINDOW
    if len(messages) <= window + 4:
        return None
    stale = messages[: -window]
    bits = []
    for msg in stale[-12:]:
        role = getattr(msg, "type", "msg")
        text = (getattr(msg, "content", None) or "").replace("\n", " ").strip()
        if text:
            bits.append(f"{role}: {text[:400]}")
    if not bits:
        return None
    prior = (state.get("summary") or "").strip()
    llm = get_llm(Config.PLANNER_MODEL, temperature=0.0)
    prompt = (
        "Summarize this document-QA chat for later retrieval. Keep paper names, "
        "agents, numbers, and unresolved questions. 8 sentences max.\n\n"
        f"Previous summary:\n{prior or '(none)'}\n\nTranscript:\n" + "\n".join(bits)
    )
    try:
        resp = await llm.ainvoke(
            [SystemMessage(content="You write compact memory summaries."), HumanMessage(content=prompt)]
        )
        text = (getattr(resp, "content", None) or "").strip()
        return text or None
    except Exception:
        logger.exception("Memory summarization failed")
        return None
