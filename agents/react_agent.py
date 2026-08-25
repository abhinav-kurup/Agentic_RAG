"""Self-sustaining ReAct loop: think → tool → ingest → critic → think."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
from pydantic import BaseModel, Field

from agents.memory import (
    format_evidence_inventory,
    merge_evidence,
    maybe_summarize,
    trim_messages,
)
from agents.prompt import REACT_AGENT_SYSTEM_PROMPT_TEXT
from agents.tools import DocumentToolkit
from core.config import Config
from core.llm import get_llm
from core.state import AgentState
from utils.helpers import dump_agent_state

logger = logging.getLogger(__name__)


class CriticVerdict(BaseModel):
    sufficient: bool = Field(description="True if working memory can answer the query.")
    missing: str = Field(default="", description="What is still missing.")
    recommendation: str = Field(
        description="One of: answer, search_again, read_pages, give_up"
    )


def _docs_from_tool_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return []
    if not isinstance(payload, dict):
        return []
    docs = []
    for chunk in payload.get("chunks") or []:
        if not isinstance(chunk, dict):
            continue
        content = (chunk.get("content") or "").strip()
        if not content:
            continue
        docs.append(
            {
                "content": content,
                "score": chunk.get("score"),
                "metadata": {
                    "source": chunk.get("source"),
                    "page_number": chunk.get("page"),
                },
            }
        )
    return docs


class AgenticController:
    def __init__(self, vector_store):
        self.toolkit = DocumentToolkit(vector_store)
        self.tools = self.toolkit.as_langchain_tools()
        self.toolkit.bind_evidence_getter(lambda: self._latest_evidence)
        self._latest_evidence: List[Dict[str, Any]] = []
        self.llm = get_llm(Config.AGENT_MODEL, temperature=0.0)
        try:
            self.critic_llm = self.llm.with_structured_output(CriticVerdict)
        except Exception:
            self.critic_llm = None

    def _sync_evidence(self, state: AgentState) -> None:
        self._latest_evidence = list(state.get("evidence_store") or [])

    def _system_prompt(self, state: AgentState) -> str:
        critic = state.get("critic") or {}
        hints = state.get("subqueries") or []
        return REACT_AGENT_SYSTEM_PROMPT_TEXT.format(
            evidence_inventory=format_evidence_inventory(state.get("evidence_store") or []),
            summary=state.get("summary") or "(none)",
            subquery_hints="\n".join(f"- {h}" for h in hints) or "(none)",
            critic=json.dumps(critic) if critic else "(none yet)",
            hop_count=state.get("hop_count") or 0,
            max_hops=Config.AGENT_MAX_HOPS,
            empty_retrievals=state.get("empty_retrievals") or 0,
        )

    @traceable(name="MemoryGate")
    async def memory_gate(self, state: AgentState) -> Dict[str, Any]:
        dump_agent_state(state, "MemoryGate")
        updates: Dict[str, Any] = {"hop_count": 0, "empty_retrievals": 0, "critic": None}
        if state.get("topic_shift"):
            updates["evidence_store"] = []
            updates["retrieved_docs"] = []
            updates["audit_log"] = [{
                "step": "MemoryGate",
                "status": "Success",
                "action": "cleared_working_memory",
                "reason": "topic_shift",
            }]
        summary = await maybe_summarize(state)
        if summary:
            updates["summary"] = summary
        if "audit_log" not in updates:
            updates["audit_log"] = [{
                "step": "MemoryGate",
                "status": "Success",
                "evidence": len(state.get("evidence_store") or []),
            }]
        return updates

    @traceable(name="ReactAgent")
    async def agent_node(self, state: AgentState) -> Dict[str, Any]:
        dump_agent_state(state, "ReactAgent")
        self._sync_evidence(state)
        hops = int(state.get("hop_count") or 0)
        force_stop = hops >= Config.AGENT_MAX_HOPS or int(state.get("empty_retrievals") or 0) >= 2
        llm = self.llm.bind_tools(self.tools) if not force_stop else self.llm
        window = trim_messages(state.get("messages") or [], Config.MEMORY_MESSAGE_WINDOW)
        user_q = state.get("standalone_query") or state.get("query", "")
        payload = [SystemMessage(content=self._system_prompt(state))] + window
        if not any(getattr(m, "type", "") in ("human", "user") for m in window):
            payload.append(HumanMessage(content=user_q))
        response = await llm.ainvoke(payload)
        tool_calls = getattr(response, "tool_calls", None) or []
        names = [
            tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
            for tc in tool_calls
        ]
        logger.info("ReactAgent tools=%s force_stop=%s", names, force_stop)
        return {
            "messages": [response],
            "route": "agentic",
            "audit_log": [{
                "step": "ReactAgent",
                "status": "Success",
                "hop": hops,
                "tool_calls": names,
                "force_stop": force_stop,
            }],
        }

    @traceable(name="IngestObservations")
    async def ingest_node(self, state: AgentState) -> Dict[str, Any]:
        dump_agent_state(state, "IngestObservations")
        incoming: List[Dict[str, Any]] = []
        last_tool = None
        traces = []
        for msg in reversed(state.get("messages") or []):
            if not isinstance(msg, ToolMessage):
                if isinstance(msg, AIMessage):
                    break
                continue
            last_tool = getattr(msg, "name", None) or last_tool
            docs = _docs_from_tool_payload(getattr(msg, "content", "") or "")
            incoming.extend(docs)
            traces.append({
                "tool": last_tool,
                "hits": len(docs),
                "preview": str(getattr(msg, "content", ""))[:240],
            })
        traces.reverse()
        empty = last_tool in {"search_docs", "read_pages"} and not incoming
        evidence = merge_evidence(
            state.get("evidence_store") or [],
            incoming,
            Config.EVIDENCE_MAX_CHUNKS,
        )
        empty_retrievals = int(state.get("empty_retrievals") or 0)
        if empty:
            empty_retrievals += 1
        elif incoming:
            empty_retrievals = 0
        return {
            "evidence_store": evidence,
            "retrieved_docs": evidence,
            "hop_count": int(state.get("hop_count") or 0) + 1,
            "empty_retrievals": empty_retrievals,
            "last_tool": last_tool,
            "tool_trace": traces,
            "audit_log": [{
                "step": "IngestObservations",
                "status": "Success",
                "last_tool": last_tool,
                "new_chunks": len(incoming),
                "working_memory": len(evidence),
                "empty_retrievals": empty_retrievals,
            }],
        }

    @traceable(name="EvidenceCritic")
    async def critic_node(self, state: AgentState) -> Dict[str, Any]:
        dump_agent_state(state, "EvidenceCritic")
        query = state.get("standalone_query") or state.get("query", "")
        hops = int(state.get("hop_count") or 0)
        empty = int(state.get("empty_retrievals") or 0)
        has_ev = bool(state.get("evidence_store"))
        fallback = CriticVerdict(
            sufficient=has_ev and empty < 2,
            missing="" if has_ev else "No document chunks in working memory.",
            recommendation=(
                "give_up" if empty >= 2 else
                ("answer" if has_ev and hops >= 2 else "search_again")
            ),
        )
        verdict = fallback
        if self.critic_llm:
            try:
                verdict = await self.critic_llm.ainvoke(
                    [
                        SystemMessage(content="Grade if working memory can answer. JSON only."),
                        HumanMessage(
                            content=(
                                f"Query: {query}\nHops: {hops}/{Config.AGENT_MAX_HOPS}\n"
                                f"Empty retrievals: {empty}\nLast tool: {state.get('last_tool')}\n\n"
                                f"{format_evidence_inventory(state.get('evidence_store') or [])}"
                            )
                        ),
                    ]
                )
                if not isinstance(verdict, CriticVerdict):
                    verdict = CriticVerdict.model_validate(verdict)
            except Exception:
                logger.exception("Critic failed; using heuristic")
                verdict = fallback
        if hops >= Config.AGENT_MAX_HOPS or empty >= 2:
            verdict.recommendation = "give_up" if not has_ev else "answer"
            verdict.sufficient = has_ev
        payload = verdict.model_dump()
        return {
            "critic": payload,
            "audit_log": [{"step": "EvidenceCritic", "status": "Success", **payload}],
        }

    async def prepare_synthesis(self, state: AgentState) -> Dict[str, Any]:
        evidence = state.get("evidence_store") or state.get("retrieved_docs") or []
        extracted = ""
        for trace in reversed(state.get("tool_trace") or []):
            if trace.get("tool") == "extract_tables":
                extracted = trace.get("preview") or ""
                break
        return {
            "retrieved_docs": evidence,
            "extracted_data": {"content": extracted} if extracted else (state.get("extracted_data") or {}),
            "route": "agentic",
            "audit_log": [{
                "step": "PrepareSynthesis",
                "status": "Success",
                "evidence": len(evidence),
                "hops": state.get("hop_count") or 0,
            }],
        }
