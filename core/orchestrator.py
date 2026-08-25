import asyncio
import logging
import time
import uuid
from typing import Optional

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from core.state import AgentState
from agents.planner import PlannerAgent, conversational_reply
from agents.analysis import AnalysisAgent
from agents.query_condenser import QueryContextualizer
from agents.react_agent import AgenticController
from vectorstore import VectorStoreManager
from core.conditions import route_query, after_agent
from utils.helpers import dump_agent_state

logger = logging.getLogger(__name__)


@traceable(name="Response")
def trace_response(final_response: str, citations: list = None):
    return final_response


class Orchestrator:
    def __init__(self, vector_store: VectorStoreManager = None, session_id: str = None):
        if vector_store is None:
            vector_store = VectorStoreManager()

        self.contextualizer = QueryContextualizer()
        self.planner_agent = PlannerAgent()
        self.analysis_agent = AnalysisAgent()
        self.controller = AgenticController(vector_store)
        self.memory = MemorySaver()
        self.config = {"configurable": {"thread_id": session_id}}

        builder = StateGraph(AgentState)
        builder.add_node("contextualizer", self.contextualizer.ainvoke)
        builder.add_node("router", self.planner_agent.ainvoke)
        builder.add_node("conversational", conversational_reply)
        builder.add_node("memory_gate", self.controller.memory_gate)
        builder.add_node("agent", self.controller.agent_node)
        builder.add_node("tools", ToolNode(self.controller.tools))
        builder.add_node("ingest", self.controller.ingest_node)
        builder.add_node("critic", self.controller.critic_node)
        builder.add_node("prepare", self.controller.prepare_synthesis)
        builder.add_node("analysis", self.analysis_agent.ainvoke)

        builder.set_entry_point("contextualizer")
        builder.add_edge("contextualizer", "router")
        builder.add_conditional_edges(
            "router",
            route_query,
            {
                "conversational": "conversational",
                "agentic": "memory_gate",
            },
        )
        builder.add_edge("conversational", END)
        builder.add_edge("memory_gate", "agent")
        builder.add_conditional_edges(
            "agent",
            after_agent,
            {"tools": "tools", "prepare": "prepare"},
        )
        builder.add_edge("tools", "ingest")
        builder.add_edge("ingest", "critic")
        builder.add_edge("critic", "agent")
        builder.add_edge("prepare", "analysis")
        builder.add_edge("analysis", END)

        self.workflow = builder.compile(checkpointer=self.memory)

    async def reconstruct_query(self, query: str, session_id: str = None) -> str:
        """Rewrite a follow-up into a standalone query using this session's chat memory."""
        query = (query or "").strip()
        if not query:
            return query

        config = {"configurable": {"thread_id": session_id or "default_session"}}
        history = []
        summary = ""
        try:
            snapshot = self.workflow.get_state(config)
            values = snapshot.values if snapshot and snapshot.values else {}
            history = list(values.get("messages") or [])
            summary = values.get("summary") or ""
        except Exception:
            logger.exception("Could not load session history for query reconstruct")

        result = await self.contextualizer.ainvoke(
            {
                "query": query,
                "messages": history + [HumanMessage(content=query)],
                "summary": summary,
            }
        )
        rewritten = (result.get("standalone_query") or query).strip()
        return rewritten or query

    def session_context(self, session_id: str = None, last_n: int = 8) -> dict:
        config = {"configurable": {"thread_id": session_id or "default_session"}}
        history_text = ""
        has_answer = False
        try:
            snapshot = self.workflow.get_state(config)
            values = snapshot.values if snapshot and snapshot.values else {}
            messages = list(values.get("messages") or [])
            lines = []
            for msg in messages[-last_n:]:
                content = (getattr(msg, "content", None) or "")[:280].replace("\n", " ")
                kind = getattr(msg, "type", "msg")
                if kind in ("ai", "assistant") and content.strip():
                    has_answer = True
                lines.append(f"{kind}: {content}")
            history_text = "\n".join(lines)
        except Exception:
            logger.exception("Could not load session context for voice router")
        return {"history_text": history_text, "has_answer": has_answer}

    @traceable(name="RAG Pipeline (Async)")
    async def arun(self, query: str, query_id: str = None, session_id: str = None) -> AgentState:
        query_id = query_id or str(uuid.uuid4())
        start_time = time.time()

        config = {
            "configurable": {"thread_id": session_id or "default_session"},
            "recursion_limit": 32,
        }

        # Do not reset evidence_store / summary — the checkpointer is session memory.
        initial_state = {
            "query": query,
            "standalone_query": None,
            "query_id": query_id,
            "messages": [HumanMessage(content=query)],
            "citations": [],
            "subqueries": [],
            "retrieval_results": [],
            "extracted_data": {},
            "final_response": None,
            "route": None,
            "topic_shift": False,
            "hop_count": 0,
            "empty_retrievals": 0,
            "last_tool": None,
            "critic": None,
            "audit_log": [{"step": "Orchestrator", "status": "Start", "query": query}],
        }

        logger.info("Starting async workflow for query: %s", query)
        try:
            result = await self.workflow.ainvoke(initial_state, config)
            trace_response(result.get("final_response"), result.get("citations"))
        except Exception:
            logger.exception("Workflow failed for query: %s", query)
            raise
        logger.info("Async workflow completed")

        dump_agent_state(result, "FinalOutput")

        latency = time.time() - start_time
        run_tree = get_current_run_tree()
        if run_tree:
            doc_names = []
            for doc in result.get("retrieved_docs", []):
                meta = doc.get("metadata", {})
                if meta and meta.get("source"):
                    doc_names.append(meta.get("source"))
            
            metadata = {
                "query": query,
                "route": result.get("route"),
                "documents": doc_names,
                "hops": result.get("hop_count"),
                "latency": latency,
            }
            if run_tree.parent_run_id is None:
                metadata["type"] = "production"
            run_tree.metadata.update(metadata)

        return result

    @traceable(name="RAG Pipeline")
    def run(self, query: str, query_id: str = None, session_id: str = None) -> AgentState:
        """Synchronous wrapper around arun for backwards compatibility."""
        return asyncio.run(self.arun(query, query_id=query_id, session_id=session_id))
