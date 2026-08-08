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
from agents.retrieval import SingleHopRetrievalAgent, MultiHopRetrievalAgent
from agents.extraction import ExtractionAgent
from agents.analysis import AnalysisAgent, calculator
from agents.query_condenser import QueryContextualizer
from vectorstore import VectorStoreManager
from core.conditions import route_query, route_after_analysis
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
        self.single_hop_retrieval_agent = SingleHopRetrievalAgent(vector_store=vector_store)
        self.multi_hop_retrieval_agent = MultiHopRetrievalAgent(vector_store=vector_store)
        self.extraction_agent = ExtractionAgent()
        self.analysis_agent = AnalysisAgent()
        self.tool_node = ToolNode([calculator])
        self.memory = MemorySaver()
        self.config = {"configurable": {"thread_id": session_id}}

        builder = StateGraph(AgentState)
        builder.add_node("contextualizer", self.contextualizer.ainvoke)
        builder.add_node("router", self.planner_agent.ainvoke)
        builder.add_node("conversational", conversational_reply)
        builder.add_node("single_hop_retrieval", self.single_hop_retrieval_agent.ainvoke)
        builder.add_node("multi_hop_retrieval", self.multi_hop_retrieval_agent.ainvoke)
        builder.add_node("extraction", self.extraction_agent.ainvoke)
        builder.add_node("analysis", self.analysis_agent.ainvoke)
        builder.add_node("tools", self.tool_node)

        builder.set_entry_point("contextualizer")
        builder.add_edge("contextualizer", "router")

        builder.add_conditional_edges(
            "router",
            route_query,
            {
                "conversational": "conversational",
                "single_hop": "single_hop_retrieval",
                "multi_hop": "multi_hop_retrieval",
            },
        )

        builder.add_edge("single_hop_retrieval", "extraction")
        builder.add_edge("multi_hop_retrieval", "extraction")
        builder.add_edge("extraction", "analysis")
        builder.add_edge("conversational", END)

        builder.add_conditional_edges(
            "analysis",
            route_after_analysis,
            {
                "tools": "tools",
                END: END,
            },
        )
        builder.add_edge("tools", "analysis")

        self.workflow = builder.compile(checkpointer=self.memory)

    @traceable(name="RAG Pipeline (Async)")
    async def arun(self, query: str, query_id: str = None, session_id: str = None) -> AgentState:
        query_id = query_id or str(uuid.uuid4())
        start_time = time.time()

        config = {"configurable": {"thread_id": session_id or "default_session"}}

        initial_state = {
            "query": query,
            "standalone_query": None,
            "query_id": query_id,
            "summary": None,

            "messages": [HumanMessage(content=query)],   
            "retrieved_docs": [],
            "subqueries": [],
            "retrieval_results": [],
            "extracted_data": {},
            "final_response": None,
            "route": None,
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
                "latency": latency
            }
            if run_tree.parent_run_id is None:
                metadata["type"] = "production"
            run_tree.metadata.update(metadata)

        return result

    @traceable(name="RAG Pipeline")
    def run(self, query: str, query_id: str = None, session_id: str = None) -> AgentState:
        """Synchronous wrapper around arun for backwards compatibility."""
        return asyncio.run(self.arun(query, query_id=query_id, session_id=session_id))
