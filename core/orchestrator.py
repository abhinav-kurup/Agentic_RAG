from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
import time

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from core.state import AgentState
from agents.planner import PlannerAgent, conversational_reply
from agents.retrieval import RetrievalAgent
from agents.extraction import ExtractionAgent
# from agents.analysis import AnalysisAgent, calculator
from agents.analysis2 import AnalysisAgent, calculator
from vectorstore.chroma import VectorStoreManager
import logging
import uuid
from core.conditions import route_query, route_after_analysis

logger = logging.getLogger(__name__)


@traceable(name="Response")
def trace_response(final_response: str, citations: list = None):
    return final_response


class Orchestrator:
    def __init__(self, vector_store: VectorStoreManager = None):
        if vector_store is None:
            vector_store = VectorStoreManager()
        self.planner_agent = PlannerAgent()
        self.retrieval_agent = RetrievalAgent(vector_store=vector_store)
        self.extraction_agent = ExtractionAgent()
        self.analysis_agent = AnalysisAgent()
        self.tool_node = ToolNode([calculator])

        builder = StateGraph(AgentState)

        builder.add_node("router", self.planner_agent.invoke)
        builder.add_node("conversational", conversational_reply)
        builder.add_node("retrieval", self.retrieval_agent.invoke)
        builder.add_node("extraction", self.extraction_agent.invoke)
        builder.add_node("analysis", self.analysis_agent.invoke)
        builder.add_node("tools", self.tool_node)

        builder.set_entry_point("router")

        builder.add_conditional_edges(
            "router",
            route_query,
            {
                "conversational": "conversational",
                "retrieval": "retrieval",
            },
        )

        builder.add_edge("retrieval", "extraction")
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

        self.workflow = builder.compile()

    @traceable(name="RAG Pipeline")
    def run(self, query: str, query_id: str = None, audit_logger=None) -> AgentState:
        query_id = query_id or str(uuid.uuid4())
        start_time = time.time()

        initial_state = {
            "query": query,
            "query_id": query_id,
            "audit_logger": audit_logger,
            "messages": [],
            "retrieved_docs": [],
            "subqueries": [],
            "retrieval_results": [],
            "extracted_data": {},
            "final_response": None,
            "route": None,
            "audit_log": [{"step": "Orchestrator", "status": "Start", "query": query}],
        }

        logger.info("Starting workflow for query: %s", query)
        try:
            result = self.workflow.invoke(initial_state)
            trace_response(result.get("final_response"), result.get("citations"))
        except Exception:
            logger.exception("Workflow failed for query: %s", query)
            raise
        logger.info("Workflow completed")

        from utils.helpers import dump_agent_state
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
