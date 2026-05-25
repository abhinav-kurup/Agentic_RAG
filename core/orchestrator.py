from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from core.state import AgentState
from agents.router import RouterAgent, reject_query, no_docs_found_reply
from agents.retrieval2 import RetrievalAgent
from agents.extraction import ExtractionAgent
from agents.analysis import AnalysisAgent, calculator
from vectorstore.chroma import VectorStoreManager
import logging
import uuid
from core.conditions import route_query, no_docs_found, route_after_analysis

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, vector_store: VectorStoreManager = None):
        if vector_store is None:
            vector_store = VectorStoreManager()
        self.router_agent = RouterAgent()
        self.retrieval_agent = RetrievalAgent(vector_store=vector_store)
        self.extraction_agent = ExtractionAgent()
        self.analysis_agent = AnalysisAgent()
        self.tool_node = ToolNode([calculator])

        builder = StateGraph(AgentState)

        builder.add_node("router", self.router_agent.invoke)
        builder.add_node("reject", reject_query)
        builder.add_node("retrieval", self.retrieval_agent.invoke)
        builder.add_node("extraction", self.extraction_agent.invoke)
        builder.add_node("analysis", self.analysis_agent.invoke)
        builder.add_node("tools", self.tool_node)
        builder.add_node("no_docs_found_reply", no_docs_found_reply)

        builder.set_entry_point("router")

        builder.add_conditional_edges(
            "router",
            route_query,
            {
                "reject": "reject",
                "retrieval": "retrieval",
            },
        )

        builder.add_conditional_edges(
            "retrieval",
            no_docs_found,
            {
                "extraction": "extraction",
                "no_docs_found": "no_docs_found_reply",
            },
        )
        builder.add_edge("extraction", "analysis")
        builder.add_edge("reject", END)
        builder.add_edge("no_docs_found_reply", END)

        builder.add_conditional_edges(
            "analysis",
            route_after_analysis,
            {
                "tools": "tools",
                END: END,
            },
        )
        builder.add_edge("tools", "analysis")
        png_data = self.workflow.get_graph().draw_mermaid_png()

        with open("workflow.png", "wb") as f:
            f.write(png_data)

        self.workflow = builder.compile()

    def run(self, query: str, query_id: str = None, audit_logger=None) -> AgentState:
        query_id = query_id or str(uuid.uuid4())

        initial_state = {
            "query": query,
            "query_id": query_id,
            "audit_logger": audit_logger,
            "messages": [],
            "retrieved_docs": [],
            "extracted_data": {},
            "final_response": None,
            "route": None,
            "audit_log": [{"step": "Orchestrator", "status": "Start", "query": query}],
        }

        logger.info("Starting workflow for query: %s", query)
        try:
            result = self.workflow.invoke(initial_state)
        except Exception:
            logger.exception("Workflow failed for query: %s", query)
            raise
        logger.info("Workflow completed")

        from utils.helpers import dump_agent_state
        dump_agent_state(result, "FinalOutput")

        return result
