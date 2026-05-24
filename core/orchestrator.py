from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from core.state import AgentState
from agents.router import RouterAgent, reject_query, no_docs_found_reply
from agents.retrieval2 import RetrievalAgent
from agents.extraction import ExtractionAgent
from agents.analysis import AnalysisAgent, calculator
from vectorstore.chroma import VectorStoreManager
import logging
import uuid
from core.conditions import route_query, no_docs_found


class Orchestrator:
    def __init__(self,vector_store:VectorStoreManager = None):
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
        
        builder.set_entry_point("router")
        
        builder.add_conditional_edges(
            "router",
            route_query,
            {
                "reject": "reject",
                "retrieval": "retrieval"
            }
        )
        
        # builder.add_edge("retrieval", "extraction")
        builder.add_conditional_edges(
            "retrivel",
            no_docs_found,
            {
                "retrieval": "extraction",
                "no_docs_found": "no_docs_found_reply"
            }
        builder.add_edge("extraction", "analysis")
        builder.add_edge("reject", END)
        
        builder.add_conditional_edges(
            "analysis",
            tools_condition,
            {
                "tools": "tools",
                END: END
            }
        )
        builder.add_edge("tools", "analysis")
        
        self.workflow = builder.compile()

    def run(self, query: str, query_id: str = None, audit_logger = None) -> AgentState:
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
            "audit_log": [{"step": "Orchestrator", "status": "Start", "query": query}]
        }
        
        logger.info(f"Starting workflow for query: {query}")
        result = self.workflow.invoke(initial_state)
        logger.info("Workflow completed")
        
        from utils.helpers import dump_agent_state
        dump_agent_state(result, "FinalOutput")
        
        return result
