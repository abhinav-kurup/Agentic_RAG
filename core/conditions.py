import logging

from langgraph.graph import END

from core.state import AgentState

logger = logging.getLogger(__name__)


def route_query(state: AgentState) -> str:    
    route = state.get("route", "document")
    if route == "conversational":
        return "reject"
    return "retrieval"


def no_docs_found(state: AgentState) -> str:
    if state.get("retrieved_docs"):
        return "extraction"
    return "no_docs_found"


def route_after_analysis(state: AgentState):
    """End on final_response or errors; only route to tools when the LLM requested them."""
    if state.get("final_response"):
        return END

    messages = state.get("messages") or []
    if not messages:
        logger.warning(
            "Analysis left no messages and no final_response; ending workflow."
        )
        return END

    last = messages[-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if tool_calls:
        return "tools"
    return END
