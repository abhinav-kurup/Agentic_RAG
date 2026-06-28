import logging

from langgraph.graph import END

from core.state import AgentState

logger = logging.getLogger(__name__)


def route_query(state: AgentState) -> str:    
    route = state.get("route", "single_hop")
    if route == "conversational":
        return "conversational"
    elif route == "multi_hop":
        return "multi_hop"
    return "single_hop"


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
