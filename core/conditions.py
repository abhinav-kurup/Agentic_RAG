import logging

from core.state import AgentState

logger = logging.getLogger(__name__)


def route_query(state: AgentState) -> str:
    """Greetings skip retrieval. Everything else enters the agentic ReAct loop."""
    route = state.get("route", "single_hop")
    if route == "conversational":
        return "conversational"
    return "agentic"


def after_agent(state: AgentState) -> str:
    """Continue the ReAct loop on native tool_calls; otherwise synthesize."""
    messages = state.get("messages") or []
    if not messages:
        return "prepare"
    last = messages[-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "prepare"
