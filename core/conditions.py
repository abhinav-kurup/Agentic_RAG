from core.state import AgentState
import logging


logger = logging.getLogger(__name__)

def route_query(state: AgentState) -> str:
    route = state.get("route", "document")
    if route == "conversational":
        return "reject"
    else:
        return "retrieval"


def no_docs_found(state: AgentState) -> str:
    if bool(state["retrieved_docs"]):
        return "retrieval"
    else:
        return "no_docs_found"

