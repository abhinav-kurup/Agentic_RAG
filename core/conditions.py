from core.state import AgentState


def route_query(state: AgentState) -> str:
    route = state.get("route", "document")
    if route == "conversational":
        return "reject"
    return "retrieval"


def no_docs_found(state: AgentState) -> str:
    if state.get("retrieved_docs"):
        return "extraction"
    return "no_docs_found"
