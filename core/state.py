# from typing import TypedDict, List, Dict, Any, Optional, Annotated
# from langchain_core.messages import BaseMessage
# import operator

# class AgentState(TypedDict):
#     query: str
#     query_id: Optional[str]
#     audit_logger: Optional[Any]
#     summary: Optional[str]
#     messages: Annotated[List[BaseMessage], operator.add]
#     route: Optional[str]
#     subqueries: List[str]
#     retrieved_docs: List[Dict[str, Any]]
#     retrieval_results: List[Dict[str, Any]]
#     extracted_data: Dict[str, Any]
#     final_response: Optional[str]
#     audit_log: Annotated[List[Dict[str, Any]], operator.add]
#     citations: List[str]

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import operator

class AgentState(TypedDict):
    query: str                          # raw current-turn input, never overwritten
    standalone_query: Optional[str]      # NEW — history-resolved version of query
    query_id: Optional[str]
    summary: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]

    route: Optional[str]
    subqueries: List[str]
    retrieved_docs: List[Dict[str, Any]]
    retrieval_results: List[Dict[str, Any]]
    extracted_data: Dict[str, Any]
    final_response: Optional[str]
    audit_log: Annotated[List[Dict[str, Any]], operator.add]   # unchanged, this one's fine as-is
    citations: List[str]