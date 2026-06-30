from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    query: str
    query_id: Optional[str]
    audit_logger: Optional[Any]
    messages: Annotated[List[BaseMessage], operator.add]
    route: Optional[str]
    subqueries: List[str]
    retrieved_docs: List[Dict[str, Any]]
    retrieval_results: List[Dict[str, Any]]
    extracted_data: Dict[str, Any]
    final_response: Optional[str]
    audit_log: Annotated[List[Dict[str, Any]], operator.add]
    citations: List[str]
