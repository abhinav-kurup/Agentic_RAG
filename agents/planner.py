from langsmith import traceable
from core.state import AgentState
from core.config import Config
from typing import Dict, Any, List
import logging
import json
import re
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from utils.helpers import log_agent_step, dump_agent_state

logger = logging.getLogger(__name__)

class PlannerOutput(BaseModel):
    route: str = Field(
        description="The category of the query. Must be exactly 'conversational', 'single_hop', or 'multi_hop'."
    )
    subqueries: List[str] = Field(
        default_factory=list,
        description="List of independent retrieval subqueries. Empty for conversational. A single query for single_hop. Two or more queries for multi_hop."
    )

class PlannerAgent:
    """
    Analyzes and classifies queries to plan the retrieval route.
    If the query is multi_hop, it decomposes it into independent subqueries.
    """
    
    def __init__(self):
        from core.llm import get_llm
        model_identifier = Config.PLANNER_MODEL
        self.llm = get_llm(model_identifier, temperature=0.0)
        
        try:
            self.structured_llm = self.llm.with_structured_output(PlannerOutput)
            logger.info(f"PlannerAgent: Initialized with structured output: {model_identifier}")
        except Exception as e:
            logger.warning(f"PlannerAgent: Structured output setup failed ({e}), using fallback parser.")
            self.structured_llm = None

    @traceable(name="Planner")
    def invoke(self, state: AgentState) -> Dict[str, Any]:
        dump_agent_state(state, "PlannerAgent")

        query = state.get("query", "").strip()
        if not query:
            return {
                "route": "conversational",
                "subqueries": [],
                "audit_log": [{
                    "step": "PlannerAgent",
                    "status": "Skipped",
                    "reason": "Empty query"
                }]
            }

        try:
            logger.info("PlannerAgent: Classifying query...")
            from agents.prompt import PLANNER_PROMPT, PLANNER_FALLBACK_PROMPT
            if self.structured_llm:
                prompt = PLANNER_PROMPT
                chain = prompt | self.structured_llm
                result = chain.invoke({"query": query})
                route = result.route
                subqueries = result.subqueries
            else:
                # Fallback mode
                prompt = PLANNER_FALLBACK_PROMPT
                parser = JsonOutputParser(pydantic_object=PlannerOutput)
                chain = prompt | self.llm | parser
                result = chain.invoke({"query": query})
                route = result.get("route", "single_hop")
                subqueries = result.get("subqueries", [])

            # Validation
            route = route.lower().strip()
            if route not in ["conversational", "single_hop", "multi_hop"]:
                route = "single_hop"

            if route == "conversational":
                subqueries = []
            elif route == "single_hop" and not subqueries:
                subqueries = [query]
            elif route == "multi_hop" and len(subqueries) < 2:
                # Fallback to single_hop if decomposition didn't produce multiple queries
                route = "single_hop"
                if not subqueries:
                    subqueries = [query]

            logger.info(f"PlannerAgent: Routed as '{route}' with subqueries: {subqueries}")

            log_agent_step(
                state=state,
                step_name="PlannerAgent",
                status="Success",
                route=route,
                subqueries=subqueries
            )

            return {
                "route": route,
                "subqueries": subqueries,
                "audit_log": [{
                    "step": "PlannerAgent",
                    "status": "Success",
                    "route": route,
                    "subqueries": subqueries
                }]
            }

        except Exception as e:
            logger.error(f"PlannerAgent Error: {e}")
            route = self._fallback_classification(query)
            subqueries = [query] if route != "conversational" else []

            log_agent_step(
                state=state,
                step_name="PlannerAgent",
                status="Success",
                route=route,
                subqueries=subqueries,
                method="fallback",
                error=str(e)
            )

            return {
                "route": route,
                "subqueries": subqueries,
                "audit_log": [{
                    "step": "PlannerAgent",
                    "status": "Success",
                    "route": route,
                    "subqueries": subqueries,
                    "method": "fallback",
                    "error": str(e)
                }]
            }

    def _fallback_classification(self, query: str) -> str:
        query_lower = query.lower()
        greetings = ['hello', 'hi', 'hey', 'thanks', 'thank', 'bye', 'help']
        for greeting in greetings:
            if greeting in query_lower:
                return "conversational"
        if len(query.split()) < 3 and '?' in query:
            return "conversational"
        return "single_hop"


def conversational_reply(state: AgentState) -> Dict[str, Any]:
    """Generates a friendly response for conversational queries."""
    dump_agent_state(state, "ConversationalReplyNode")
    
    query = state.get("query", "")
    content = (
        "Hello! I am DocuMind AI, your intelligent document analysis assistant. "
        "You can upload PDF documents in the sidebar, and I will help you analyze them, "
        "extract relevant data, and answer any specific questions you have about them."
    )
    
    log_agent_step(state, "ConversationalReplyNode", "Success", query=query)
    
    return {
        "final_response": content,
        "audit_log": [{"step": "ConversationalReplyNode", "status": "Success"}]
    }
