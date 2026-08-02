import asyncio
import logging
import json
import re
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langsmith import traceable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser

from core.state import AgentState
from core.config import Config
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
    Analyzes and classifies queries to plan the retrieval route asynchronously.
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
    async def ainvoke(self, state: AgentState) -> Dict[str, Any]:
        dump_agent_state(state, "PlannerAgent")

        query = state.get("standalone_query") or state.get("query", "")
        query = query.strip()
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
                result = await chain.ainvoke({"query": query})
                route = result.route
                subqueries = result.subqueries
            else:
                prompt = PLANNER_FALLBACK_PROMPT
                parser = JsonOutputParser(pydantic_object=PlannerOutput)
                chain = prompt | self.llm | parser
                result = await chain.ainvoke({"query": query})
                route = result.get("route", "single_hop")
                subqueries = result.get("subqueries", [])

            route = route.lower().strip()
            if route not in ["conversational", "single_hop", "multi_hop"]:
                route = "single_hop"

            if route == "conversational":
                subqueries = []
            elif route == "single_hop" and not subqueries:
                subqueries = [query]
            elif route == "multi_hop" and len(subqueries) < 2:
                route = "single_hop"
                if not subqueries:
                    subqueries = [query]

            logger.info(f"PlannerAgent: Routed as '{route}' with subqueries: {subqueries}")
            log_agent_step(state=state, step_name="PlannerAgent", status="Success", route=route, subqueries=subqueries)

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
            log_agent_step(state=state, step_name="PlannerAgent", status="Success", route=route, subqueries=subqueries, method="fallback", error=str(e))

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

    def invoke(self, state: AgentState) -> Dict[str, Any]:
        return asyncio.run(self.ainvoke(state))

    def _fallback_classification(self, query: str) -> str:
        query_lower = query.lower().strip()
        greetings = ['hello', 'hi', 'hey', 'thanks', 'thank you', 'bye', 'who are you', 'what can you do']
        if query_lower in greetings or any(query_lower == g for g in greetings):
            return "conversational"
        return "single_hop"


async def conversational_reply(state: AgentState) -> Dict[str, Any]:
    """Generates a friendly dynamic response for conversational queries asynchronously."""
    dump_agent_state(state, "ConversationalReplyNode")
    query = state.get("query", "")

    from core.llm import get_llm
    llm = get_llm(Config.PLANNER_MODEL, temperature=0.7)
    
    try:
        response = await llm.ainvoke(
            f"You are DocuMind AI, a helpful document analysis assistant. "
            f"Respond warmly and concisely to this user greeting: '{query}'"
        )
        content = getattr(response, "content", str(response))
    except Exception as e:
        logger.error(f"ConversationalReplyNode LLM error: {e}")
        content = (
            "Hello! I am DocuMind AI, your intelligent document analysis assistant. "
            "You can upload PDF documents in the sidebar, and I will help you analyze them, "
            "extract relevant data, and answer any specific questions you have about them."
        )

    log_agent_step(state, "ConversationalReplyNode", "Success", query=query)
    return {
        "final_response": content,
        "messages": [AIMessage(content=content)],
        "audit_log": [{"step": "ConversationalReplyNode", "status": "Success"}]
    }

