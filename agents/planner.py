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
        raw_query = (state.get("query") or "").strip()
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

        if self._is_session_meta(raw_query) or self._is_session_meta(query):
            log_agent_step(
                state=state,
                step_name="PlannerAgent",
                status="Success",
                route="conversational",
                subqueries=[],
                method="session_meta",
            )
            return {
                "route": "conversational",
                "subqueries": [],
                "audit_log": [{
                    "step": "PlannerAgent",
                    "status": "Success",
                    "route": "conversational",
                    "subqueries": [],
                    "method": "session_meta",
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

            if route == "conversational" and not (
                self._is_session_meta(raw_query)
                or self._is_session_meta(query)
                or self._is_greeting(raw_query)
                or self._is_greeting(query)
            ):
                route = "single_hop"
                subqueries = [query]

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
        if self._is_session_meta(query) or self._is_greeting(query):
            return "conversational"
        return "single_hop"

    @staticmethod
    def _is_greeting(query: str) -> bool:
        q = (query or "").lower().strip().rstrip("!.")
        return q in (
            "hello",
            "hi",
            "hey",
            "thanks",
            "thank you",
            "bye",
            "goodbye",
            "who are you",
            "what can you do",
        )

    @staticmethod
    def _is_session_meta(query: str) -> bool:
        q = (query or "").lower()
        needles = (
            "how many question",
            "how many questions",
            "questions have i asked",
            "questions asked",
            "what did i just ask",
            "what did i ask",
            "my last question",
            "last question i",
            "summarize our conversation",
            "summarise our conversation",
            "what have we talked",
            "what have we discussed",
        )
        return any(n in q for n in needles)


async def conversational_reply(state: AgentState) -> Dict[str, Any]:
    """Answers greetings and questions about this chat session."""
    dump_agent_state(state, "ConversationalReplyNode")
    query = state.get("standalone_query") or state.get("query", "")
    messages = state.get("messages") or []
    user_turns = [
        m for m in messages
        if getattr(m, "type", "") in ("human", "user")
    ]
    question_count = len(user_turns)
    history_lines = []
    for msg in messages[-16:]:
        content = (getattr(msg, "content", None) or "").replace("\n", " ")[:280]
        history_lines.append(f"{getattr(msg, 'type', 'msg')}: {content}")
    history_text = "\n".join(history_lines) or "(no prior turns)"

    from core.llm import get_llm
    llm = get_llm(Config.PLANNER_MODEL, temperature=0.2)

    try:
        response = await llm.ainvoke(
            "You are DocuMind AI. Answer greetings and questions about THIS chat "
            "session only. Do not define terms, people, or topics from general "
            "knowledge, and do not use uploaded PDFs.\n"
            f"User questions in this session (including this one): {question_count}\n"
            f"Recent transcript:\n{history_text}\n\n"
            f"User said: {query}\n\n"
            "If they asked how many questions were asked, the count is "
            f"{question_count}. Do not call them a new or first user. "
            "Be concise."
        )
        content = getattr(response, "content", str(response))
    except Exception as e:
        logger.error("ConversationalReplyNode LLM error: %s", e)
        if PlannerAgent._is_session_meta(query):
            content = (
                f"In this chat you have asked {question_count} question"
                f"{'s' if question_count != 1 else ''} so far, including this one."
            )
        else:
            content = (
                "Hello! I am DocuMind AI. Upload PDFs in the sidebar and ask "
                "questions about them."
            )

    log_agent_step(state, "ConversationalReplyNode", "Success", query=query)
    return {
        "final_response": content,
        "citations": [],
        "retrieved_docs": [],
        "retrieval_results": [],
        "messages": [AIMessage(content=content)],
        "audit_log": [{
            "step": "ConversationalReplyNode",
            "status": "Success",
            "question_count": question_count,
        }]
    }

