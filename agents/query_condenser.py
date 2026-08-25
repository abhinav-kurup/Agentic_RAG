import asyncio
import json
import logging
import re

from langchain_core.messages import HumanMessage
from core.llm import get_llm

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def _history_text(messages, limit: int = 6) -> str:
    lines = []
    for msg in list(messages)[-limit:]:
        content = (getattr(msg, "content", None) or "").replace("\n", " ").strip()
        if not content:
            continue
        lines.append(f"{getattr(msg, 'type', 'msg').upper()}: {content[:360]}")
    return "\n".join(lines)


def _parse_decision(raw: str, original: str) -> tuple[str, bool]:
    blob = (raw or "").strip()
    match = _JSON_BLOCK.search(blob)
    if not match:
        return original, False
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return original, False
    related = data.get("related")
    rewritten = (data.get("query") or "").strip()
    if related is False:
        return original, True
    if related is True and rewritten:
        return rewritten, False
    return original, False


class QueryContextualizer:
    def __init__(self):
        self.llm = get_llm("groq/llama-3.1-8b-instant")

    async def ainvoke(self, state):
        history = state.get("messages", [])[:-1]
        summary = state.get("summary", "")
        query = state["query"]

        if not history and not summary:
            return {"standalone_query": query, "topic_shift": False}

        from agents.planner import PlannerAgent
        if PlannerAgent._is_session_meta(query):
            return {"standalone_query": query, "topic_shift": False}

        history_text = _history_text(history)
        prompt = f"""You decide whether a new user question continues the current conversation or starts a new topic.

RELATED — the user is still talking about the same subject as the recent history:
- pronouns or shortcuts (it, they, that, the other one, what about the second agent)
- asking more about the same paper, system, agents, or entities just discussed
- clarifying a term that was a main subject of the last answer

NEW — a different subject, even if a word happens to appear in history:
- a complete question that makes sense without the previous turn
- switching documents or domains (e.g. after a MAMO/agents paper they ask what nature means)
- when unsure, choose NEW

If RELATED, rewrite into a standalone question using only the history needed to resolve it.
If NEW, copy the user question unchanged. Do not add names, papers, or agents from history.

Return JSON only: {{"related": true or false, "query": "..."}}

Examples:
History: MAMO Task-Execution and Weight-Adaptation agents
Question: What do you mean by nature?
JSON: {{"related": false, "query": "What do you mean by nature?"}}

History: MAMO Task-Execution and Weight-Adaptation agents
Question: what does the other agent do?
JSON: {{"related": true, "query": "What does the Weight-Adaptation agent do in MAMO?"}}

History: explained biodiversity in the nature PDF
Question: how do we protect it?
JSON: {{"related": true, "query": "How do we protect biodiversity?"}}

Summary: {summary or "(none)"}

Recent history:
{history_text or "(none)"}

User question: {query}
"""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            rewritten, topic_shift = _parse_decision(getattr(response, "content", "") or "", query)
        except Exception:
            rewritten, topic_shift = query, False

        logger.info("QueryContextualizer: %r -> %r topic_shift=%s", query, rewritten, topic_shift)
        return {
            "standalone_query": rewritten,
            "topic_shift": topic_shift,
            "audit_log": [{
                "step": "QueryContextualizer",
                "status": "Success",
                "original": query,
                "rewritten": rewritten,
                "topic_shift": topic_shift,
            }],
        }

    def invoke(self, state):
        return asyncio.run(self.ainvoke(state))
