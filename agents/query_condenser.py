import asyncio
from langchain_core.messages import HumanMessage
from core.llm import get_llm

class QueryContextualizer:
    def __init__(self):
        self.llm = get_llm("groq/llama-3.1-8b-instant")

    async def ainvoke(self, state):
        history = state.get("messages", [])[:-1]   # exclude the just-added current query
        summary = state.get("summary", "")
        query = state["query"]

        if not history and not summary:
            # Nothing to resolve against — first turn, pass through unchanged
            return {"standalone_query": query}

        history_text = "\n".join(f"{m.type.upper()}: {m.content}" for m in history)

        prompt = f"""Given the conversation summary and recent history below, rewrite the 
follow-up question into a standalone question containing all necessary context. 
If it's already standalone, return it unchanged. Return ONLY the question, nothing else.

Summary: {summary or "(none)"}

Recent history:
{history_text or "(none)"}

Follow-up question: {query}

Standalone question:"""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            rewritten = response.content.strip()
        except Exception:
            rewritten = query   # fail safe — fall back to original on any error

        return {
            "standalone_query": rewritten,
            "audit_log": [{"step": "QueryContextualizer", "status": "Success",
                           "original": query, "rewritten": rewritten}],
        }

    def invoke(self, state):
        return asyncio.run(self.ainvoke(state))