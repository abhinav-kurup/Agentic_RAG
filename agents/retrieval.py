"""
Retrieval module facade.
Re-exports BaseRetrievalAgent, SingleHopRetrievalAgent, MultiHopRetrievalAgent, and RetrievalAgent
for backward compatibility across the codebase.
"""
import asyncio
from typing import Dict, Any
from langsmith import traceable

from core.state import AgentState
from agents.retrieval_base import BaseRetrievalAgent
from agents.single_hop import SingleHopRetrievalAgent
from agents.multi_hop import MultiHopRetrievalAgent


class RetrievalAgent(BaseRetrievalAgent):
    """Facade for dispatching state asynchronously to SingleHop or MultiHop agents."""

    @traceable(name="Retrieval")
    async def ainvoke(self, state: AgentState) -> Dict[str, Any]:
        route = state.get("route", "single_hop")
        if route == "multi_hop":
            agent = MultiHopRetrievalAgent(self.vector_store)
        else:
            agent = SingleHopRetrievalAgent(self.vector_store)
        return await agent.ainvoke(state)

    def invoke(self, state: AgentState) -> Dict[str, Any]:
        return asyncio.run(self.ainvoke(state))
