import asyncio
import logging
import hashlib
from typing import Dict, Any
from langsmith import traceable

from core.state import AgentState
from utils.helpers import log_agent_step, dump_agent_state
from agents.retrieval_base import BaseRetrievalAgent

logger = logging.getLogger(__name__)


class SingleHopRetrievalAgent(BaseRetrievalAgent):
    """Executes single-hop hybrid retrieval pipeline asynchronously."""

    @traceable(name="SingleHopRetrieval")
    async def ainvoke(self, state: AgentState) -> Dict[str, Any]:
        dump_agent_state(state, "SingleHopRetrievalAgent")

        query = state.get("standalone_query") or state.get("query", "")
        if not query:
            return {"audit_log": [{"step": "SingleHopRetrievalAgent", "status": "Skipped", "reason": "Empty query"}]}

        subqueries = state.get("subqueries", [])
        active_queries = subqueries if subqueries else [query]
        k = 8

        print(f"[Retrieval Route] Single-hop queries: {active_queries} (k={k})")
        retrieval_results, ranked_lists, candidate_docs = [], [], []
        seen_hashes = set()

        for sub_q in active_queries:
            docs = await self._retrieve_single_query(query, sub_q, k)
            retrieval_results.append({
                "subquery": sub_q,
                "documents": docs,
                "status": "found" if docs else "not_found"
            })
            print(f"  -> Query '{sub_q}' retrieved {len(docs)} candidate chunks")
            ranked_lists.append(docs)

            for doc in docs:
                h = hashlib.md5(doc["content"].strip().encode()).hexdigest()
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    doc["subqueries_matched"] = {query}
                    candidate_docs.append(doc)

        try:
            final_docs = await self._post_process_and_select(query, candidate_docs, ranked_lists, active_queries)
            logger.info(f"SingleHopRetrievalAgent: Final docs count {len(final_docs)}")
            log_agent_step(
                state=state,
                step_name="SingleHopRetrievalAgent",
                status="Success",
                query=query,
                subqueries=active_queries,
                retrieved_count=len(final_docs),
                strategy="single_hop",
            )
            return {
                "retrieved_docs": final_docs,
                "retrieval_results": retrieval_results,
                "audit_log": [{
                    "step": "SingleHopRetrievalAgent",
                    "status": "Success",
                    "query": query,
                    "subqueries": active_queries,
                    "retrieved_count": len(final_docs),
                    "strategy": "single_hop",
                }],
            }
        except Exception as e:
            logger.error(f"SingleHopRetrievalAgent Error: {e}")
            return {"audit_log": [{"step": "SingleHopRetrievalAgent", "status": "Error", "error": str(e)}]}

    def invoke(self, state: AgentState) -> Dict[str, Any]:
        return asyncio.run(self.ainvoke(state))
