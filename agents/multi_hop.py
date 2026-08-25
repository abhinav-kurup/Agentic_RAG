import asyncio
import logging
from typing import Dict, Any
from langsmith import traceable

from core.state import AgentState
from utils.helpers import log_agent_step, dump_agent_state
from agents.retrieval_base import BaseRetrievalAgent

logger = logging.getLogger(__name__)


class MultiHopRetrievalAgent(BaseRetrievalAgent):
    """Executes multi-hop parallel retrieval with Reciprocal Rank Fusion using asyncio.gather."""

    @traceable(name="MultiHopRetrieval")
    async def ainvoke(self, state: AgentState) -> Dict[str, Any]:
        dump_agent_state(state, "MultiHopRetrievalAgent")

        query = state.get("standalone_query") or state.get("query", "")
        if not query:
            return {"audit_log": [{"step": "MultiHopRetrievalAgent", "status": "Skipped", "reason": "Empty query"}]}

        subqueries = state.get("subqueries", [])
        # Dedupe while preserving order (set() was nondeterministic)
        active_subqueries = list(dict.fromkeys([query] + list(subqueries or [])))
        k = max(8, min(15, 40 // len(active_subqueries)))

        print(f"[Retrieval Route] Multi-hop active subqueries: {active_subqueries} (k={k})")

        retrieval_results = []
        subquery_to_docs = {}

        # Native asyncio.gather for parallel subquery retrieval on the event loop
        tasks = [self._retrieve_single_query(query, sub_q, k) for sub_q in active_subqueries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for sub_q, docs_or_exc in zip(active_subqueries, results):
            if isinstance(docs_or_exc, Exception):
                logger.error(f"Subquery '{sub_q}' error: {docs_or_exc}")
                retrieval_results.append({"subquery": sub_q, "documents": [], "status": "not_found"})
                subquery_to_docs[sub_q] = []
            else:
                retrieval_results.append({
                    "subquery": sub_q,
                    "documents": docs_or_exc,
                    "status": "found" if docs_or_exc else "not_found"
                })
                print(f"  -> Subquery '{sub_q}' retrieved {len(docs_or_exc)} candidate chunks")
                subquery_to_docs[sub_q] = docs_or_exc

        ordered_active_subqueries = [sub_q for sub_q in active_subqueries if sub_q in subquery_to_docs]
        ranked_lists = [subquery_to_docs[sub_q] for sub_q in ordered_active_subqueries]

        # RRF Fusion
        candidate_docs = self._reciprocal_rank_fusion(ranked_lists, ordered_active_subqueries)

        try:
            final_docs = await self._post_process_and_select(query, candidate_docs, ranked_lists, ordered_active_subqueries)
            logger.info(f"MultiHopRetrievalAgent: Final docs count {len(final_docs)}")
            log_agent_step(
                state=state,
                step_name="MultiHopRetrievalAgent",
                status="Success",
                query=query,
                subqueries=ordered_active_subqueries,
                retrieved_count=len(final_docs),
                strategy="multi_hop",
            )
            return {
                "retrieved_docs": final_docs,
                "retrieval_results": retrieval_results,
                "audit_log": [{
                    "step": "MultiHopRetrievalAgent",
                    "status": "Success",
                    "query": query,
                    "subqueries": ordered_active_subqueries,
                    "retrieved_count": len(final_docs),
                    "strategy": "multi_hop",
                }],
            }
        except Exception as e:
            logger.error(f"MultiHopRetrievalAgent Error: {e}")
            return {"audit_log": [{"step": "MultiHopRetrievalAgent", "status": "Error", "error": str(e)}]}

    def invoke(self, state: AgentState) -> Dict[str, Any]:
        return asyncio.run(self.ainvoke(state))
