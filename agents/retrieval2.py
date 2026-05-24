import logging
import hashlib
import threading
from core.config import Config
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from core.state import AgentState
from utils.helpers import log_agent_step, dump_agent_state
from core.llm import get_llm
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_cross_encoder_instance = None
_cross_encoder_lock = threading.Lock()
_cross_encoder_load_failed = False
_predict_lock = threading.Lock()


def get_cross_encoder():
    """Process-wide lazy singleton. Returns None if disabled or load failed."""
    global _cross_encoder_instance, _cross_encoder_load_failed

    if not Config.USE_CROSS_ENCODER:
        return None
    if _cross_encoder_load_failed:
        return None
    if _cross_encoder_instance is not None:
        return _cross_encoder_instance

    with _cross_encoder_lock:
        if _cross_encoder_load_failed:
            return None
        if _cross_encoder_instance is not None:
            return _cross_encoder_instance
        try:
            logger.info(
                "Loading CrossEncoder for reranking: %s",
                Config.CROSS_ENCODER_MODEL,
            )
            _cross_encoder_instance = CrossEncoder(Config.CROSS_ENCODER_MODEL)
            logger.info("CrossEncoder ready")
        except Exception as e:
            logger.error("Failed to load CrossEncoder: %s", e)
            _cross_encoder_load_failed = True
            return None

    return _cross_encoder_instance


class SearchPlan(BaseModel):
    sub_queries: List[str] = Field(
        description="1-3 optimized search queries derived from user query"
    )


class RetrievalAgent:
    def __init__(self, vector_store, model_identifier: str = None):
        self.vector_store = vector_store
        model_identifier = model_identifier or Config.MODEL_NAME
        self.llm = get_llm(model_identifier, temperature=0.1)

        try:
            self.structured_llm = self.llm.with_structured_output(SearchPlan)
            logger.info(f"Initialized with structured output: {model_identifier}")
        except NotImplementedError:
            logger.warning("Structured output not supported, fallback mode")
            self.structured_llm = None
        except Exception as e:
            logger.warning(
                "Structured output setup failed (%s: %s), fallback mode",
                type(e).__name__,
                e,
            )
            self.structured_llm = None

    def invoke(self, state: AgentState) -> Dict[str, Any]:
        dump_agent_state(state, "AdvancedRetrievalAgent")

        query = state.get("query", "")
        if not query:
            return {"audit_log": [{"step": "AdvancedRetrievalAgent", "status": "Skipped"}]}

        logger.info(f"AdvancedRetrievalAgent: Query -> '{query}'")

        try:
            search_plan = self._create_search_plan(query)

            docs = self._retrieve_documents(query, search_plan)

            logger.info(f"AdvancedRetrievalAgent: Final docs count {len(docs)}")

            status = "Success"
            metadata_key = "retrieved_count"
            metadata_val = len(docs)

            if not docs:
                status = "Error"
                metadata_key = "error"
                metadata_val = "No documents found"

            log_agent_step(
                state=state,
                step_name="AdvancedRetrievalAgent",
                status=status,
                query=query,
                **{metadata_key: metadata_val},
            )

            return {
                "retrieved_docs": docs,
                "audit_log": [{
                    "step": "AdvancedRetrievalAgent",
                    "status": status,
                    "query": query,
                    metadata_key: metadata_val,
                }],
            }
        except Exception as e:
            logger.error(f"AdvancedRetrievalAgent Error: {e}")
            return {
                "audit_log": [{
                    "step": "AdvancedRetrievalAgent",
                    "status": "Error",
                    "error": str(e),
                }],
            }

    def _create_search_plan(self, query: str) -> SearchPlan:
        """
        Smart query planning:
        - Use LLM to breakdown complex queries
        """
        if len(query.split()) <= 5:
            return SearchPlan(sub_queries=[query])

        if self.structured_llm:
            try:
                prompt = (
                    "Break the user query into 1-3 semantic search queries.\n"
                    "Do not include metadata filters.\n\n"
                    f"Query: {query}"
                )
                plan = self.structured_llm.invoke(prompt)
                plan.sub_queries = list(set([query] + plan.sub_queries))
                return plan

            except Exception as e:
                logger.warning(f"Search plan failed, fallback: {e}")

        return SearchPlan(sub_queries=[query])

    def _retrieve_documents(self, query: str, plan: SearchPlan) -> List[Dict]:
        """
        Retrieval pipeline:
        - hybrid search per sub-query
        - deduplication
        - cross-encoder ranking against original query
        """
        all_docs = []
        seen_hashes = set()

        for sub_q in plan.sub_queries:
            logger.info(f"Sub-query: {sub_q}")

            results = self.vector_store.hybrid_search(
                sub_q,
                k=10,
                filter=None,
            )

            for doc in results:
                content = doc["content"].strip()

                content_hash = hashlib.md5(content.encode()).hexdigest()
                if content_hash in seen_hashes:
                    continue

                seen_hashes.add(content_hash)

                all_docs.append({
                    "content": content,
                    "metadata": doc["metadata"],
                })

        cross_encoder = get_cross_encoder()
        if cross_encoder and all_docs:
            try:
                pairs = [[query, doc["content"]] for doc in all_docs]
                with _predict_lock:
                    scores = cross_encoder.predict(pairs)

                for idx, score in enumerate(scores):
                    all_docs[idx]["score"] = float(score)

                ranked_docs = sorted(
                    all_docs,
                    key=lambda x: x.get("score", -float("inf")),
                    reverse=True,
                )
            except Exception as e:
                logger.error(f"Cross-encoder reranking failed: {e}")
                ranked_docs = all_docs
        else:
            ranked_docs = all_docs

        final_docs = [
            {
                "content": d["content"],
                "metadata": d["metadata"],
            }
            for d in ranked_docs[:5]
        ]

        return final_docs
