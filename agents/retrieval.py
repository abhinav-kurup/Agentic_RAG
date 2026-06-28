import logging
import hashlib
import threading
from langsmith import traceable
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


class BaseRetrievalAgent:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        model_identifier = Config.RETRIEVAL_MODEL
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

    @traceable(name="ReciprocalRankFusion")
    def _reciprocal_rank_fusion(self, ranked_lists: List[List[Dict]], subqueries: List[str], k: int = 60) -> List[Dict]:
        """Fuses multiple ranked lists of documents using Reciprocal Rank Fusion (RRF), tracking subquery coverage."""
        rrf_scores = {}  # content_hash -> float
        doc_map = {}     # content_hash -> dict (document representation)

        for sub_q, ranked_list in zip(subqueries, ranked_lists):
            for rank, doc in enumerate(ranked_list, start=1):
                content = doc.get("content", "").strip()
                if not content:
                    continue
                content_hash = hashlib.md5(content.encode()).hexdigest()
                
                score = 1.0 / (k + rank)
                rrf_scores[content_hash] = rrf_scores.get(content_hash, 0.0) + score
                
                if content_hash not in doc_map:
                    doc_copy = dict(doc)
                    doc_copy["subqueries_matched"] = {sub_q}
                    doc_map[content_hash] = doc_copy
                else:
                    doc_map[content_hash]["subqueries_matched"].add(sub_q)

        # Sort documents by their fused RRF score
        sorted_hashes = sorted(rrf_scores.keys(), key=lambda h: rrf_scores[h], reverse=True)
        
        fused_docs = []
        for h in sorted_hashes:
            doc = doc_map[h]
            doc["rrf_score"] = rrf_scores[h]
            fused_docs.append(doc)
            
        logger.info(f"RRF fused {len(ranked_lists)} lists into {len(fused_docs)} unique documents")
        return fused_docs

    @traceable(name="RerankDocuments")
    def _rerank_documents(self, query: str, docs: List[Dict], subqueries: List[str], protected_hashes: set = None) -> List[Dict]:
        """Reranks the list of documents against the matched subqueries, combines with RRF score, and filters weak candidates."""
        cross_encoder = get_cross_encoder()
        
        # 1. Subquery Reranking Setup (Scoring against original query)
        pairs = [[query, doc["content"]] for doc in docs]

        if cross_encoder and pairs:
            try:
                with _predict_lock:
                    scores = cross_encoder.predict(pairs)
                
                for doc_idx, doc in enumerate(docs):
                    doc["score"] = float(scores[doc_idx])
                logger.info(f"Cross-encoder reranked {len(docs)} documents against original query.")
            except Exception as e:
                logger.error(f"Cross-encoder reranking failed: {e}")
                for doc in docs:
                    doc["score"] = 0.0
        else:
            for doc in docs:
                doc["score"] = 0.0

        if docs:
            # 2. Score Normalization & Weight Fusion (0.7 CE + 0.3 RRF/Retrieval)
            ce_scores = [d.get("score", 0.0) for d in docs]
            rrf_scores = [d.get("rrf_score") if d.get("rrf_score") is not None else d.get("score", 0.0) for d in docs]
            
            ce_min, ce_max = min(ce_scores), max(ce_scores)
            rrf_min, rrf_max = min(rrf_scores), max(rrf_scores)
            
            ce_range = ce_max - ce_min
            rrf_range = rrf_max - rrf_min
            
            for doc in docs:
                ce_raw = doc.get("score", 0.0)
                rrf_raw = doc.get("rrf_score") if doc.get("rrf_score") is not None else doc.get("score", 0.0)
                
                ce_norm = (ce_raw - ce_min) / ce_range if ce_range > 0.0 else 1.0
                rrf_norm = (rrf_raw - rrf_min) / rrf_range if rrf_range > 0.0 else 1.0
                
                combined_score = 0.7 * ce_norm + 0.3 * rrf_norm
                doc["ce_score_raw"] = ce_raw
                doc["rrf_score_raw"] = rrf_raw
                doc["score"] = combined_score

            # 3. Sort & Relative Filtering Stage (threshold = max_score * 0.6 with safety floor & top-1 protection)
            docs = sorted(
                docs,
                key=lambda x: x.get("score", -float("inf")),
                reverse=True,
            )
            
            max_score = docs[0]["score"]
            threshold = max_score * 0.6
            
            filtered_docs = []
            for d in docs:
                h = hashlib.md5(d["content"].strip().encode()).hexdigest()
                is_protected = protected_hashes and h in protected_hashes
                if d["score"] >= threshold or is_protected:
                    filtered_docs.append(d)
                    
            # Safety floor: always keep at least 3 documents
            if len(filtered_docs) < 3:
                filtered_docs = docs[:3]
            logger.info(
                f"Filtered candidate pool from {len(docs)} to {len(filtered_docs)} documents "
                f"using relative threshold {threshold:.4f} (best combined score: {max_score:.4f})"
            )
            docs = filtered_docs

        return [
            {
                "content": d["content"],
                "metadata": d["metadata"],
                "score": d.get("score"),
                "rrf_score": d.get("rrf_score"),
                "ce_score_raw": d.get("ce_score_raw"),
                "rrf_score_raw": d.get("rrf_score_raw"),
                "subqueries_matched": d.get("subqueries_matched", set()),
            }
            for d in docs
        ]

    @traceable(name="SelectDiverseContexts")
    def _select_diverse_contexts(self, reranked_docs: List[Dict], subqueries: List[str], limit: int = 5) -> List[Dict]:
        """Greedily selects chunks using a soft diversity penalty and coverage bonus, validating evidence completeness."""
        selected = []
        candidates = list(reranked_docs)
        
        page_counts = {}  # (source, page) -> count
        doc_counts = {}   # source -> count
        covered_subqueries = set()
        
        for _ in range(limit):
            if not candidates:
                break
                
            best_idx = -1
            best_adjusted_score = -float("inf")
            
            for idx, doc in enumerate(candidates):
                score = doc.get("score", 0.0)
                meta = doc.get("metadata", {})
                source = meta.get("source", "")
                page = meta.get("page_number", 0)
                matched_subqueries = doc.get("subqueries_matched", set())
                
                # Page & Document Repetitive Penalties
                page_key = (source, page)
                p_penalty = page_counts.get(page_key, 0) * 0.05
                d_penalty = doc_counts.get(source, 0) * 0.02
                
                # Coverage Bonus
                new_covers = matched_subqueries - covered_subqueries
                c_bonus = len(new_covers) * 0.10
                
                adj_score = score - p_penalty - d_penalty + c_bonus
                
                if adj_score > best_adjusted_score:
                    best_adjusted_score = adj_score
                    best_idx = idx
                    
            if best_idx != -1:
                chosen = candidates.pop(best_idx)
                selected.append(chosen)
                
                # Update trackers
                meta = chosen.get("metadata", {})
                source = meta.get("source", "")
                page = meta.get("page_number", 0)
                page_key = (source, page)
                page_counts[page_key] = page_counts.get(page_key, 0) + 1
                doc_counts[source] = doc_counts.get(source, 0) + 1
                
                matched_subqueries = chosen.get("subqueries_matched", set())
                covered_subqueries.update(matched_subqueries)
            else:
                break
        
        # Evidence Completeness Validation Swap
        all_subqueries = set(subqueries)
        uncovered = all_subqueries - covered_subqueries
        
        for missing_sub_q in uncovered:
            # Find candidate matching this subquery
            sub_q_candidates = [d for d in reranked_docs if missing_sub_q in d.get("subqueries_matched", set())]
            if sub_q_candidates:
                best_missing_doc = sub_q_candidates[0]
                if best_missing_doc not in selected:
                    if len(selected) < limit:
                        selected.append(best_missing_doc)
                    else:
                        selected.sort(key=lambda x: x.get("score", 0.0))
                        selected[0] = best_missing_doc
                    # Always update coverage regardless of which branch was taken
                    covered_subqueries.update(best_missing_doc.get("subqueries_matched", set()))
                        
        selected.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return selected

    @traceable(name="RetrieveSingleQuery")
    def _retrieve_single_query(self, query: str, sub_query: str, k: int = 10) -> List[Dict]:
        """Runs the standard retrieval pipeline for a single subquery without reranking."""
        logger.info(f"Retrieving for sub-query: {sub_query}")
        results = self.vector_store.hybrid_search(
            sub_query,
            k=k,
            filter=None,
        )
        
        docs = []
        seen_hashes = set()
        for doc in results:
            content = doc["content"].strip()
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            
            # Print statement for retrieved doc with score
            print(f"    - Retrieved: {doc['metadata'].get('source')} (Page {doc['metadata'].get('page_number')}) | Score: {doc.get('score', 0.0):.4f}")
            
            docs.append({
                "content": content,
                "metadata": doc["metadata"],
                "score": doc.get("score", 0.0),
            })
            
        return docs[:k]

    @traceable(name="CreateSearchPlan")
    def _create_search_plan(self, query: str) -> SearchPlan:
        """Smart query planning: fallback/legacy helper"""
        if len(query.split()) <= 5:
            return SearchPlan(sub_queries=[query])

        if self.structured_llm:
            try:
                from agents.prompt import RETRIEVAL_DECOMPOSER_PROMPT
                formatted_prompt = RETRIEVAL_DECOMPOSER_PROMPT.format(query=query)
                plan = self.structured_llm.invoke(formatted_prompt)
                plan.sub_queries = list(set([query] + plan.sub_queries))
                return plan

            except Exception as e:
                logger.warning(f"Search plan failed, fallback: {e}")

        return SearchPlan(sub_queries=[query])

    @traceable(name="RetrieveDocumentsLegacy")
    def _retrieve_documents(self, query: str, plan: SearchPlan) -> List[Dict]:
        """Retrieval pipeline: legacy helper"""
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


class SingleHopRetrievalAgent(BaseRetrievalAgent):
    @traceable(name="SingleHopRetrieval")
    def invoke(self, state: AgentState) -> Dict[str, Any]:
        dump_agent_state(state, "SingleHopRetrievalAgent")

        query = state.get("query", "")
        if not query:
            return {"audit_log": [{"step": "AdvancedRetrievalAgent", "status": "Skipped"}]}

        subqueries = state.get("subqueries", [])
        logger.info(f"SingleHopRetrievalAgent: Subqueries -> {subqueries}")

        try:
            retrieval_results = []
            ranked_lists = []
            
            active_queries = subqueries if subqueries else [query]
            k = 8
            
            print(f"[Retrieval Route] Processing queries: {active_queries} (k={k})")
            ordered_active_subqueries = list(active_queries)
            for sub_q in ordered_active_subqueries:
                docs = self._retrieve_single_query(query, sub_q, k)
                status = "found" if docs else "not_found"
                retrieval_results.append({
                    "subquery": sub_q,
                    "documents": docs,
                    "status": status
                })
                print(f"  -> Query/Subquery '{sub_q}' retrieved {len(docs)} candidate chunks")
                ranked_lists.append(docs)

            # Bypassing RRF for single-hop queries: just combine all retrieved documents without RRF scoring
            final_docs = []
            seen_hashes = set()
            for r_list in ranked_lists:
                for doc in r_list:
                    h = hashlib.md5(doc["content"].strip().encode()).hexdigest()
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        doc["subqueries_matched"] = {query} # Set default matched subquery
                        final_docs.append(doc)

            print(f"[Retrieval Merging] Total merged candidate documents before reranking: {len(final_docs)}")
            
            # Print statement for retrieved documents before cross-encoder with scores
            print(f"[Retrieval Output] Retrieved documents before cross-encoder:")
            for idx, doc in enumerate(final_docs):
                print(f"  {idx + 1}. Source: {doc['metadata'].get('source')} (Page {doc['metadata'].get('page_number')}) | Retrieval Score: {doc.get('score', 0.0):.4f} | RRF Score: {doc.get('rrf_score', 0.0):.4f}")

            # Run global reranking on all merged candidate documents against original/sub-queries (with top-1 retrieved protection)
            protected_hashes = set()
            for r_list in ranked_lists:
                if r_list:
                    top_doc = r_list[0]
                    h = hashlib.md5(top_doc["content"].strip().encode()).hexdigest()
                    protected_hashes.add(h)
            final_docs = self._rerank_documents(query, final_docs, ordered_active_subqueries, protected_hashes=protected_hashes)
            
            print(f"[Reranking Scores] Documents after Cross-Encoder reranking:")
            for idx, doc in enumerate(final_docs):
                print(f"  {idx + 1}. Source: {doc['metadata'].get('source')} (Page {doc['metadata'].get('page_number')}) | Score: {doc.get('score')} | RRF: {doc.get('rrf_score')}")
            
            # Select diverse and coverage-complete contexts
            final_docs = self._select_diverse_contexts(final_docs, ordered_active_subqueries, limit=5)
            
            # Convert subqueries_matched from set to list for JSON serialization compatibility
            for d in final_docs:
                if "subqueries_matched" in d and isinstance(d["subqueries_matched"], set):
                    d["subqueries_matched"] = sorted(list(d["subqueries_matched"]))
                    
            print(f"[Retrieval Output] Top {len(final_docs)} documents sent to AnalysisAgent:")
            for idx, doc in enumerate(final_docs):
                print(f"  {idx + 1}. Source: {doc['metadata'].get('source')} (Page {doc['metadata'].get('page_number')}) | Score: {doc.get('score')} | RRF: {doc.get('rrf_score')} | Subqueries: {doc.get('subqueries_matched')}")

            logger.info(f"AdvancedRetrievalAgent (SingleHop): Final merged docs count {len(final_docs)}")

            status_str = "Success"
            log_agent_step(
                state=state,
                step_name="AdvancedRetrievalAgent",
                status=status_str,
                query=query,
                retrieved_count=len(final_docs),
            )

            return {
                "retrieved_docs": final_docs,
                "retrieval_results": retrieval_results,
                "audit_log": [{
                    "step": "AdvancedRetrievalAgent",
                    "status": status_str,
                    "query": query,
                    "retrieved_count": len(final_docs),
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


class MultiHopRetrievalAgent(BaseRetrievalAgent):
    @traceable(name="MultiHopRetrieval")
    def invoke(self, state: AgentState) -> Dict[str, Any]:
        dump_agent_state(state, "MultiHopRetrievalAgent")

        query = state.get("query", "")
        if not query:
            return {"audit_log": [{"step": "AdvancedRetrievalAgent", "status": "Skipped"}]}

        subqueries = state.get("subqueries", [])
        logger.info(f"MultiHopRetrievalAgent: Subqueries -> {subqueries}")

        try:
            retrieval_results = []
            ranked_lists = []
            ordered_active_subqueries = []

            # Include original query to ensure broad semantic recall alongside specific subqueries
            active_subqueries = list(set([query] + subqueries))
            
            # Adaptive retrieval size (Bug 4 Fix)
            num_queries = len(active_subqueries)
            k = max(8, min(15, 40 // num_queries))
            
            print(f"[Retrieval Route] Multi-hop active subqueries: {active_subqueries} (k={k})")
            
            subquery_to_docs = {}
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_subquery = {
                    executor.submit(self._retrieve_single_query, query, sub_q, k): sub_q
                    for sub_q in active_subqueries
                }
                
                for future in concurrent.futures.as_completed(future_to_subquery):
                    sub_q = future_to_subquery[future]
                    try:
                        docs = future.result()
                        status = "found" if docs else "not_found"
                        retrieval_results.append({
                            "subquery": sub_q,
                            "documents": docs,
                            "status": status
                        })
                        print(f"  -> Subquery '{sub_q}' retrieved {len(docs)} candidate chunks")
                        subquery_to_docs[sub_q] = docs
                    except Exception as exc:
                        logger.error(f"Subquery '{sub_q}' generated an exception: {exc}")
                        retrieval_results.append({
                            "subquery": sub_q,
                            "documents": [],
                            "status": "not_found"
                        })
                        subquery_to_docs[sub_q] = []
                        
            # Reconstruct in deterministic order to align with subqueries parameter
            for sub_q in active_subqueries:
                if sub_q in subquery_to_docs:
                    ordered_active_subqueries.append(sub_q)
                    ranked_lists.append(subquery_to_docs[sub_q])

            # Fuse ranked lists using Reciprocal Rank Fusion (RRF)
            final_docs = self._reciprocal_rank_fusion(ranked_lists, ordered_active_subqueries)

            print(f"[Retrieval Merging] Total merged candidate documents before reranking: {len(final_docs)}")
            
            # Print statement for retrieved documents before cross-encoder with scores
            print(f"[Retrieval Output] Retrieved documents before cross-encoder:")
            for idx, doc in enumerate(final_docs):
                print(f"  {idx + 1}. Source: {doc['metadata'].get('source')} (Page {doc['metadata'].get('page_number')}) | Retrieval Score: {doc.get('score', 0.0):.4f} | RRF Score: {doc.get('rrf_score', 0.0):.4f}")

            # Run global reranking on all merged candidate documents against original/sub-queries (with top-1 retrieved protection)
            protected_hashes = set()
            for r_list in ranked_lists:
                if r_list:
                    top_doc = r_list[0]
                    h = hashlib.md5(top_doc["content"].strip().encode()).hexdigest()
                    protected_hashes.add(h)
            final_docs = self._rerank_documents(query, final_docs, ordered_active_subqueries, protected_hashes=protected_hashes)
            
            print(f"[Reranking Scores] Documents after Cross-Encoder reranking:")
            for idx, doc in enumerate(final_docs):
                print(f"  {idx + 1}. Source: {doc['metadata'].get('source')} (Page {doc['metadata'].get('page_number')}) | Score: {doc.get('score')} | RRF: {doc.get('rrf_score')}")
            
            # Select diverse and coverage-complete contexts
            final_docs = self._select_diverse_contexts(final_docs, ordered_active_subqueries, limit=5)
            
            # Convert subqueries_matched from set to list for JSON serialization compatibility
            for d in final_docs:
                if "subqueries_matched" in d and isinstance(d["subqueries_matched"], set):
                    d["subqueries_matched"] = sorted(list(d["subqueries_matched"]))
                    
            print(f"[Retrieval Output] Top {len(final_docs)} documents sent to AnalysisAgent:")
            for idx, doc in enumerate(final_docs):
                print(f"  {idx + 1}. Source: {doc['metadata'].get('source')} (Page {doc['metadata'].get('page_number')}) | Score: {doc.get('score')} | RRF: {doc.get('rrf_score')} | Subqueries: {doc.get('subqueries_matched')}")

            logger.info(f"AdvancedRetrievalAgent (MultiHop): Final merged docs count {len(final_docs)}")

            status_str = "Success"
            log_agent_step(
                state=state,
                step_name="AdvancedRetrievalAgent",
                status=status_str,
                query=query,
                retrieved_count=len(final_docs),
            )

            return {
                "retrieved_docs": final_docs,
                "retrieval_results": retrieval_results,
                "audit_log": [{
                    "step": "AdvancedRetrievalAgent",
                    "status": status_str,
                    "query": query,
                    "retrieved_count": len(final_docs),
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


class RetrievalAgent(BaseRetrievalAgent):
    @traceable(name="Retrieval")
    def invoke(self, state: AgentState) -> Dict[str, Any]:
        route = state.get("route", "single_hop")
        if route == "multi_hop":
            agent = MultiHopRetrievalAgent(self.vector_store)
        else:
            agent = SingleHopRetrievalAgent(self.vector_store)
        return agent.invoke(state)
