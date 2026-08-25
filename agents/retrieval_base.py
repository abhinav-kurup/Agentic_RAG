import asyncio
import logging
import hashlib
from typing import Dict, Any, List
from langsmith import traceable
import httpx

from core.config import Config
from core.llm import get_llm

logger = logging.getLogger(__name__)

COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"


async def _cohere_rerank_scores(query: str, texts: List[str]) -> List[float]:
    """Score documents with Cohere Rerank v2. Raises on HTTP/API errors."""
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            COHERE_RERANK_URL,
            headers={
                "Authorization": f"Bearer {Config.COHERE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": Config.COHERE_RERANK_MODEL,
                "query": query,
                "documents": texts,
            },
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Cohere rerank {response.status_code}: {response.text[:300]}"
            )
        payload = response.json()
    scores = [0.0] * len(texts)
    for item in payload.get("results") or []:
        idx = item.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(texts):
            continue
        scores[idx] = float(item.get("relevance_score", 0.0))
    return scores


class BaseRetrievalAgent:
    """Base class providing shared hybrid search, RRF fusion, reranking, and diversity selection asynchronously."""

    def __init__(self, vector_store):
        self.vector_store = vector_store
        model_identifier = Config.RETRIEVAL_MODEL
        self.llm = get_llm(model_identifier, temperature=0.1)

    async def _retrieve_single_query(self, query: str, sub_query: str, k: int = 10) -> List[Dict]:
        """Runs vector + keyword hybrid search for a single subquery asynchronously in a thread worker."""
        logger.info("Retrieving for sub-query: %s", sub_query)
        
        # Offload synchronous ChromaDB vectorstore search to thread pool to keep loop non-blocking
        results = await asyncio.to_thread(self.vector_store.hybrid_search, sub_query, k=k, filter=None)

        docs = []
        seen_hashes = set()
        for doc in results:
            content = doc["content"].strip()
            if not content:
                continue
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            print(
                f"    - Retrieved: {doc['metadata'].get('source')} "
                f"(Page {doc['metadata'].get('page_number')}) | Score: {doc.get('score', 0.0):.4f}"
            )

            docs.append({
                "content": content,
                "metadata": doc["metadata"],
                "score": doc.get("score", 0.0),
            })

        return docs[:k]

    @traceable(name="ReciprocalRankFusion")
    def _reciprocal_rank_fusion(
        self, ranked_lists: List[List[Dict]], subqueries: List[str], k: int = 60
    ) -> List[Dict]:
        """Fuses multiple ranked lists of documents using Reciprocal Rank Fusion (RRF)."""
        rrf_scores = {}
        doc_map = {}

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

        sorted_hashes = sorted(rrf_scores.keys(), key=lambda h: rrf_scores[h], reverse=True)
        fused_docs = []
        for h in sorted_hashes:
            doc = doc_map[h]
            doc["rrf_score"] = rrf_scores[h]
            fused_docs.append(doc)

        logger.info(f"RRF fused {len(ranked_lists)} lists into {len(fused_docs)} unique documents")
        return fused_docs

    @traceable(name="RerankDocuments")
    async def _rerank_documents(
        self, query: str, docs: List[Dict], subqueries: List[str], protected_hashes: set = None
    ) -> List[Dict]:
        """Reranks candidate documents using Cohere rerank-v4.0-fast."""
        reranked = False
        if Config.USE_CROSS_ENCODER and docs:
            if not Config.COHERE_API_KEY:
                logger.warning(
                    "Rerank enabled but COHERE_API_KEY is missing; using hybrid scores."
                )
            else:
                try:
                    scores = await _cohere_rerank_scores(
                        query, [doc["content"] for doc in docs]
                    )
                    for doc, score in zip(docs, scores):
                        doc["score"] = score
                    reranked = True
                    logger.info(
                        "Cohere %s reranked %d documents against original query.",
                        Config.COHERE_RERANK_MODEL,
                        len(docs),
                    )
                except Exception as e:
                    logger.error("Cohere reranking failed: %s", e)

        if docs and reranked:
            ce_scores = [d.get("score", 0.0) for d in docs]
            rrf_scores = [
                d.get("rrf_score") if d.get("rrf_score") is not None else d.get("score", 0.0)
                for d in docs
            ]

            ce_min, ce_max = min(ce_scores), max(ce_scores)
            rrf_min, rrf_max = min(rrf_scores), max(rrf_scores)
            ce_range, rrf_range = ce_max - ce_min, rrf_max - rrf_min

            for doc in docs:
                ce_raw = doc.get("score", 0.0)
                rrf_raw = doc.get("rrf_score") if doc.get("rrf_score") is not None else doc.get("score", 0.0)
                ce_norm = (ce_raw - ce_min) / ce_range if ce_range > 0.0 else 1.0
                rrf_norm = (rrf_raw - rrf_min) / rrf_range if rrf_range > 0.0 else 1.0

                doc["ce_score_raw"] = ce_raw
                doc["rrf_score_raw"] = rrf_raw
                doc["score"] = 0.7 * ce_norm + 0.3 * rrf_norm

        if docs:
            docs = sorted(docs, key=lambda x: x.get("score", -float("inf")), reverse=True)
            max_score = docs[0]["score"]
            # Softer than 0.6×max so secondary gold pages are less often dropped
            threshold = max_score * 0.45

            filtered_docs = []
            for d in docs:
                h = hashlib.md5(d["content"].strip().encode()).hexdigest()
                if d["score"] >= threshold or (protected_hashes and h in protected_hashes):
                    filtered_docs.append(d)

            if len(filtered_docs) < 5:
                filtered_docs = docs[:5]

            logger.info(
                f"Filtered candidate pool from {len(docs)} to {len(filtered_docs)} documents "
                f"using relative threshold {threshold:.4f} (best score: {max_score:.4f})"
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
    def _select_diverse_contexts(
        self, reranked_docs: List[Dict], subqueries: List[str], limit: int = 6
    ) -> List[Dict]:
        """Greedily selects chunks using diversity penalty and coverage bonus."""
        selected = []
        candidates = list(reranked_docs)
        page_counts, doc_counts = {}, {}
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

                p_penalty = page_counts.get((source, page), 0) * 0.05
                d_penalty = doc_counts.get(source, 0) * 0.02
                c_bonus = len(matched_subqueries - covered_subqueries) * 0.10

                adj_score = score - p_penalty - d_penalty + c_bonus
                if adj_score > best_adjusted_score:
                    best_adjusted_score = adj_score
                    best_idx = idx

            if best_idx != -1:
                chosen = candidates.pop(best_idx)
                selected.append(chosen)

                meta = chosen.get("metadata", {})
                source = meta.get("source", "")
                page = meta.get("page_number", 0)
                page_counts[(source, page)] = page_counts.get((source, page), 0) + 1
                doc_counts[source] = doc_counts.get(source, 0) + 1
                covered_subqueries.update(chosen.get("subqueries_matched", set()))
            else:
                break

        uncovered = set(subqueries) - covered_subqueries
        for missing_sub_q in uncovered:
            sub_q_candidates = [
                d for d in reranked_docs if missing_sub_q in d.get("subqueries_matched", set())
            ]
            if sub_q_candidates:
                best_missing_doc = sub_q_candidates[0]
                if best_missing_doc not in selected:
                    if len(selected) < limit:
                        selected.append(best_missing_doc)
                    else:
                        selected.sort(key=lambda x: x.get("score", 0.0))
                        selected[0] = best_missing_doc
                    covered_subqueries.update(best_missing_doc.get("subqueries_matched", set()))

        selected.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return selected

    async def _post_process_and_select(
        self, query: str, candidate_docs: List[Dict], ranked_lists: List[List[Dict]], ordered_subqueries: List[str]
    ) -> List[Dict]:
        """Unified async post-processing: top-1 protection, reranking, diversity selection, and JSON formatting."""
        print(f"[Retrieval Merging] Total merged candidate documents before reranking: {len(candidate_docs)}")
        print("[Retrieval Output] Retrieved documents before rerank:")
        for idx, doc in enumerate(candidate_docs):
            print(
                f"  {idx + 1}. Source: {doc['metadata'].get('source')} (Page {doc['metadata'].get('page_number')}) "
                f"| Retrieval Score: {doc.get('score', 0.0):.4f} | RRF Score: {doc.get('rrf_score', 0.0):.4f}"
            )

        protected_hashes = {
            hashlib.md5(r_list[0]["content"].strip().encode()).hexdigest()
            for r_list in ranked_lists if r_list
        }

        reranked_docs = await self._rerank_documents(
            query, candidate_docs, ordered_subqueries, protected_hashes=protected_hashes
        )

        print("[Reranking Scores] Documents after Cohere reranking:")
        for idx, doc in enumerate(reranked_docs):
            print(
                f"  {idx + 1}. Source: {doc['metadata'].get('source')} (Page {doc['metadata'].get('page_number')}) "
                f"| Score: {doc.get('score')} | RRF: {doc.get('rrf_score')}"
            )

        final_docs = self._select_diverse_contexts(reranked_docs, ordered_subqueries, limit=6)

        for d in final_docs:
            if "subqueries_matched" in d and isinstance(d["subqueries_matched"], set):
                d["subqueries_matched"] = sorted(list(d["subqueries_matched"]))

        print(f"[Retrieval Output] Top {len(final_docs)} documents sent to AnalysisAgent:")
        for idx, doc in enumerate(final_docs):
            print(
                f"  {idx + 1}. Source: {doc['metadata'].get('source')} (Page {doc['metadata'].get('page_number')}) "
                f"| Score: {doc.get('score')} | RRF: {doc.get('rrf_score')} | Subqueries: {doc.get('subqueries_matched')}"
            )

        return final_docs
