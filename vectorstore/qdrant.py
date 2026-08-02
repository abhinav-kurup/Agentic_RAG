import os
import uuid
import logging
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    Distance,
    Modifier,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Prefetch,
    FusionQuery,
    Fusion,
)
from fastembed import SparseTextEmbedding
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import Config

logger = logging.getLogger(__name__)


class QdrantVectorStoreManager:
    """
    Manager for Qdrant Vector Database supporting 100% Native Server-Side
    Hybrid Search (Dense BAAI/bge-m3 + Sparse Qdrant/bm25 with RRF Fusion).
    Zero local disk files or Python RAM index required.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.url = url or Config.QDRANT_URL
        self.collection_name = collection_name or Config.QDRANT_COLLECTION_NAME

        logger.info(f"Initializing QdrantVectorStoreManager connected to server at {self.url}")
        try:
            self.client = QdrantClient(url=self.url)
            self.client.get_collections()
            logger.info(f"Successfully connected to Qdrant server at {self.url}")
        except Exception as conn_err:
            logger.error(f"Failed to connect to Qdrant server at {self.url}: {conn_err}")
            raise ConnectionError(
                f"Could not connect to Qdrant server at {self.url}. "
                "Please ensure the Qdrant service/container is running on port 6333."
            ) from conn_err

        # Initialize Embedders
        logger.info("Initializing Dense Embedder (BAAI/bge-m3)...")
        self.dense_embedder = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)

        logger.info("Initializing Sparse Embedder (Qdrant/bm25)...")
        self.sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25")

        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Create Qdrant native hybrid collection if it does not already exist."""
        try:
            if not self.client.collection_exists(self.collection_name):
                logger.info(
                    f"Creating Qdrant native hybrid collection '{self.collection_name}' "
                    "with Dense (1024-dim Cosine) and Sparse (Qdrant/bm25 IDF) vector configurations."
                )
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(size=1024, distance=Distance.COSINE)
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(modifier=Modifier.IDF)
                    },
                )
        except Exception as e:
            logger.error(f"Error checking/creating Qdrant collection: {e}")
            raise e



    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Embeds chunks using Dense (BAAI/bge-m3) and Sparse (Qdrant/bm25) models
        and upserts PointStruct points directly to Qdrant server.
        """
        if not chunks:
            return

        texts = [c["text"] for c in chunks]

        # Compute Dense & Sparse Vectors
        logger.info(f"Computing Dense & Sparse embeddings for {len(chunks)} chunks...")
        dense_vecs = self.dense_embedder.embed_documents(texts)
        sparse_objs = list(self.sparse_embedder.embed(texts))

        points = []
        for c, dvec, sobj in zip(chunks, dense_vecs, sparse_objs):
            chunk_id = c.get("id") or str(uuid.uuid4())
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk_id)))

            svec = models.SparseVector(
                indices=sobj.indices.tolist(),
                values=sobj.values.tolist()
            )

            payload = {
                "chunk_id": chunk_id,
                "content": c["text"],
                "doc_id": c.get("doc_id", ""),
                "page_number": c.get("page_number", 0),
                "chunk_index": c.get("chunk_index", 0),
                "type": c.get("type", "text"),
                "source": c.get("metadata", {}).get("source", ""),
                **{
                    k: v
                    for k, v in c.get("metadata", {}).items()
                    if isinstance(v, (str, int, float, bool))
                },
            }

            points.append(
                PointStruct(
                    id=point_uuid,
                    vector={
                        "dense": dvec,
                        "sparse": svec,
                    },
                    payload=payload,
                )
            )

        try:
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
            logger.info(f"Upserted {len(chunks)} dual-vector points directly to Qdrant server collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Error adding chunks to Qdrant: {e}")
            raise e

    def delete_by_source(self, source: str) -> int:
        """Delete all Qdrant points matching the source filename."""
        try:
            filter_selector = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            )
            count_res = self.client.count(
                collection_name=self.collection_name, count_filter=filter_selector
            )
            deleted_count = count_res.count

            if deleted_count > 0:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=filter_selector,
                )

            logger.info("Deleted %d points for source '%s' from Qdrant server.", deleted_count, source)
            return deleted_count
        except Exception as e:
            logger.error(f"Error deleting points by source '{source}': {e}")
            raise e

    def hybrid_search(self, query: str, k: int = 10, filter: dict = None) -> List[Dict[str, Any]]:
        """
        Executes 100% Native Server-Side Hybrid Search on Qdrant Server using
        Dense Prefetch + Sparse Prefetch + RRF (Reciprocal Rank Fusion).
        """
        try:
            q_dense = self.dense_embedder.embed_query(query)
            q_sparse_obj = list(self.sparse_embedder.embed([query]))[0]
            q_sparse = models.SparseVector(
                indices=q_sparse_obj.indices.tolist(),
                values=q_sparse_obj.values.tolist(),
            )

            qdrant_filter = None
            if filter:
                must_conditions = [
                    FieldCondition(key=k_meta, match=MatchValue(value=v_meta))
                    for k_meta, v_meta in filter.items()
                ]
                qdrant_filter = Filter(must=must_conditions)

            dense_kwargs = {"query": q_dense, "using": "dense", "limit": k * 2}
            sparse_kwargs = {"query": q_sparse, "using": "sparse", "limit": k * 2}

            if qdrant_filter:
                dense_kwargs["filter"] = qdrant_filter
                sparse_kwargs["filter"] = qdrant_filter

            dense_branch = Prefetch(**dense_kwargs)
            sparse_branch = Prefetch(**sparse_kwargs)

            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[dense_branch, sparse_branch],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=k,
                with_payload=True,
            )


            results = []
            for hit in response.points:
                payload = hit.payload or {}
                content = payload.get("content") or payload.get("text") or ""
                results.append({
                    "content": content,
                    "metadata": payload,
                    "score": float(hit.score),
                })
            return results

        except Exception as e:
            logger.error(f"Native Qdrant server hybrid search failed: {e}")
            return []

    def similarity_search(self, query: str, k: int = 5) -> List[Any]:
        results = self.hybrid_search(query, k=k)
        class DocWrapper:
            def __init__(self, content, metadata):
                self.page_content = content
                self.metadata = metadata
        return [DocWrapper(r["content"], r["metadata"]) for r in results]

    def similarity_search_with_score(self, query: str, k: int = 4, filter: dict = None):
        results = self.hybrid_search(query, k=k, filter=filter)
        class DocWrapper:
            def __init__(self, content, metadata):
                self.page_content = content
                self.metadata = metadata
        return [(DocWrapper(r["content"], r["metadata"]), r["score"]) for r in results]

    def _qdrant_chunk_count(self) -> int:
        try:
            res = self.client.count(collection_name=self.collection_name)
            return res.count
        except Exception:
            return 0

    def get_processed_documents(self) -> List[str]:
        """Scroll Qdrant server payloads to list unique document sources."""
        try:
            sources = set()
            offset = None
            while True:
                scroll_res, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=250,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in scroll_res:
                    meta = point.payload or {}
                    if "source" in meta and meta["source"]:
                        sources.add(meta["source"])
                if next_offset is None:
                    break
                offset = next_offset
            return sorted(list(sources))
        except Exception as e:
            logger.error(f"Error fetching processed documents from Qdrant: {e}")
            return []

    def clear_database(self) -> None:
        """Clear Qdrant collection on server."""
        try:
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)
                self._ensure_collection_exists()
            logger.info("Cleared Qdrant collection '%s' on server.", self.collection_name)
        except Exception as e:
            logger.error(f"Error clearing Qdrant collection: {e}")
            raise e
