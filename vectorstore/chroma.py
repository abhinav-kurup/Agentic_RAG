import pickle
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any, Optional
import os
import logging
from core.config import Config
import numpy as np
import hashlib

logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(self, persist_directory: str = None, collection_name: str = "documind_collection"):
        persist_directory = persist_directory or Config.CHROMA_PERSIST_DIRECTORY
        os.makedirs(persist_directory, exist_ok=True)

        self.persist_directory = persist_directory
        self.collection_name = collection_name

        logger.info(f"Initializing VectorStore in {persist_directory}")

        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self.vectordb = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function,
            collection_name=self.collection_name
        )

        self.bm25_corpus_path = os.path.join(self.persist_directory, "bm25_corpus.pkl")
        self._load_bm25()
        self._sync_bm25_with_chroma_if_needed()

    def _chroma_chunk_count(self) -> int:
        try:
            data = self.vectordb.get(include=[])
            return len(data.get("ids") or [])
        except Exception as e:
            logger.error(f"Failed to count Chroma chunks: {e}")
            return 0

    def _rebuild_bm25_index(self) -> None:
        if self.bm25_corpus:
            tokenized_corpus = [doc["content"].lower().split() for doc in self.bm25_corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def _persist_bm25_corpus(self, corpus: List[Dict[str, Any]]) -> None:
        """Atomically write BM25 pickle and refresh in-memory index."""
        temp_path = f"{self.bm25_corpus_path}.tmp"
        with open(temp_path, "wb") as f:
            pickle.dump(corpus, f)
        os.replace(temp_path, self.bm25_corpus_path)
        self.bm25_corpus = corpus
        self._rebuild_bm25_index()
        logger.info("BM25 corpus persisted (%d chunks).", len(corpus))

    def _load_bm25(self):
        self.bm25_corpus = []
        self.bm25 = None
        if os.path.exists(self.bm25_corpus_path):
            try:
                with open(self.bm25_corpus_path, "rb") as f:
                    self.bm25_corpus = pickle.load(f)
                self._rebuild_bm25_index()
                logger.info(f"Loaded BM25 corpus with {len(self.bm25_corpus)} documents.")
            except Exception as e:
                logger.error(f"Error loading BM25 corpus: {e}")
                self.bm25_corpus = []
                self.bm25 = None

    def _sync_bm25_with_chroma_if_needed(self) -> None:
        chroma_count = self._chroma_chunk_count()
        bm25_count = len(self.bm25_corpus)
        if chroma_count != bm25_count:
            logger.warning(
                "BM25/Chroma count mismatch (chroma=%d, bm25=%d); rebuilding BM25 from Chroma.",
                chroma_count,
                bm25_count,
            )
            self.rebuild_bm25_from_chroma()

    def rebuild_bm25_from_chroma(self) -> int:
        """Rebuild BM25 pickle and index from Chroma (single source of truth)."""
        try:
            data = self.vectordb.get(include=["documents", "metadatas"])
            documents = data.get("documents") or []
            metadatas = data.get("metadatas") or []
            corpus = [
                {"content": doc, "metadata": meta or {}}
                for doc, meta in zip(documents, metadatas)
            ]
            if corpus:
                self._persist_bm25_corpus(corpus)
            else:
                self.bm25_corpus = []
                self.bm25 = None
                if os.path.exists(self.bm25_corpus_path):
                    os.remove(self.bm25_corpus_path)
            logger.info("Rebuilt BM25 from Chroma (%d chunks).", len(corpus))
            return len(corpus)
        except Exception as e:
            logger.error(f"Failed to rebuild BM25 from Chroma: {e}")
            raise

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        metadatas = []
        ids = []

        for c in chunks:
            ids.append(c["id"])
            meta = {
                "doc_id": c.get("doc_id", ""),
                "page_number": c.get("page_number", 0),
                "chunk_index": c.get("chunk_index", 0),
                **{
                    k: v
                    for k, v in c.get("metadata", {}).items()
                    if isinstance(v, (str, int, float, bool))
                },
            }
            metadatas.append(meta)

        corpus_backup = list(self.bm25_corpus)
        new_entries = [
            {"content": text, "metadata": meta}
            for text, meta in zip(texts, metadatas)
        ]
        updated_corpus = corpus_backup + new_entries

        try:
            self.vectordb.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            logger.info(f"Added {len(chunks)} chunks to Chroma.")
            self._persist_bm25_corpus(updated_corpus)
            logger.info("Updated BM25 index in sync with Chroma.")
        except Exception as e:
            logger.error(f"Error adding chunks: {e}")
            try:
                self.vectordb.delete(ids=ids)
                logger.info("Rolled back Chroma ids after failed BM25 persist.")
            except Exception as rollback_err:
                logger.error(f"Chroma rollback failed: {rollback_err}")
            self.bm25_corpus = corpus_backup
            self._rebuild_bm25_index()
            raise

    def delete_by_source(self, source: str) -> int:
        """Remove all chunks for a filename from Chroma and BM25 together."""
        try:
            data = self.vectordb.get(include=["metadatas"])
            ids_to_delete = [
                chunk_id
                for chunk_id, meta in zip(
                    data.get("ids") or [], data.get("metadatas") or []
                )
                if meta and meta.get("source") == source
            ]

            if ids_to_delete:
                self.vectordb.delete(ids=ids_to_delete)

            self.bm25_corpus = [
                doc
                for doc in self.bm25_corpus
                if doc.get("metadata", {}).get("source") != source
            ]

            if self.bm25_corpus:
                self._persist_bm25_corpus(self.bm25_corpus)
            else:
                self.bm25 = None
                if os.path.exists(self.bm25_corpus_path):
                    os.remove(self.bm25_corpus_path)

            logger.info("Deleted %d chunks for source '%s'.", len(ids_to_delete), source)
            return len(ids_to_delete)
        except Exception as e:
            logger.error(f"Error deleting by source '{source}': {e}")
            self.rebuild_bm25_from_chroma()
            raise

    def hybrid_search(self, query: str, k: int = 10, filter=None) -> List[Dict[str, Any]]:
        vector_results = []
        try:
            results = self.vectordb.similarity_search_with_score(
                query, k=k, filter=filter
            )
            for doc, score in results:
                vector_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score,
                })
        except Exception as e:
            logger.error(f"Vector search failed: {e}")

        bm25_results = []
        if self.bm25 and self.bm25_corpus:
            try:
                tokenized_query = query.lower().split()
                scores = self.bm25.get_scores(tokenized_query)
                top_n = np.argsort(scores)[::-1][:k]
                for idx in top_n:
                    if scores[idx] > 0:
                        doc = self.bm25_corpus[idx]
                        if filter and not self._metadata_matches_filter(
                            doc.get("metadata", {}), filter
                        ):
                            continue
                        bm25_results.append({
                            "content": doc["content"],
                            "metadata": doc["metadata"],
                            "score": float(scores[idx]),
                        })
            except Exception as e:
                logger.error(f"BM25 search failed: {e}")

        rrf_scores = {}
        combined_docs = {}

        def get_hash(text):
            return hashlib.md5(text.encode()).hexdigest()

        for rank, item in enumerate(vector_results):
            h = get_hash(item["content"])
            combined_docs[h] = item
            rrf_scores[h] = rrf_scores.get(h, 0) + 1.0 / (60 + rank)

        for rank, item in enumerate(bm25_results):
            h = get_hash(item["content"])
            if h not in combined_docs:
                combined_docs[h] = item
            rrf_scores[h] = rrf_scores.get(h, 0) + 1.0 / (60 + rank)

        ranked_hashes = sorted(rrf_scores.keys(), key=lambda h: rrf_scores[h], reverse=True)
        return [combined_docs[h] for h in ranked_hashes[:k]]

    @staticmethod
    def _metadata_matches_filter(metadata: dict, filter: dict) -> bool:
        """Support simple equality filters used by Chroma."""
        for key, val in filter.items():
            if metadata.get(key) != val:
                return False
        return True

    def similarity_search(self, query: str, k: int = 5) -> List[Any]:
        return self.vectordb.similarity_search(query, k=k)

    def similarity_search_with_score(self, query, k=4, filter=None):
        return self.vectordb.similarity_search_with_score(
            query=query, k=k, filter=filter
        )

    def get_processed_documents(self) -> List[str]:
        try:
            data = self.vectordb.get(include=["metadatas"])
            metadatas = data.get("metadatas", [])
            sources = set()
            for meta in metadatas:
                if meta and "source" in meta:
                    sources.add(meta["source"])
            return sorted(list(sources))
        except Exception as e:
            logger.error(f"Error getting processed documents: {e}")
            return []

    def clear_database(self):
        try:
            self.vectordb.delete_collection()
            self.vectordb = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function,
                collection_name=self.collection_name,
            )

            self.bm25_corpus = []
            self.bm25 = None
            if os.path.exists(self.bm25_corpus_path):
                os.remove(self.bm25_corpus_path)

            logger.info("Vector database and BM25 index cleared successfully")

        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            raise
