from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import hashlib


def make_chunk_id(source: str, page_number, chunk_index: int, text: str) -> str:
    content_hash = hashlib.md5(text.encode()).hexdigest()
    key = f"{source}|{page_number}|{chunk_index}|{content_hash}"
    return hashlib.sha256(key.encode()).hexdigest()


class DocumentChunker:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        """
        Args:
            chunk_size: Approx characters per chunk (aiming for ~500 tokens)
            chunk_overlap: Approx characters overlap
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_documents(
        self,
        pages: List[Dict[str, Any]],
        doc_id: str,
        source: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Splits pages into chunks with metadata.
        Chunk ids are stable per source/page/index/content for upsert-friendly re-ingest.
        """
        chunks = []
        chunk_counter = 0

        for page in pages:
            text = page.get("text", "")
            if not text:
                continue

            page_chunks = self.splitter.split_text(text)
            page_number = page.get("page_number")

            for chunk_text in page_chunks:
                chunks.append({
                    "id": make_chunk_id(source, page_number, chunk_counter, chunk_text),
                    "doc_id": doc_id,
                    "page_number": page_number,
                    "chunk_index": chunk_counter,
                    "text": chunk_text,
                    "metadata": {
                        **page.get("metadata", {}),
                    },
                })
                chunk_counter += 1

        return chunks
