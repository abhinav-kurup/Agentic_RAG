from langsmith import traceable
from typing import List, Dict, Any, Optional
from document_processing.type_chunker import TypeBasedChunker, make_chunk_id


class DocumentChunker:
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 180):
        self.chunker = TypeBasedChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    @traceable(name="Chunker")
    def split_documents(
        self,
        pages: List[Dict[str, Any]],
        doc_id: str,
        source: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Splits layout-parsed pages into type-based chunks (text, table, image) with rich metadata.
        """
        return self.chunker.split_layout_pages(pages=pages, doc_id=doc_id, source=source)
