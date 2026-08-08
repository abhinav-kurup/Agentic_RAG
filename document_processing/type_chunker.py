import hashlib
import logging
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

logger = logging.getLogger(__name__)


def make_chunk_id(source: str, page_number: int, chunk_index: int, text: str) -> str:
    content_hash = hashlib.md5(text.encode()).hexdigest()
    key = f"{source}|{page_number}|{chunk_index}|{content_hash}"
    return hashlib.sha256(key.encode()).hexdigest()


class TypeBasedChunker:
    """
    Layout-Aware and Type-Based Document Chunker.
    Categorizes and chunks content cleanly based on structural block type:
    - 'table': Standalone chunks keeping Markdown tabular structures intact.
    - 'image': Standalone chunks storing LLM Vision descriptions with image file references.
    - 'text': Markdown text split using recursive text splitter with heading awareness.
    """

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 180):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        )

    @traceable(name="TypeBasedChunker")
    def split_layout_pages(
        self,
        pages: List[Dict[str, Any]],
        doc_id: str,
        source: str = "",
    ) -> List[Dict[str, Any]]:
        chunks = []
        chunk_counter = 0

        for page in pages:
            page_num = page.get("page_number", 1)
            blocks = page.get("blocks", [])

            if not blocks:
                # Fallback if page has no parsed blocks
                raw_text = page.get("text", "")
                if raw_text:
                    sub_texts = self.text_splitter.split_text(raw_text)
                    for sub in sub_texts:
                        chunks.append({
                            "id": make_chunk_id(source, page_num, chunk_counter, sub),
                            "doc_id": doc_id,
                            "page_number": page_num,
                            "chunk_index": chunk_counter,
                            "text": sub,
                            "type": "text",
                            "metadata": {
                                "source": source,
                                "type": "text",
                                **page.get("metadata", {}),
                            },
                        })
                        chunk_counter += 1
                continue

            for block in blocks:
                b_type = block.get("type", "text")
                content = block.get("content", "").strip()
                if not content:
                    continue

                if b_type == "table":
                    # Standalone table chunk to preserve markdown tabular integrity
                    table_text = f"### [Table on Page {page_num}]\n{content}"
                    chunks.append({
                        "id": make_chunk_id(source, page_num, chunk_counter, table_text),
                        "doc_id": doc_id,
                        "page_number": page_num,
                        "chunk_index": chunk_counter,
                        "text": table_text,
                        "type": "table",
                        "metadata": {
                            "source": source,
                            "type": "table",
                            **page.get("metadata", {}),
                        },
                    })
                    chunk_counter += 1

                elif b_type == "image":
                    # Standalone image description chunk
                    img_path = block.get("image_path", "")
                    img_text = f"### [Image on Page {page_num}]\nDescription: {content}"
                    chunks.append({
                        "id": make_chunk_id(source, page_num, chunk_counter, img_text),
                        "doc_id": doc_id,
                        "page_number": page_num,
                        "chunk_index": chunk_counter,
                        "text": img_text,
                        "type": "image",
                        "metadata": {
                            "source": source,
                            "type": "image",
                            "image_path": img_path,
                            **page.get("metadata", {}),
                        },
                    })
                    chunk_counter += 1

                else:
                    # Text paragraph / heading blocks
                    text_splits = self.text_splitter.split_text(content)
                    for sub_text in text_splits:
                        chunks.append({
                            "id": make_chunk_id(source, page_num, chunk_counter, sub_text),
                            "doc_id": doc_id,
                            "page_number": page_num,
                            "chunk_index": chunk_counter,
                            "text": sub_text,
                            "type": "text",
                            "metadata": {
                                "source": source,
                                "type": "text",
                                **page.get("metadata", {}),
                            },
                        })
                        chunk_counter += 1

        logger.info(f"Split {len(pages)} pages into {len(chunks)} type-aware chunks for source '{source}'.")
        return chunks
