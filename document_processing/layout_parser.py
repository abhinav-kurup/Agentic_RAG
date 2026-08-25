import os
import json
import time
import datetime
import fitz  # PyMuPDF (used solely for image extraction)
import logging
import re
from typing import List, Dict, Any, Optional

from core.config import Config
from document_processing.image_describer import ImageDescriber

logger = logging.getLogger(__name__)

# Markdown pipe tables: header + separator + one or more body rows
_TABLE_RE = re.compile(
    r"(\|[^\n]+\|\n\|[-:| ]+\|\n(?:\|[^\n]+\|\n?)+)",
)


def _blocks_from_page_markdown(page_md: str) -> List[Dict[str, Any]]:
    """Split page markdown into text/table blocks without duplicating table content in text."""
    blocks: List[Dict[str, Any]] = []
    last_end = 0
    for match in _TABLE_RE.finditer(page_md):
        before = page_md[last_end:match.start()].strip()
        if before:
            blocks.append({"type": "text", "content": before})
        blocks.append({"type": "table", "content": match.group(1).strip()})
        last_end = match.end()
    after = page_md[last_end:].strip()
    if after:
        blocks.append({"type": "text", "content": after})
    if not blocks and page_md.strip():
        blocks.append({"type": "text", "content": page_md.strip()})
    return blocks


class PDFLayoutParser:
    """
    LlamaParse Document Layout Parser.
    Converts PDF pages directly to structured Markdown using LlamaParse,
    enriches visual figures/charts using Gemini LLM Vision descriptions,
    saves images locally for retrieval agents, and generates parsing statistics in JSON.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        image_describer: Optional[ImageDescriber] = None,
    ):
        self.api_key = api_key or getattr(Config, "LLAMA_CLOUD_API_KEY", None) or os.getenv("LLAMA_CLOUD_API_KEY")
        self.image_describer = image_describer or ImageDescriber()

    def parse_document(self, file_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        source_name = os.path.basename(file_path)

        logger.info(f"Parsing document '{source_name}' using LlamaParse...")
        from llama_parse import LlamaParse

        parser = LlamaParse(
            api_key=self.api_key,
            result_type="markdown",
            verbose=False,
        )
        documents = parser.load_data(file_path)
        logger.info(f"LlamaParse successfully parsed {len(documents)} pages for '{source_name}'")

        return self._process_llama_documents(documents, file_path, source_name)

    def _process_llama_documents(
        self, documents: List[Any], file_path: str, source_name: str
    ) -> List[Dict[str, Any]]:
        """Processes LlamaParse markdown pages, enriches them with extracted image descriptions, and saves stats JSON."""
        clean_name = os.path.splitext(source_name)[0]

        # Directory setup
        markdown_dir = os.path.join("data", "parsed_markdown")
        stats_dir = os.path.join("data", "parsing_stats")
        os.makedirs(markdown_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)

        saved_md_path = os.path.join(markdown_dir, f"{clean_name}.md")
        saved_stats_path = os.path.join(stats_dir, f"{clean_name}_stats.json")

        processed_pages = []
        full_doc_md_parts = []
        
        # Statistics Tracking
        total_tables = 0
        total_images = 0
        image_details = []
        table_details = []

        doc = fitz.open(file_path) if Config.ENABLE_IMAGE_PROCESSING else None

        for page_idx, llama_doc in enumerate(documents):
            page_num = page_idx + 1
            blocks = []
            page_md = getattr(llama_doc, "text", str(llama_doc))
            full_doc_md_parts.append(f"<!-- Page {page_num} -->\n{page_md}\n")

            print(f"\n==================== LLAMAPARSE RESULT (Page {page_num}/{len(documents)}) ====================")
            print(page_md[:800] + ("\n... [truncated]" if len(page_md) > 800 else ""))
            print("=================================================================================\n")

            # Emit separate table blocks so TypeBasedChunker keeps tables intact
            page_blocks = _blocks_from_page_markdown(page_md)
            table_count = sum(1 for b in page_blocks if b["type"] == "table")
            if table_count:
                total_tables += table_count
                for tbl_idx, block in enumerate(
                    (b for b in page_blocks if b["type"] == "table"), start=1
                ):
                    table_details.append({
                        "page_number": page_num,
                        "table_index": tbl_idx,
                        "preview": block["content"][:200] + "...",
                    })
            blocks.extend(page_blocks)

            # Extract embedded images (optional — gated by ENABLE_IMAGE_PROCESSING)
            if Config.ENABLE_IMAGE_PROCESSING and doc is not None and page_idx < len(doc):
                fitz_page = doc[page_idx]
                image_list = fitz_page.get_images(full=True)
                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]
                    try:
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image.get("image")
                        if not image_bytes or len(image_bytes) < 1000:
                            continue

                        img_result = self.image_describer.describe_image(
                            image_bytes=image_bytes,
                            page_number=page_num,
                            img_index=img_idx,
                            source_name=source_name,
                        )

                        total_images += 1
                        image_details.append({
                            "page_number": page_num,
                            "image_index": img_idx + 1,
                            "image_path": img_result["image_path"],
                            "description": img_result["description"],
                        })

                        blocks.append({
                            "type": "image",
                            "content": img_result["description"],
                            "image_path": img_result["image_path"],
                            "image_index": img_idx,
                        })
                    except Exception as img_err:
                        logger.warning(f"Image extraction error on LlamaParse page {page_num}: {img_err}")

            processed_pages.append({
                "page_number": page_num,
                "text": page_md,
                "blocks": blocks,
                "metadata": (doc.metadata if doc else {}) or {},
            })

        if doc is not None:
            doc.close()

        # Write complete document markdown to disk
        try:
            with open(saved_md_path, "w", encoding="utf-8") as md_file:
                md_file.write("\n\n".join(full_doc_md_parts))
            logger.info(f"Saved complete LlamaParse Markdown to disk at: {saved_md_path}")
            print(f"--> Saved complete LlamaParse Markdown file: {saved_md_path}")
        except Exception as save_err:
            logger.warning(f"Could not save LlamaParse markdown file to disk: {save_err}")

        # Write Parsing Statistics JSON to disk
        text_block_count = sum(
            1 for p in processed_pages for b in p.get("blocks", []) if b.get("type") == "text"
        )
        parsing_stats = {
            "document_name": source_name,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_pages": len(documents),
            "total_text_blocks": text_block_count,
            "total_tables": total_tables,
            "total_images": total_images,
            "markdown_file": saved_md_path,
            "table_details": table_details,
            "image_details": image_details,
        }

        try:
            with open(saved_stats_path, "w", encoding="utf-8") as json_file:
                json.dump(parsing_stats, json_file, indent=2)
            logger.info(f"Saved parsing statistics JSON to disk at: {saved_stats_path}")
            print(f"--> Saved parsing statistics JSON file: {saved_stats_path}")
            print(f"    [Stats Summary: {len(documents)} pages, {total_tables} tables, {total_images} images saved locally]")
        except Exception as stats_err:
            logger.warning(f"Could not save parsing statistics JSON: {stats_err}")

        return processed_pages
