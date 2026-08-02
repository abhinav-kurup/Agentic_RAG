from langsmith import traceable
from typing import List, Dict, Any, Optional
import logging
import os

from document_processing.layout_parser import PDFLayoutParser

logger = logging.getLogger(__name__)

class PDFLoader:
    def __init__(self, layout_parser: Optional[PDFLayoutParser] = None):
        self.parser = layout_parser or PDFLayoutParser()

    @traceable(name="Loader")
    def load(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Loads a PDF file with layout-aware block parsing (text, tables, images with LLM vision descriptions).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        return self.parser.parse_document(file_path)
