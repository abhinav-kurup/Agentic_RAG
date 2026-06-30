from langsmith import traceable
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)

class PDFLoader:
    def __init__(self):
        pass

    @traceable(name="Loader")
    def load(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Loads a PDF file and extracts content from each page.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            List of dictionaries containing page content and metadata.
            [
                {
                    "page_number": int,
                    "text": str,
                    "is_scanned": bool,
                    "images": List[Any],
                    "metadata": Dict
                },
                ...
            ]
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            logger.error(f"Failed to open PDF {file_path}: {e}")
            raise e

        processed_pages = []

        for i, page in enumerate(doc):
            text = page.get_text()
            
            page_data = {
                "page_number": i + 1,
                "text": text,
                "metadata": doc.metadata
            }
            processed_pages.append(page_data)

        doc.close()
        return processed_pages
