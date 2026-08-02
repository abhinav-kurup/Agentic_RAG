import os
import time
import base64
import logging
from typing import Dict, Any, Optional

from core.config import Config

logger = logging.getLogger(__name__)


class ImageDescriber:
    """
    Multimodal Vision Module using LLM (Gemini 2.0 Vision) to describe visual content
    (charts, diagrams, figures, embedded tables) extracted from document pages.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or getattr(Config, "VISION_MODEL", "gemini/gemini-2.0-flash-lite")
        self.api_key = getattr(Config, "GOOGLE_API_KEY", None)

    def describe_image(
        self,
        image_bytes: bytes,
        page_number: int,
        img_index: int = 0,
        source_name: str = "document",
    ) -> Dict[str, Any]:
        """
        Saves extracted image bytes to disk and invokes Gemini Vision to describe image content.
        """
        clean_source = os.path.splitext(os.path.basename(source_name))[0]
        output_dir = os.path.join("data", "extracted_images", clean_source)
        os.makedirs(output_dir, exist_ok=True)

        image_filename = f"page_{page_number}_img_{img_index + 1}.png"
        image_path = os.path.join(output_dir, image_filename)

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        description = ""
        b64_img = base64.b64encode(image_bytes).decode("utf-8")

        prompt_text = (
            f"Analyze this image extracted from page {page_number} of document '{source_name}'. "
            "Provide a concise, focused summary (maximum 3-5 bullet points or under 150 words). "
            "Focus strictly on key visual elements, main text/numbers, chart trends, or diagram structures."
        )


        # Gemini Vision LLM invocation with retry handling for 429 rate limits
        if self.api_key:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage

            gemini_model = self.model_name.replace("gemini/", "")
            llm = ChatGoogleGenerativeAI(
                model=gemini_model,
                google_api_key=self.api_key,
                temperature=0.2,
                max_retries=1,
            )


            prompt_content = [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                },
            ]

            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = llm.invoke([HumanMessage(content=prompt_content)])
                    res_text = response.content

                    if isinstance(res_text, str):
                        description = res_text.strip()
                    elif isinstance(res_text, list):
                        parts = []
                        for part in res_text:
                            if isinstance(part, dict) and "text" in part:
                                parts.append(part["text"])
                            else:
                                parts.append(str(part))
                        description = "\n".join(parts).strip()
                    else:
                        description = str(res_text).strip()

                    logger.info(f"Generated Gemini Vision description for {image_filename} ({len(description)} chars).")

                    break
                except Exception as err:
                    err_str = str(err)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Too Many Requests" in err_str:
                        if attempt < max_retries - 1:
                            logger.warning(f"Gemini Vision rate-limited (429) on page {page_number}. Waiting 2s before retry...")
                            time.sleep(2.0)
                            continue
                    logger.warning(f"Gemini Vision description unavailable for page {page_number}: {err}")
                    break

        if not description:
            description = (
                f"[Extracted Image on Page {page_number} of {source_name}: "
                f"Visual figure / diagram (saved to {image_path})]"
            )

        return {
            "page_number": page_number,
            "img_index": img_index,
            "image_path": image_path,
            "description": description,
        }
