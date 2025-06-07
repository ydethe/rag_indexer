from typing import Tuple

import docx

from .. import logger
from .Document import Document


class DocDocument(Document):
    def get_raw_text(self) -> Tuple[str, dict]:
        try:
            doc = docx.Document(str(self.get_abs_path()))
            page_count = sum(p.contains_page_break for p in doc.paragraphs) + 1
            logger.info(f"Reading {page_count} pages doc file")
            paragraphs = [p.text for p in doc.paragraphs]
            return "\n".join(paragraphs).strip(), {"ocr_used": False}
        except Exception as e:
            logger.error(f"Error reading DOCX: {e}")
            return "", {"ocr_used": False}
