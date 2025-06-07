from typing import Tuple

from .. import logger
from .Document import Document


class MarkdownDocument(Document):
    def get_raw_text(self) -> Tuple[str, dict]:
        try:
            with open(self.get_abs_path(), "r", encoding="utf-8", errors="ignore") as f:
                return f.read(), {"ocr_used": False}
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
            return "", {"ocr_used": False}
