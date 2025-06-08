from typing import Tuple

from .Document import Document


class MarkdownDocument(Document):
    def get_raw_text(self) -> Tuple[str, dict]:
        with open(self.get_abs_path(), "r", encoding="utf-8", errors="ignore") as f:
            return f.read(), {"ocr_used": False}
