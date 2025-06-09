from typing import Iterable, Tuple

from .Document import Document


class MarkdownDocument(Document):
    def iterate_raw_text(self) -> Iterable[Tuple[str, dict]]:
        with open(self.get_abs_path(), "r", encoding="utf-8", errors="ignore") as f:
            yield f.read(), {"ocr_used": False}
