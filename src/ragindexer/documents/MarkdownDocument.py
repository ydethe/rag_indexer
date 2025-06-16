from typing import Iterable, Tuple

from .ADocument import ADocument


class MarkdownDocument(ADocument):
    def iterate_raw_text(self) -> Iterable[Tuple[str, dict]]:
        with open(self.get_abs_path(), "r", encoding="utf-8", errors="ignore") as f:
            yield 0, f.read(), {"ocr_used": False}
