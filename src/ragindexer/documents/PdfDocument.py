from pathlib import Path
from typing import Iterable, Tuple

import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader

from .. import logger
from .ADocument import ADocument
from ..config import config


def ocr_pdf(path: Path, k_page: int, ocr_dir: Path) -> str:
    ocr_dir.mkdir(parents=True, exist_ok=True)

    # Convert the page to an image
    ocr_txt = ocr_dir / f"page{k_page:05}.cache"
    if ocr_txt.exists():
        with open(ocr_txt, "r") as f:
            txt = f.read()

    else:
        img = convert_from_path(path, first_page=k_page, last_page=k_page, dpi=300)[0]

        try:
            txt = pytesseract.image_to_string(img, lang=config.OCR_LANG)
            with open(ocr_txt, "w") as f:
                f.write(txt)
        except Exception as e:
            logger.error(f"OCR failed : {e}")
            txt = None

    return txt


class PdfDocument(ADocument):
    def __init__(self, abspath):
        super().__init__(abspath)

        if abspath.parts[0] == "/":
            self.ocr_dir = (
                config.STATE_DB_PATH.parent
                / "cache"
                / abspath.parent.relative_to("/")
                / (abspath.parts[-1] + ".ocr")
            )
        else:
            self.ocr_dir = (
                config.STATE_DB_PATH.parent
                / "cache"
                / abspath.parent
                / (abspath.parts[-1] + ".ocr")
            )

        if self.ocr_dir.exists():
            logger.info(f"Reusing OCR cache for {self.ocr_dir}")
            self.using_ocr = True
        else:
            self.using_ocr = False

    def iterate_raw_text(self) -> Iterable[Tuple[str, dict]]:
        path = self.get_abs_path()
        try:
            reader = PdfReader(path)
            nb_pages = len(reader.pages)
        except Exception:
            logger.error("Error while reading the file. Skipping")
            return None, {"ocr_used": False}

        logger.info(f"Reading {nb_pages} pages pdf file")
        file_metadata = {"ocr_used": False}
        avct = -1
        for k_page, page in enumerate(reader.pages):
            new_avct = int(k_page / nb_pages * 100 / 10)
            if new_avct != avct:
                logger.info(f"Lecture page {k_page+1}/{nb_pages}")
                avct = new_avct

            try:
                txt = page.extract_text() or ""
            except Exception as e:
                logger.error(f"While extracting text: {e}")
                txt = ""

            if len(txt) < 10:
                if not self.using_ocr:
                    self.using_ocr = True
                    logger.info(f"Using OCR for '{self.get_abs_path()}' in '{self.ocr_dir}")

                file_metadata["ocr_used"] = True
                txt = ocr_pdf(path, k_page + 1, self.ocr_dir)

            if txt is None or txt == "":
                continue

            yield k_page, txt, file_metadata
