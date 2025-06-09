from pathlib import Path
import time
from typing import List, Tuple

import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader

from .. import logger
from .Document import Document
from ..config import config


def ocr_pdf(path: Path, nb_pages: int) -> List[str]:
    text = ""
    logger.info("OCR")
    relpath = path.relative_to(config.DOCS_PATH)
    ocr_dir = config.STATE_DB_PATH.parent / relpath.parent / (path.parts[-1] + ".ocr")
    ocr_dir.mkdir(parents=True, exist_ok=True)

    time_img = 0
    time_ocr = 0
    # Convert each page to an image
    for k_page in range(1, 1 + nb_pages):
        ocr_txt = ocr_dir / f"page{k_page:05}.cache"
        if ocr_txt.exists():
            with open(ocr_txt, "r") as f:
                txt = f.read()

        else:
            t0 = time.time()
            img = convert_from_path(path, first_page=k_page, last_page=k_page, dpi=300)[0]
            time_img += time.time() - t0

            t0 = time.time()
            txt = pytesseract.image_to_string(img, lang=config.OCR_LANG)
            time_ocr += time.time() - t0
            with open(ocr_txt, "w") as f:
                f.write(txt)

        text = text + "\n\n" + txt

    if time_img < 0.1:
        logger.info(f"Time spent in image generation: {time_img} s")
        logger.info(f"Time spent in OCR: {time_ocr} s")

    return text


class PdfDocument(Document):
    def get_raw_text(self) -> Tuple[str, dict]:
        path = self.get_abs_path()
        text_chunks = []
        len_text = 0
        try:
            reader = PdfReader(path)
        except Exception:
            logger.warning("Error while reading the file. Skipping")
            return None, {"ocr_used": False}

        nb_pages = len(reader.pages)
        logger.info(f"Reading {nb_pages} pages pdf file")
        for page in reader.pages:
            txt = page.extract_text() or ""
            text_chunks.append(txt)
            len_text += len(txt)

        full_text = "\n".join(text_chunks).strip()

        # If nearly empty, fallback to OCR
        if len_text < 10 * nb_pages:
            logger.info("PDF text is too short; falling back to OCR")
            return ocr_pdf(path, nb_pages), {"ocr_used": True}

        return full_text, {"ocr_used": False}
