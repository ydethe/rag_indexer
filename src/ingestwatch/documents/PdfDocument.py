from pathlib import Path
from typing import Tuple

import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader

from .. import logger
from .Document import Document
from ..config import config


def ocr_pdf(path: Path, nb_pages: int) -> str:
    text = []
    logger.info("OCR")
    try:
        # Convert each page to an image
        for k_page in range(1, 1 + nb_pages):
            images = convert_from_path(path, first_page=k_page, last_page=k_page, dpi=300)
            for img in images:
                txt = pytesseract.image_to_string(img, lang=config.OCR_LANG)
                text.append(txt)
    except Exception as e:
        logger.error(f"OCR failed: {e}")
    return "\n".join(text).strip()


class PdfDocument(Document):
    def get_raw_text(self) -> Tuple[str, dict]:
        path = self.get_abs_path()
        text_chunks = []
        len_text = 0
        reader = PdfReader(path)
        nb_pages = len(reader.pages)
        try:
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

        except Exception as e:
            logger.warning(f"Error extracting PDF text: {e}. Using OCR.")
            return ocr_pdf(path, nb_pages), {"ocr_used": True}
