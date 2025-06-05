#!/usr/bin/env python3
import os
import sqlite3
from pathlib import Path
from typing import Optional, List

import rich.progress as rp
from nltk.tokenize import sent_tokenize
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import docx
import openpyxl

from . import logger
from .config import config


# === SQLite state DB helpers ===
def initialize_state_db():
    os.makedirs(os.path.dirname(config.STATE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.STATE_DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            last_modified REAL
        )
    """
    )
    conn.commit()
    conn.close()


def get_stored_timestamp(path: Path) -> Optional[float]:
    conn = sqlite3.connect(config.STATE_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT last_modified FROM files WHERE path = ?", (str(path),))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def set_stored_timestamp(path: Path, ts: float):
    conn = sqlite3.connect(config.STATE_DB_PATH)
    c = conn.cursor()
    c.execute("REPLACE INTO files (path, last_modified) VALUES (?, ?)", (str(path), ts))
    conn.commit()
    conn.close()


def delete_stored_file(path: Path):
    conn = sqlite3.connect(config.STATE_DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM files WHERE path = ?", (str(path),))
    conn.commit()
    conn.close()


def rename_folder(srcpath: Path, destpath: Path):
    conn = sqlite3.connect(config.STATE_DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE files SET path=? WHERE path LIKE ?", (str(destpath), f"%{srcpath}"))
    conn.commit()
    conn.close()


def rename_stored_file(srcpath: Path, destpath: Path):
    conn = sqlite3.connect(config.STATE_DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE files SET path=? WHERE path=?", (str(destpath), str(srcpath)))
    conn.commit()
    conn.close()


# === Text extraction per filetype ===
def extract_text_from_pdf(path: str) -> str:
    text_chunks = []
    try:
        reader = PdfReader(path)
        for page in rp.track(
            reader.pages, description=f"Reading {len(reader.pages)} pages pdf file"
        ):
            txt = page.extract_text() or ""
            text_chunks.append(txt)
        full_text = "\n".join(text_chunks).strip()
        # If nearly empty, fallback to OCR
        if len(full_text) < 100:
            logger.info(f"PDF text is too short; falling back to OCR: {path}")
            return ocr_pdf(path)
        return full_text
    except Exception as e:
        logger.warning(f"Error extracting PDF text ({path}): {e}. Using OCR.")
        return ocr_pdf(path)


def ocr_pdf(path: str) -> str:
    text = []
    try:
        # Convert each page to an image
        images = convert_from_path(path)
        for img in rp.track(images, description="Running OCR on pdf pages"):
            txt = pytesseract.image_to_string(img, lang=config.OCR_LANG)
            text.append(txt)
    except Exception as e:
        logger.error(f"OCR failed for {path}: {e}")
    return "\n".join(text).strip()


def extract_text_from_docx(path: str) -> str:
    try:
        doc = docx.Document(path)
        page_count = sum(p.contains_page_break for p in doc.paragraphs) + 1
        paragraphs = [
            p.text
            for p in rp.track(doc.paragraphs, description=f"Reading {page_count} pages doc file")
        ]
        return "\n".join(paragraphs).strip()
    except Exception as e:
        logger.error(f"Error reading DOCX ({path}): {e}")
        return ""


def extract_text_from_xlsx(path: str) -> str:
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        all_text = []
        for sheet in rp.track(
            wb.worksheets, description=f"Reading {len(wb.worksheets)} pages excel file"
        ):
            for row in sheet.iter_rows(values_only=True):
                row_text = [str(cell) for cell in row if cell is not None]
                if row_text:
                    all_text.append(" ".join(row_text))
        return "\n".join(all_text).strip()
    except Exception as e:
        logger.error(f"Error reading XLSX ({path}): {e}")
        return ""


def extract_text_from_md_or_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading text file ({path}): {e}")
        return ""


def extract_text(path: str) -> str:
    ext = path.lower().split(".")[-1]
    if ext == "pdf":
        return extract_text_from_pdf(path)
    elif ext in ("docx",):
        return extract_text_from_docx(path)
    elif ext in ("xlsx", "xlsm", "xls"):
        return extract_text_from_xlsx(path)
    elif ext in ("md", "txt"):
        return extract_text_from_md_or_txt(path)
    else:
        logger.warning(f"Unsupported extension (skipping text extraction): {path}")
        return ""


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Splits text into overlapping chunks of ~chunk_size characters, aligned on sentences.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sent in rp.track(sentences, description="Building chunks"):
        if len(current_chunk) + len(sent) + 1 <= chunk_size:
            current_chunk += " " + sent if current_chunk else sent
        else:
            chunks.append(current_chunk)
            # Start new chunk: include overlap
            overlap_text = (
                current_chunk[-chunk_overlap:]
                if chunk_overlap < len(current_chunk)
                else current_chunk
            )
            current_chunk = overlap_text + " " + sent

    if current_chunk:
        chunks.append(current_chunk)
    return chunks
