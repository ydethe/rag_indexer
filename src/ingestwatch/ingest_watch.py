#!/usr/bin/env python3
import hashlib
import os
import sys
import time
import sqlite3
import threading
from pathlib import Path
from typing import Optional, List
import uuid

import rich.progress as rp
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
import nltk
from nltk.tokenize import sent_tokenize
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import docx
import openpyxl

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PointIdsList,
)

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


# === Qdrant helper ===
class QdrantIndexer:
    def __init__(self):
        self.client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            api_key=config.QDRANT_API_KEY,
            https=False,
        )
        # Check if collection exists; if not, create
        existing = [c.name for c in self.client.get_collections().collections]
        if config.COLLECTION_NAME not in existing:
            # We’ll determine vector size after loading the model
            pass

    def create_collection_if_missing(self, vector_size: int):
        existing = [c.name for c in self.client.get_collections().collections]
        if config.COLLECTION_NAME not in existing:
            self.client.recreate_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                on_disk_payload=True,
            )
            logger.info(
                f"Created Qdrant collection '{config.COLLECTION_NAME}' (size={vector_size})."
            )

    def upsert(self, points: List[PointStruct]):
        if points:
            self.client.upsert(collection_name=config.COLLECTION_NAME, points=points)

    def delete(self, ids: List[str]):
        if ids:
            pil = PointIdsList(points=ids)
            self.client.delete(collection_name=config.COLLECTION_NAME, points_selector=pil)


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


# === File Processor ===
class DocumentIndexer:
    def __init__(self):
        # Load embedding model
        self.model = SentenceTransformer(
            config.EMBEDDING_MODEL, trust_remote_code=config.EMBEDDING_MODEL_TRUST_REMOTE_CODE
        )
        self.vector_size = self.model.get_sentence_embedding_dimension()

        # Initialize Qdrant
        self.qdrant = QdrantIndexer()
        self.qdrant.create_collection_if_missing(self.vector_size)

        # Root folder to watch
        self.root = config.DOCS_PATH

        # Lock around state & indexing operations
        self.lock = threading.Lock()

    def process_file(self, filepath: Path):
        """
        Extract text, chunk, embed, and upsert into Qdrant.
        The file ID in Qdrant will be a SHA1 of its absolute path.
        """
        try:
            stat = os.path.getmtime(filepath)
            stored = get_stored_timestamp(filepath)
            if stored is not None and stored == stat:
                # No change
                return

            logger.info(f"[INDEX] Processing changed file: {filepath}")
            text = extract_text(str(filepath))
            if not text:
                logger.warning(f"No text extracted; skipping: {filepath}")
                return

            chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            logger.info(f"Embedding {len(chunks)} chunks")
            embeddings = self.model.encode(chunks, show_progress_bar=True)

            points: list[PointStruct] = []
            # Use MD5 of path + chunk index as unique point ID
            for idx, (chunk, emb) in rp.track(
                enumerate(zip(chunks, embeddings)),
                description=f"Building {len(chunks)} embeddings for qdrant",
            ):
                pid = str(
                    uuid.UUID(
                        int=int(hashlib.md5(f"{filepath}::{idx}".encode("utf-8")).hexdigest(), 16)
                    )
                )
                payload = {
                    "source": str(filepath),
                    "chunk_index": idx,
                    "text": chunk,
                }
                points.append(PointStruct(id=pid, vector=emb.tolist(), payload=payload))

            # Upsert into Qdrant
            self.qdrant.upsert(points)

            # Update state DB
            set_stored_timestamp(filepath, stat)
            logger.info(f"[INDEX] Upserted {len(points)} vectors from {filepath}")

        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")

    def rename_file(self, src_abs_path, dest_abs_path):
        rename_stored_file(src_abs_path, dest_abs_path)

    def remove_file(self, abspath: str):
        """
        Delete all vectors whose payload.source == this file's absolute path.
        We identify by regenerating all chunk IDs for old state—but since we store
        last-modified in SQLite, we know it existed before; we'll iterate over state DB
        to remove associated chunk IDs. Simpler: query by payload.source in Qdrant.
        """
        logger.info(f"[DELETE] Removing file from index: {abspath}")

        # Query Qdrant for all points with payload.source == abspath
        # filter_ = {"must": [{"key": "source", "match": {"value": abspath}}]}
        filter_ = Filter(must=[FieldCondition(key="source", match=MatchValue(value=str(abspath)))])

        # Retrieve IDs matching that filter
        hits = self.qdrant.client.search(
            collection_name=config.COLLECTION_NAME,
            query_vector=[0.0] * self.vector_size,  # dummy vector; we only want IDs
            limit=1000,
            query_filter=filter_,
        )

        ids_to_delete = [hit.id for hit in hits]
        if ids_to_delete:
            self.qdrant.delete(ids_to_delete)
            logger.info(f"[DELETE] Removed {len(ids_to_delete)} vectors for {abspath}")

        # Remove from state DB
        delete_stored_file(abspath)

    def initial_scan(self):
        """
        On startup, walk entire DOCS_PATH and index any new/modified files.
        Also, find any entries in state DB that no longer exist on disk, and remove them.
        """
        logger.info("Performing initial scan of documents folder...")

        # 1. Build a set of all file paths on disk
        disk_files: list[Path] = []
        for ext in ("*.pdf", "*.docx", "*.xlsx", "*.xlsm", "*.md", "*.txt"):
            disk_files.extend(self.root.rglob(ext))
        disk_files = [p.resolve() for p in disk_files]

        # 2. For each file on disk, check timestamp vs. state DB
        for file_path in disk_files:
            stored = get_stored_timestamp(file_path)
            modified = os.path.getmtime(str(file_path))
            if stored is None or stored != modified:
                # New or changed
                self.process_file(file_path)

        # 3. For each file in state DB, if not on disk anymore, delete from Qdrant
        conn = sqlite3.connect(config.STATE_DB_PATH)
        c = conn.cursor()
        c.execute("SELECT path FROM files")
        rows = c.fetchall()
        conn.close()

        for (stored_path,) in rows:
            if not Path(stored_path).exists():
                # Remove from Qdrant
                self.remove_file(stored_path)

    def on_created_or_modified(self, event: FileSystemEvent):
        if event.is_directory:
            return

        filepath = Path(event.src_path)
        if filepath.suffix in (".pdf", ".docx", ".xlsx", ".xlsm", ".md", ".txt"):
            with self.lock:
                # Small delay to allow file write to finish
                time.sleep(0.5)
                self.process_file(filepath)

    def on_deleted(self, event: FileSystemEvent):
        if event.is_directory:
            return

        filepath = Path(event.src_path)
        if filepath.suffix in (".pdf", ".docx", ".xlsx", ".xlsm", ".md", ".txt"):
            with self.lock:
                self.remove_file(filepath)

    def on_moved(self, event: FileSystemEvent):
        # TODO Implement folder and file renaming
        if event.is_directory:
            return

        srcpath = Path(event.src_path)
        destpath = Path(event.dest_path)
        if srcpath.suffix in (".pdf", ".docx", ".xlsx", ".xlsm", ".md", ".txt"):
            with self.lock:
                time.sleep(0.5)
                self.remove_file(srcpath)
                self.process_file(destpath)

    def start_watcher(self):
        event_handler = FileSystemEventHandler()
        event_handler.on_created = self.on_created_or_modified
        event_handler.on_modified = self.on_created_or_modified
        event_handler.on_moved = self.on_moved
        event_handler.on_deleted = self.on_deleted

        observer = Observer()
        observer.schedule(event_handler, str(self.root), recursive=True)
        observer.start()

        logger.info(f"Started file watcher on: {config.DOCS_PATH}")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()


def main():
    # === Ensure NLTK punkt is available ===
    nltk.download("punkt_tab")
    nltk.download("punkt")

    # Ensure state DB exists
    initialize_state_db()

    # Ensure documents folder exists
    if not config.DOCS_PATH.exists():
        logger.error(f"Documents folder not found: {config.DOCS_PATH}")
        sys.exit(1)

    indexer = DocumentIndexer()

    # Initial full scan
    indexer.initial_scan()

    # Start the filesystem watcher loop
    indexer.start_watcher()


if __name__ == "__main__":
    main()
