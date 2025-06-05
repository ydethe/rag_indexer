# === File Processor ===
#!/usr/bin/env python3
import hashlib
import os
import time
import sqlite3
import threading
from pathlib import Path
import uuid

import rich.progress as rp
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from sentence_transformers import SentenceTransformer
from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from . import logger
from .ingest_watch import (
    chunk_text,
    delete_stored_file,
    extract_text,
    get_stored_timestamp,
    rename_stored_file,
    set_stored_timestamp,
)
from .config import config
from .QdrantIndexer import QdrantIndexer


class DocumentIndexer:
    def __init__(self):
        # Load embedding model
        self.model = SentenceTransformer(
            config.EMBEDDING_MODEL, trust_remote_code=config.EMBEDDING_MODEL_TRUST_REMOTE_CODE
        )
        self.vector_size = self.model.get_sentence_embedding_dimension()

        # Initialize Qdrant
        self.qdrant = QdrantIndexer(vector_size=self.vector_size)
        self.qdrant.create_collection_if_missing()

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
        hits = self.qdrant.search(limit=1000, query_filter=filter_)

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

        self.__observer = Observer()
        self.__observer.schedule(event_handler, str(self.root), recursive=True)
        self.__observer.start()

        logger.info(f"Started file watcher on: {config.DOCS_PATH}")
        # try:
        #     while True:
        #         time.sleep(1)
        # except KeyboardInterrupt:
        #     self.__observer.stop()
        # self.__observer.join()

    def join(self):
        self.__observer.join()
