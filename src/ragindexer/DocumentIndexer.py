import os
import time
import threading
from pathlib import Path
from typing import Iterable, List, Tuple

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from sentence_transformers import SentenceTransformer
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
)

from .documents.DocumentFactory import DocumentFactory
from . import logger
from .index_database import (
    delete_stored_file,
    get_stored_timestamp,
    set_stored_timestamp,
    list_stored_files,
)
from .config import config
from .QdrantIndexer import QdrantIndexer
from .models import ChunkType, EmbeddingType


class DocumentIndexer:
    """
    Object that reacts to filesystem events (document creation/modification/deletion)
    and updates the databases

    """

    def __init__(self):
        # Load embedding model
        self.model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            trust_remote_code=config.EMBEDDING_MODEL_TRUST_REMOTE_CODE,
            backend="torch",
        )
        self.vector_size = self.model.get_sentence_embedding_dimension()
        self.doc_factory = DocumentFactory()
        self.doc_factory.set_embedding_model(self.model)

        # Initialize Qdrant
        self.qdrant = QdrantIndexer(vector_size=self.vector_size)
        self.qdrant.create_collection_if_missing()

        # Lock around state & indexing operations
        self.lock = threading.Lock()

    def extract_text(
        self, abspath: Path
    ) -> Iterable[Tuple[List[ChunkType], List[EmbeddingType], dict]]:
        """Extract chunks, embeddings and metadata from file path

        Args:
            abspath: Path to a file to analyse

        Yields:
            A tuple with a list of chunks, the corresponding list of embeddings, and the file metadata

        """
        for chunks, embeddings, file_metadata in self.doc_factory.processDocument(abspath):
            yield chunks, embeddings, file_metadata

    def process_file(self, filepath: Path):
        """
        Extract text, chunk, embed, and upsert into Qdrant.

        Args:
            filepath: Path to the file to be analyzed

        """
        stat = os.path.getmtime(filepath)
        stored = get_stored_timestamp(filepath)
        if stored is not None and stored == stat:
            # No change
            return

        logger.info(72 * "=")
        logger.info(f"[INDEX] Processing changed file: '{filepath}'")
        nb_emb = 0
        for chunks, embeddings, file_metadata in self.extract_text(filepath):
            # Upsert into Qdrant
            self.qdrant.record_embeddings(chunks, embeddings, file_metadata)
            nb_emb += len(embeddings)

        # Update state DB
        set_stored_timestamp(filepath, stat)
        logger.info(f"[INDEX] Upserted {nb_emb} vectors")

    def remove_file(self, filepath: Path):
        """
        Delete all vectors whose payload.source == this file's absolute path.
        We identify by regenerating all chunk IDs for old state—but since we store
        last-modified in SQLite, we know it existed before; we'll iterate over state DB
        to remove associated chunk IDs. Simpler: query by payload.source in Qdrant.

        Args:
            filepath: Path to the file to be analyzed, relative to DOCS_PATH

        """
        logger.info(f"[DELETE] Removing file from index: '{filepath}'")

        # Query Qdrant for all points with payload.source == abspath
        # filter_ = {"must": [{"key": "source", "match": {"value": abspath}}]}
        filter_ = Filter(must=[FieldCondition(key="source", match=MatchValue(value=str(filepath)))])

        # Retrieve IDs matching that filter
        hits = self.qdrant.search(limit=1000, query_filter=filter_)

        ids_to_delete = [hit.id for hit in hits]
        if ids_to_delete:
            self.qdrant.delete(ids_to_delete)
            logger.info(f"[DELETE] Removed {len(ids_to_delete)} vectors")

        # Remove from state DB
        delete_stored_file(filepath)

    def initial_scan(self):
        """
        On startup, walk entire DOCS_PATH and index any new/modified files.
        Also, find any entries in state DB that no longer exist on disk, and remove them.
        """
        logger.info("Performing initial scan of documents folder...")

        # 1. Build a set of all file paths on disk
        disk_files: list[Path] = []
        for ext in ("*.pdf", "*.docx", "*.xlsx", "*.xlsm", "*.md", "*.txt"):
            disk_files.extend(config.DOCS_PATH.rglob(ext))
        for ext in ("*.pdf", "*.docx", "*.xlsx", "*.xlsm", "*.md", "*.txt"):
            disk_files.extend(config.EMAILS_PATH.rglob(ext))
        disk_files = [p.resolve() for p in disk_files]

        # 2. For each file on disk, check timestamp vs. state DB
        files_to_index = []
        for file_path in disk_files:
            stored = get_stored_timestamp(file_path)
            modified = os.path.getmtime(str(file_path))
            if stored is None or stored != modified:
                files_to_index.append(file_path)

        # 3. For each modified file on disk, process its chunks
        tot_nb_files = len(files_to_index)
        for n_file, file_path in enumerate(files_to_index):
            logger.info(f"Initial indexation of {n_file}/{tot_nb_files} - '{file_path}'")
            stored = get_stored_timestamp(file_path)
            modified = os.path.getmtime(str(file_path))
            self.process_file(file_path)

        # 3. For each file in state DB, if not on disk anymore, delete from Qdrant
        for relpath in list_stored_files():
            abspath = config.DOCS_PATH / relpath
            if not abspath.exists():
                # Remove from Qdrant
                self.remove_file(relpath)

    def __on_created_or_modified(self, event: FileSystemEvent):
        if event.is_directory:
            return

        filepath = Path(event.src_path)
        if not self.doc_factory.filter_file(filepath):
            return

        with self.lock:
            # Small delay to allow file write to finish
            time.sleep(0.5)
            self.process_file(filepath)

    def __on_deleted(self, event: FileSystemEvent):
        if event.is_directory:
            return

        filepath = Path(event.src_path)
        if not self.doc_factory.filter_file(filepath):
            return

        with self.lock:
            self.remove_file(filepath)

    def __on_moved(self, event: FileSystemEvent):
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
        """
        Launch the filesystem monitoring as a non blocking thread

        """
        event_handler = FileSystemEventHandler()
        event_handler.on_created = self.__on_created_or_modified
        event_handler.on_modified = self.__on_created_or_modified
        event_handler.on_moved = self.__on_moved
        event_handler.on_deleted = self.__on_deleted

        # Files observer
        self.__docs_observer = Observer()
        self.__docs_observer.schedule(event_handler, path=str(config.DOCS_PATH), recursive=True)
        self.__docs_observer.start()

        logger.info(f"Started file watcher on: '{config.DOCS_PATH}'")

        # Emails observer
        self.__emails_observer = Observer()
        self.__emails_observer.schedule(event_handler, path=str(config.EMAILS_PATH), recursive=True)
        self.__emails_observer.start()

        logger.info(f"Started emails watcher on: '{config.EMAILS_PATH}'")

        self.__docs_observer.join()
        self.__emails_observer.join()
