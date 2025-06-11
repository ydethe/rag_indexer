import sys

import nltk
import torch

from .index_database import initialize_state_db
from .config import config
from . import logger
from .DocumentIndexer import DocumentIndexer


def main():
    torch.set_num_threads(config.TORCH_NUM_THREADS)

    # === Ensure NLTK punkt is available ===
    nltk.download("punkt", download_dir="/code/nltk")
    nltk.data.path.append("/code/nltk")

    # Ensure state DB exists
    initialize_state_db()

    # Ensure documents folder exists
    if not config.DOCS_PATH.exists():
        logger.error(f"Documents folder not found: '{config.DOCS_PATH}'")
        sys.exit(1)

    indexer = DocumentIndexer()

    # Initial full scan
    indexer.initial_scan()

    # Start the filesystem watcher loop
    indexer.start_watcher()


if __name__ == "__main__":
    main()
