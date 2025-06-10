import sys

import nltk
import torch

from .index_database import initialize_state_db
from .config import config
from . import logger
from .frontend import ChatDocFontend
from .DocumentIndexer import DocumentIndexer


def main():
    torch.set_num_threads(config.TORCH_NUM_THREADS)

    # === Ensure NLTK punkt is available ===
    nltk.download("punkt_tab")
    nltk.download("punkt")

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

    frontend = ChatDocFontend()
    frontend.start()


if __name__ == "__main__":
    main()
