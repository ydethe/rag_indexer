import os
import sqlite3
from pathlib import Path
from typing import Optional

from . import logger
from .config import config


def initialize_state_db():
    """
    Initialize the sqlite database

    """
    os.makedirs(os.path.dirname(config.STATE_DB_PATH), exist_ok=True)

    logger.info(f"Using sqlite database '{config.STATE_DB_PATH}'")

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


def get_stored_timestamp(relpath: Path) -> Optional[float]:
    """
    Get the stored timestamp for the given path

    Args:
        relpath: Path to a file that has already been processed

    Returns:
        The timestamp of last processing if found. None otherwise

    """
    conn = sqlite3.connect(config.STATE_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT last_modified FROM files WHERE path = ?", (str(relpath),))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


def set_stored_timestamp(relpath: Path, ts: float):
    """
    Stores the processing timestamp for the given path

    Args:
        relpath: Path to a file that has already been processed
        ts: The timestamp of last processing

    """
    conn = sqlite3.connect(config.STATE_DB_PATH)
    c = conn.cursor()
    c.execute("REPLACE INTO files (path, last_modified) VALUES (?, ?)", (str(relpath), ts))
    conn.commit()
    conn.close()


def delete_stored_file(relpath: Path):
    """
    Delete the given path

    Args:
        relpath: Path to a file that has already been processed

    """
    conn = sqlite3.connect(config.STATE_DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM files WHERE path = ?", (str(relpath),))
    conn.commit()
    conn.close()


def list_stored_files(absolute: bool = False) -> list[Path]:
    """
    List all paths stored in the database

    Args:
        absolute: True to return absolute paths

    Returns:
        The list of all paths stored in the database

    """
    conn = sqlite3.connect(config.STATE_DB_PATH)
    c = conn.cursor()
    c.execute("SELECT path FROM files")
    rows = c.fetchall()
    conn.close()

    files_list = []
    for (stored_path,) in rows:
        relpath = Path(stored_path)
        if absolute:
            files_list.append(config.DOCS_PATH / relpath)
        else:
            files_list.append(relpath)

    return files_list
