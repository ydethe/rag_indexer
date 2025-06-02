"""

.. include:: ../../README.md

# CLI usage

ingestwatch comes with a CLI tool called ingestwatch.

# Testing

## Run the tests

To run tests, just run:

    uv run pytest

## Test reports

[See test report](../tests/report.html)

[See coverage](../coverage/index.html)

.. include:: ../../CHANGELOG.md

"""


import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

from .config import config


# création de l'objet logger qui va nous servir à écrire dans les logs
logger = logging.getLogger("ingestwatch_logger")
logger.setLevel(config.LOGLEVEL.upper())

log_pth = Path("logs")
if not log_pth.exists():
    log_pth.mkdir(exist_ok=True)
file_handler = RotatingFileHandler("logs/ingestwatch.log", maxBytes=10e6, backupCount=5)
term_handler = RichHandler(rich_tracebacks=False)

logger.addHandler(file_handler)
logger.addHandler(term_handler)
