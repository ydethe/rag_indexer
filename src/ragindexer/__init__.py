"""

.. include:: ../../README.md

# CLI usage

ragindexer comes with a CLI tool called ragindexer.

# Testing

## Run the tests

To run tests, just run:

    pytest

## Test reports

[See test report](../tests/report.html)

[See coverage](../coverage/index.html)

.. include:: ../../CHANGELOG.md

"""

import sys
import logging

from pythonjsonlogger.json import JsonFormatter

from .config import config


# création de l'objet logger qui va nous servir à écrire dans les logs
logger = logging.getLogger("ragindexer_logger")
logger.setLevel(config.LOGLEVEL.upper())

# Create stream handler for stdout
logHandler = logging.StreamHandler(sys.stdout)

# JSON formatter
formatter = JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")

logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
