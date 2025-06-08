import unittest

from ingestwatch.config import config
from ingestwatch.__main__ import main


class TestIngestWatch(unittest.TestCase):
    def test_config(self):
        print(config)

    def test_main(self):
        main()


if __name__ == "__main__":
    a = TestIngestWatch()

    # a.test_config()
    a.test_main()

# ingestion-1  | {"asctime": "2025-06-08 10:58:20,292", "levelname": "ERROR", "name": "ingestwatch_logger", "message": "Error processing '/docs/Scans/DessinElsa.pdf': cannot unpack non-iterable NoneType object"}
# ingestion-1  | {"asctime": "2025-06-08 11:02:56,626", "levelname": "ERROR", "name": "ingestwatch_logger", "message": "Error processing '/docs/Technique/BayesianDataAnalysis.pdf': 'PdfDocument' object has no attribute 'model'"}
