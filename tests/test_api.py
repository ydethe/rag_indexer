import unittest

from ingestwatch.ingest_watch import main


class TestIngestWatch(unittest.TestCase):
    def test_main(self):
        main()


if __name__ == "__main__":
    a = TestIngestWatch()

    a.test_main()
