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

    a.test_config()
    # a.test_main()
