import unittest

from ingestwatch.__main__ import main


class TestIngestWatch(unittest.TestCase):
    def test_main(self):
        main()


if __name__ == "__main__":
    a = TestIngestWatch()

    a.test_main()
