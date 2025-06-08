from pathlib import Path
from solus import Singleton

from .Document import Document
from .XlsDocument import XlsDocument
from .PdfDocument import PdfDocument
from .MarkdownDocument import MarkdownDocument
from .DocDocument import DocDocument


class DocumentFactory(Singleton):
    def __init__(self):
        self.__association = {}

    def register(self, ext: str, cls: type):
        self.__association[ext] = cls

    def getBuild(self, ext: str) -> Document:
        return self.__association[ext]

    def createDocument(self, abspath: Path) -> Document:
        ext = abspath.suffix
        cls = self.getBuild(ext)
        return cls(abspath)


DocumentFactory().register(".doc", DocDocument)
DocumentFactory().register(".docx", DocDocument)
DocumentFactory().register(".docm", DocDocument)

DocumentFactory().register(".xls", XlsDocument)
DocumentFactory().register(".xlsx", XlsDocument)
DocumentFactory().register(".xlsm", XlsDocument)

DocumentFactory().register(".pdf", PdfDocument)

DocumentFactory().register(".txt", MarkdownDocument)
DocumentFactory().register(".md", MarkdownDocument)
