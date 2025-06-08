from pathlib import Path
from typing import List, Tuple
from solus import Singleton
from sentence_transformers import SentenceTransformer

from .Document import Document
from .XlsDocument import XlsDocument
from .PdfDocument import PdfDocument
from .MarkdownDocument import MarkdownDocument
from .DocDocument import DocDocument


class DocumentFactory(Singleton):
    def __init__(self):
        self.__association = {}
        self.__embedding_model = None

    def register(self, ext: str, cls: type):
        self.__association[ext] = cls

    def getBuild(self, ext: str) -> Document:
        return self.__association[ext]

    def set_embedding_model(self, embedding_model: SentenceTransformer):
        self.__embedding_model = embedding_model

    def processDocument(self, abspath: Path) -> Tuple[List[str], List[List[float]], dict]:
        ext = abspath.suffix
        cls = self.getBuild(ext)
        doc: Document = cls(abspath)
        chunks, embeddings, file_metadata = doc.process(self.__embedding_model)
        return chunks, embeddings, file_metadata


DocumentFactory().register(".doc", DocDocument)
DocumentFactory().register(".docx", DocDocument)
DocumentFactory().register(".docm", DocDocument)

DocumentFactory().register(".xls", XlsDocument)
DocumentFactory().register(".xlsx", XlsDocument)
DocumentFactory().register(".xlsm", XlsDocument)

DocumentFactory().register(".pdf", PdfDocument)

DocumentFactory().register(".txt", MarkdownDocument)
DocumentFactory().register(".md", MarkdownDocument)
