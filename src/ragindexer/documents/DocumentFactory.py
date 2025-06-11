from pathlib import Path
from typing import Iterable, List, Tuple
from solus import Singleton
from sentence_transformers import SentenceTransformer

from ..models import ChunkType, EmbeddingType
from .ADocument import ADocument
from .XlsDocument import XlsDocument
from .PdfDocument import PdfDocument
from .MarkdownDocument import MarkdownDocument
from .DocDocument import DocDocument


class DocumentFactory(Singleton):
    def __init__(self):
        self.__association = {}
        self.__embedding_model = None

    def filter_file(self, path: Path) -> bool:
        if path.suffix not in self.__association.keys():
            return False

        if path.stem.startswith(".sftpgo-upload"):
            return False

        return True

    def register(self, ext: str, cls: type):
        self.__association[ext] = cls

    def getBuild(self, ext: str) -> ADocument:
        return self.__association[ext]

    def set_embedding_model(self, embedding_model: SentenceTransformer):
        self.__embedding_model = embedding_model

    def processDocument(
        self, abspath: Path
    ) -> Iterable[Tuple[List[ChunkType], List[EmbeddingType], dict]]:
        ext = abspath.suffix
        cls = self.getBuild(ext)
        doc: ADocument = cls(abspath)
        for chunks, embeddings, file_metadata in doc.process(self.__embedding_model):
            yield chunks, embeddings, file_metadata


DocumentFactory().register(".doc", DocDocument)
DocumentFactory().register(".docx", DocDocument)
DocumentFactory().register(".docm", DocDocument)

DocumentFactory().register(".xls", XlsDocument)
DocumentFactory().register(".xlsx", XlsDocument)
DocumentFactory().register(".xlsm", XlsDocument)

DocumentFactory().register(".pdf", PdfDocument)

DocumentFactory().register(".txt", MarkdownDocument)
DocumentFactory().register(".md", MarkdownDocument)
