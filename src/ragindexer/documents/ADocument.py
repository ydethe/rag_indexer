from abc import abstractmethod, ABC
from pathlib import Path
from typing import List, Tuple, Iterable

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

from ..config import config
from ..models import ChunkType, EmbeddingType


class ADocument(ABC):
    """
    Handle documents based on their extension

    Args:
        abspath: Path to the file to handle

    """

    def __init__(self, abspath: Path):
        self.__abspath = abspath

    def get_abs_path(self) -> Path:
        """
        Get the abspath to the handled file

        Returns:
            Path to the handled file

        """
        return self.__abspath

    @abstractmethod
    def iterate_raw_text(self) -> Iterable[Tuple[str, dict]]:
        """
        Abstract method that should implement the concrete way to handle the file.

        Yields:
            A tuple with extracted text and file metadata

        """

    def __get_chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[ChunkType]:
        """
        Splits text into overlapping chunks of ~chunk_size characters, aligned on sentences.
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sent in sentences:
            if len(current_chunk) + len(sent) + 1 <= chunk_size:
                current_chunk += " " + sent if current_chunk else sent
            else:
                chunks.append(current_chunk)
                # Start new chunk: include overlap
                overlap_text = (
                    current_chunk[-chunk_overlap:]
                    if chunk_overlap < len(current_chunk)
                    else current_chunk
                )
                current_chunk = overlap_text + " " + sent

        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def __get_embeddings(
        self, text: str, embedding_model: SentenceTransformer
    ) -> Tuple[List[ChunkType], List[EmbeddingType]]:
        chunks = self.__get_chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        embeddings = embedding_model.encode(chunks, device="cpu", show_progress_bar=False).tolist()
        return chunks, embeddings

    def process(
        self, embedding_model: SentenceTransformer
    ) -> Iterable[Tuple[List[ChunkType], List[EmbeddingType], dict]]:
        for text, file_metadata in self.iterate_raw_text():
            file_metadata["abspath"] = self.get_abs_path()

            chunks, embeddings = self.__get_embeddings(text, embedding_model)
            while "" in chunks:
                chunks.remove("")

            yield chunks, embeddings, file_metadata
