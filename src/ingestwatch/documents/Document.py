from abc import abstractmethod, ABC
from pathlib import Path
from typing import List, Tuple

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

from .. import logger
from ..config import config


class Document(ABC):
    def __init__(self, abspath: Path):
        self.__abspath = abspath

    def get_abs_path(self) -> Path:
        return self.__abspath

    @abstractmethod
    def get_raw_text(self) -> Tuple[str, dict]:
        pass

    def __get_chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Splits text into overlapping chunks of ~chunk_size characters, aligned on sentences.
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        logger.info("Building chunks")
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
    ) -> Tuple[List[str], List[List[float]]]:
        chunks = self.__get_chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        logger.info(f"Embedding {len(chunks)} chunks")
        embeddings = embedding_model.encode(chunks, device="cpu", show_progress_bar=True).tolist()
        return chunks, embeddings

    def process(
        self, embedding_model: SentenceTransformer
    ) -> Tuple[List[str], List[List[float]], dict]:
        text, file_metadata = self.get_raw_text()
        file_metadata["abspath"] = self.get_abs_path()

        if not text:
            logger.warning(f"No text extracted; skipping: '{self.get_abs_path()}'")
            return [], [], file_metadata

        chunks, embeddings = self.__get_embeddings(text, embedding_model)

        return chunks, embeddings, file_metadata
