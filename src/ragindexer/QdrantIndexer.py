import hashlib
import time
from typing import Optional, List, Sequence, Union
import uuid

from qdrant_client.conversions import common_types as types
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    PointIdsList,
)

from . import logger
from .config import config
from .models import ChunkType, EmbeddingType


# === Qdrant helper ===
class QdrantIndexer:
    """Qdrant client that handles database operations based on the configuration

    Args:
        vector_size: Size of the embedding vectors

    """

    def __init__(self, vector_size: int):
        self.__client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
        self.vector_size = vector_size
        self.__create_collection_if_missing()

    def info(self) -> types.CollectionInfo:
        info = self.__client.get_collection(collection_name=config.COLLECTION_NAME)
        return info

    def search(
        self,
        query_vector: Optional[
            Union[
                Sequence[float],
                tuple[str, list[float]],
                types.NamedVector,
                types.NamedSparseVector,
                types.NumpyArray,
            ]
        ] = None,
        limit: Optional[int] = 10,
        query_filter: Optional[types.Filter] = None,
    ):
        """Search a vector in the database
        See https://qdrant.tech/documentation/concepts/search/
        and https://qdrant.tech/documentation/concepts/filtering/ for more details

        Args:
            query_vector: Search for vectors closest to this. If None, allows listing ids
            limit: How many results return
            query_filter:
                - Exclude vectors which doesn't fit given conditions.
                - If `None` - search among all vectors

        Returns:
            List of found close points with similarity scores.

        """
        if query_vector is None:
            query_vect = [0.0] * self.vector_size  # dummy vector; we only want IDs
        else:
            query_vect = query_vector

        hits = self.__client.query_points(
            collection_name=config.COLLECTION_NAME,
            query=query_vect,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        ).points
        return hits

    def __create_collection_if_missing(self):
        """Creates the collection provided in the COLLECTION_NAME environment variable, if not already created"""
        existing = [c.name for c in self.__client.get_collections().collections]
        if config.COLLECTION_NAME not in existing:
            logger.info(f"Creating Qdrant collection : '{config.COLLECTION_NAME}'...")
            self.__client.recreate_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                on_disk_payload=True,
            )
            logger.info("... Done")

    def delete(self, ids: List[str]):
        """Deletes selected points from collection

        Args:
            ids: Selects points based on list of IDs

        """
        if ids:
            pil = PointIdsList(points=ids)
            self.__client.delete(collection_name=config.COLLECTION_NAME, points_selector=pil)

    def record_embeddings(
        self,
        k_page: int,
        chunks: List[ChunkType],
        embeddings: List[EmbeddingType],
        file_metadata: dict,
    ):
        """
        Update or insert a new chunk into the collection.

        Args:
            chunks: List of chunks to record
            embeddings: The corresponding list of vectors to record
            file_metadata: Original file's information

        """
        filepath = file_metadata["abspath"]

        points: list[PointStruct] = []
        # Use MD5 of path + chunk index as unique point ID
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            file_hash = hashlib.md5(f"{filepath}::{k_page}::{idx}".encode("utf-8")).hexdigest()
            pid = str(uuid.UUID(int=int(file_hash, 16)))
            payload = {
                "source": str(filepath),
                "chunk_index": idx,
                "text": chunk,
                "page": k_page,
                "ocr_used": file_metadata.get("ocr_used", False),
            }
            points.append(PointStruct(id=pid, vector=emb, payload=payload))

        # Upsert into Qdrant
        if len(points) > 0:
            self.__client.upsert(collection_name=config.COLLECTION_NAME, points=points)
            time.sleep(0.1)
