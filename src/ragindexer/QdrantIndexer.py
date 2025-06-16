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
        self.__client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            api_key=config.QDRANT_API_KEY,
            https=config.QDRANT_HTTPS,
        )

        self.vector_size = vector_size

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

    def create_collection_if_missing(self):
        """Creates the collection provided in the COLLECTION_NAME environment variable, if not already created"""
        existing = [c.name for c in self.__client.get_collections().collections]
        if config.COLLECTION_NAME not in existing:
            self.__client.recreate_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                on_disk_payload=True,
            )
            logger.info(
                f"Created Qdrant collection '{config.COLLECTION_NAME}' (size={self.vector_size})."
            )

    def delete(self, ids: List[str]):
        """Deletes selected points from collection

        Args:
            ids: Selects points based on list of IDs

        """
        if ids:
            pil = PointIdsList(points=ids)
            self.__client.delete(collection_name=config.COLLECTION_NAME, points_selector=pil)

    def record_embeddings(
        self, chunks: List[ChunkType], embeddings: List[EmbeddingType], file_metadata: dict
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
            # file_hash = hashlib.md5(f"{filepath}::{idx}".encode("utf-8")).hexdigest()
            # pid = str(uuid.UUID(int=int(file_hash, 16)))
            pid = str(uuid.uuid4())
            payload = {
                "source": str(filepath),
                "chunk_index": idx,
                "text": chunk,
                "ocr_used": file_metadata.get("ocr_used", False),
            }
            points.append(PointStruct(id=pid, vector=emb, payload=payload))

        # Upsert into Qdrant
        self.__client.upsert(collection_name=config.COLLECTION_NAME, points=points)
        time.sleep(0.1)
