from typing import Optional, List, Sequence, Union

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


# === Qdrant helper ===
class QdrantIndexer:
    def __init__(self, vector_size: int):
        self.__client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            # api_key=config.QDRANT_API_KEY,
            https=False,
        )

        self.vector_size = vector_size

        # Check if collection exists; if not, create
        existing = [c.name for c in self.__client.get_collections().collections]
        if config.COLLECTION_NAME not in existing:
            # We’ll determine vector size after loading the model
            pass

    def search(
        self,
        query_vector: Union[
            Sequence[float],
            tuple[str, list[float]],
            types.NamedVector,
            types.NamedSparseVector,
            types.NumpyArray,
        ] = None,
        limit: int = 10,
        query_filter: Optional[types.Filter] = None,
    ):
        if query_vector is None:
            query_vect = ([0.0] * self.vector_size,)  # dummy vector; we only want IDs
        else:
            query_vect = query_vector

        hits = self.__client.search(
            collection_name=config.COLLECTION_NAME,
            query_vector=query_vect,
            limit=limit,
            query_filter=query_filter,
        )
        return hits

    def create_collection_if_missing(self):
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

    def upsert(self, points: List[PointStruct]):
        if points:
            self.__client.upsert(collection_name=config.COLLECTION_NAME, points=points)

    def delete(self, ids: List[str]):
        if ids:
            pil = PointIdsList(points=ids)
            self.__client.delete(collection_name=config.COLLECTION_NAME, points_selector=pil)
