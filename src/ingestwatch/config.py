from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=True, env_file=".env", env_file_encoding="utf-8", extra="allow"
    )

    LOGLEVEL: str
    QDRANT_HOST: str
    QDRANT_PORT: int
    OPENAI_API_KEY: str
    DOCS_PATH: Path
    COLLECTION_NAME: str
    EMBEDDING_MODEL: str
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int


config = Config()
