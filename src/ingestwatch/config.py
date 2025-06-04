from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=True, env_file=".env", env_file_encoding="utf-8", extra="allow"
    )

    LOGLEVEL: str
    QDRANT_HOST: str
    QDRANT_PORT: int
    DOCS_PATH: Path
    STATE_DB_PATH: Path
    COLLECTION_NAME: str
    EMBEDDING_MODEL: str
    EMBEDDING_MODEL_TRUST_REMOTE_CODE: bool
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    OCR_LANG: str


config = Config()
