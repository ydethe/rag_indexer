from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=True, env_file=".env", env_file_encoding="utf-8", extra="allow"
    )

    LOGLEVEL: str
    QDRANT_URL: str
    QDRANT_QUERY_LIMIT: int
    QDRANT_API_KEY: str
    DOCS_PATH: Path
    EMAILS_PATH: Path
    STATE_DB_PATH: Path
    COLLECTION_NAME: str
    DAV_ROOT: str
    EMBEDDING_MODEL: str
    EMBEDDING_MODEL_TRUST_REMOTE_CODE: bool
    OPEN_MODEL_ENDPOINT: str
    OPEN_MODEL_API_KEY: str
    OPEN_MODEL_PREF: str
    OPEN_MODEL_TEMPERATURE: float
    MIN_EXPECTED_CHAR: int
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    OCR_LANG: str
    TORCH_NUM_THREADS: int


config = Config()
