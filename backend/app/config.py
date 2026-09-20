"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    top_k: int = 5
    relevance_threshold: float = 0.72

    chunk_size: int = 800
    chunk_overlap: int = 150

    upload_dir: Path = Path("./data/uploads")
    chroma_dir: Path = Path("./data/chroma")
    documents_db: Path = Path("./data/documents.json")

    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # When True, use deterministic fake embeddings (for tests without API key)
    use_fake_embeddings: bool = False

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
