"""Runtime configuration. Secrets come from the environment or a local .env file."""
import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = os.getenv("DATABASE_URL")
    qdrant_url: str = os.getenv("QDRANT_URL")
    qdrant_collection: str = "rfp_chunks"
    google_api_key: str = os.getenv("GOOGLE_API_KEY") or ""
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY") or ""
    llm_model: str = os.getenv("LLM_MODEL")
    llm_reasoning_enabled: bool = os.getenv("LLM_REASONING_ENABLED", "false").lower() in {"1", "true", "yes"}
    llm_request_timeout_seconds: float = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "120"))
    llm_thinking_level: str = os.getenv("LLM_THINKING_LEVEL") or ""
    data_dir: Path = os.getenv("DATA_DIR")
    output_dir: Path = os.getenv("OUTPUT_DIR")
    embedding_model: str = os.getenv("EMBEDDING_MODEL")
    embedding_dim: int = 384
    reranker_model: str = os.getenv("RERANKER_MODEL")
    chunk_max_tokens: int = 512
    search_candidate_k: int = 20
    rrf_k: int = 60
    api_base_url: str = os.getenv("API_BASE_URL")
    watcher_debounce_seconds: float = 5.0
    backoff_base_seconds: float = 0.5
    backoff_cap_seconds: float = 8.0
    io_max_attempts: int = 5
    validator_max_retries: int = 2
    ingestion_max_attempts: int = 3
    extract_section_concurrency: int = int(os.getenv("EXTRACT_SECTION_CONCURRENCY", "2"))
    extract_group_concurrency: int = int(os.getenv("EXTRACT_GROUP_CONCURRENCY", "2"))
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
