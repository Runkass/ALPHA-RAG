"""Centralized configuration from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = "yandex"  # yandex | gigachat | openrouter | ollama

    yandex_api_key: str = ""
    yandex_folder_id: str = ""
    yandex_base_url: str = "https://llm.api.cloud.yandex.net/v1"
    yandex_model: str = "yandexgpt-lite/latest"

    gigachat_credentials: str = ""
    gigachat_base_url: str = "https://gigachat.devices.sberbank.ru/api/v1"
    gigachat_model: str = "GigaChat"
    gigachat_verify_ssl: bool = False

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openrouter/auto"
    openrouter_site_url: str = ""
    openrouter_app_name: str = "alpha-rag"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_M"
    ollama_api_key: str = "ollama"

    dense_backend: str = "faiss"  # faiss | tfidf
    fastembed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    fastembed_rerank_model: str = "BAAI/bge-reranker-base"

    chunk_size: int = Field(default=1000, ge=50)
    chunk_overlap: int = Field(default=100, ge=0)
    top_k_retrieve: int = Field(default=30, ge=1)
    top_k_final: int = Field(default=8, ge=1)
    max_context_chars: int = Field(default=5000, ge=500)
    rrf_k: int = Field(default=60, ge=1)
    min_rrf_score: float = Field(default=0.003, ge=0.0)
    refuse_rerank_hard: float = -7.0
    refuse_rerank_soft: float = -3.5
    refuse_enabled: bool = True
    keyword_min_overlap: int = Field(default=1, ge=0)
    soft_fallback_max_len: int = Field(default=260, ge=50)
    max_answer_chars: int = Field(default=900, ge=100)
    answerability_enabled: bool = True
    answerability_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    embed_batch_size: int = Field(default=32, ge=1)

    websites_csv: str = "websites.csv"
    questions_csv: str = "questions.csv"
    artifacts_dir: str = "artifacts"
    cache_db: str = "data/cache/answers.db"

    llm_temperature: float = Field(default=0.15, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=280, ge=32)
    llm_concurrency: int = Field(default=6, ge=1)
    yandex_request_delay_ms: int = Field(default=0, ge=0)
    reranker_enabled: bool = True
    cache_batch_size: int = Field(default=50, ge=1)

    @property
    def websites_path(self) -> Path:
        return PROJECT_ROOT / self.websites_csv

    @property
    def questions_path(self) -> Path:
        return PROJECT_ROOT / self.questions_csv

    @property
    def artifacts_path(self) -> Path:
        return PROJECT_ROOT / self.artifacts_dir

    @property
    def chunks_path(self) -> Path:
        return self.artifacts_path / "chunks.parquet"

    @property
    def faiss_path(self) -> Path:
        return self.artifacts_path / "index.faiss"

    @property
    def tfidf_path(self) -> Path:
        return self.artifacts_path / "tfidf.pkl"

    @property
    def bm25_path(self) -> Path:
        return self.artifacts_path / "bm25.pkl"

    @property
    def cache_db_path(self) -> Path:
        return PROJECT_ROOT / self.cache_db


@lru_cache
def get_settings() -> Settings:
    return Settings()
