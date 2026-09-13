"""Backend configuration. Reads from environment / backend/.env.

Settings here drive store loading + the Ollama generation call; the
frontend reads BACKEND_URL (its own .env). Never hard-code values in code.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    vector_store_path: str = str(REPO_ROOT / "data" / "vector_store" / "chroma")
    embed_model: str = "BAAI/bge-m3"
    default_collection: str = "medical_docs"
    top_k: int = 5
    min_score: float = 0.35

    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "command-r7b-arabic"

    # Frontend origin(s). Streamlit runs on its own port + a DIFFERENT origin
    # than the backend; the seam must allow exactly these (§7 CORS).
    cors_origins: list[str] = ["http://localhost:8501", "http://localhost:8502"]

    # English gets an English-grounded turn; Arabic defaults.
    language: str = "ar"

    @property
    def is_demo_embedder(self) -> bool:
        return self.embed_model in ("", "TEST", "test")


settings = Settings()