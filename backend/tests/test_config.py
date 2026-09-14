"""config.py — backend Settings contract, matched to the notebook's cell 3.

The backend defaults MUST mirror the notebook's env-driven defaults
(COLLECTION / MODEL_NAME / TOP_K / MIN_SCORE, store dir) — the notebook is the
index side, this config is the query side of the SAME pipeline. Test doubles
use ``Settings(_env_file=None)`` so these tests never read a stray .env.
"""

from __future__ import annotations

from backend.app.config import Settings


class TestDefaults:
    def test_store_path_defaults_to_notebook_export_dir(self) -> None:
        s = Settings(_env_file=None)
        assert s.vector_store_path.endswith("data/vector_store/chroma")

    def test_embedder_matches_notebook(self) -> None:
        s = Settings(_env_file=None)
        assert s.embed_model == "BAAI/bge-m3"

    def test_collection_matches_notebook(self) -> None:
        assert Settings(_env_file=None).default_collection == "medical_docs"

    def test_retrieval_defaults_match_notebook(self) -> None:
        s = Settings(_env_file=None)
        assert s.top_k == 5
        assert s.min_score == 0.35

    def test_ollama_defaults(self) -> None:
        s = Settings(_env_file=None)
        assert s.ollama_url == "http://localhost:11434"
        assert s.ollama_model == "command-r7b-arabic"

    def test_cors_allows_only_local_frontend_origins(self) -> None:
        s = Settings(_env_file=None)
        assert "http://localhost:8501" in s.cors_origins
        assert "http://localhost:8502" in s.cors_origins

    def test_language_default_arabic_primary(self) -> None:
        assert Settings(_env_file=None).language == "ar"


class TestDemoEmbedder:
    def test_demo_flag_false_for_real_model(self) -> None:
        assert Settings(_env_file=None, embed_model="BAAI/bge-m3").is_demo_embedder is False

    def test_demo_flag_true_for_test_markers(self) -> None:
        for marker in ("", "TEST", "test"):
            assert Settings(_env_file=None, embed_model=marker).is_demo_embedder is True


class TestEnvOverrides:
    def test_env_override_store_path(self, monkeypatch) -> None:
        monkeypatch.setenv("VECTOR_STORE_PATH", "/tmp/store/chroma")
        assert Settings(_env_file=None).vector_store_path == "/tmp/store/chroma"

    def test_env_override_ollama_target(self, monkeypatch) -> None:
        monkeypatch.setenv("OLLAMA_URL", "http://10.0.0.2:11434")
        monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5-med")
        s = Settings(_env_file=None)
        assert s.ollama_url == "http://10.0.0.2:11434"
        assert s.ollama_model == "qwen2.5-med"

    def test_env_override_retrieval_tuning(self, monkeypatch) -> None:
        monkeypatch.setenv("TOP_K", "7")
        monkeypatch.setenv("MIN_SCORE", "0.40")
        s = Settings(_env_file=None)
        assert s.top_k == 7
        assert s.min_score == 0.40