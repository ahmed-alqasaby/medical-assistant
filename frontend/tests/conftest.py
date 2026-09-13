"""Fixture seam for the frontend tests (mirrors backend/tests/conftest.py).

The frontend client is exercised against the REAL FastAPI seam app
(``main.build_app``) over an ASGI transport — a live server is never started,
and bge-m3/Ollama are never touched (deterministic doubles only).
"""

from __future__ import annotations

import httpx
import pytest

from backend.app.chunking import DocumentChunk
from backend.app.main import build_app
from backend.app.retrieval import DeterministicEmbedder, VectorStore
from backend.app.generation import DeterministicLlm


@pytest.fixture
def store(tmp_path) -> VectorStore:
    s = VectorStore(
        persist_dir=str(tmp_path / "chroma"),
        embedder=DeterministicEmbedder(dim=64, n_gram=3),
    )
    s.load()
    s.add_documents(
        [
            DocumentChunk(
                doc_id="rx-001",
                collection="rx_photos",
                seq=0,
                section="dispensing",
                text="دواء الأسبيرين (ASA) 100 ملغ يُؤخذ قرصاً واحداً يومياً بعد الطعام.",
                metadata={"doc_id": "rx-001", "collection": "rx_photos", "section": "dispensing"},
            ),
            DocumentChunk(
                doc_id="rx-002",
                collection="rx_photos",
                seq=0,
                section="dispensing",
                text="باراسيتامول: الحد الأقصى 4 غرام يومياً للبالغين، لا يتجاوز 8 أقراص من 500 ملغ.",
                metadata={"doc_id": "rx-002", "collection": "rx_photos", "section": "dispensing"},
            ),
            DocumentChunk(
                doc_id="rx-003",
                collection="medical_guidelines",
                seq=0,
                section="hypertension",
                text="الميتفورمين يُبدأ بـ 500 ملغ مرة واحدة يومياً مع الوجبات لتقليل أعراض الجهاز الهضمي.",
                metadata={"doc_id": "rx-003", "collection": "medical_guidelines", "section": "hypertension"},
            ),
        ]
    )
    return s


@pytest.fixture
def llm() -> DeterministicLlm:
    return DeterministicLlm()


@pytest.fixture
def app(store: VectorStore, llm: DeterministicLlm):
    return build_app(store=store, llm=llm, min_score=0.35)


@pytest.fixture
def client(app) -> "BackendClient":
    from frontend.client import BackendClient

    transport = httpx.ASGITransport(app=app)
    return BackendClient(base_url="http://testserver", transport=transport)


@pytest.fixture
def noclient(tmp_path) -> "BackendClient":
    """A client pointed at a port with no server (backend-down error state)."""
    from frontend.client import BackendClient

    return BackendClient("http://127.0.0.1:1")