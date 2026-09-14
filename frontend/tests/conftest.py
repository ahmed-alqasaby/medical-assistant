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


def _row_chunk(doc_id, section, text, lang, source) -> DocumentChunk:
    """Notebook-shaped row chunk (cell 5 metadata contract)."""
    return DocumentChunk(
        doc_id=doc_id,
        collection="medical_docs",
        seq=0,
        section=section,
        text=text,
        metadata={
            "doc_id": doc_id,
            "collection": "medical_docs",
            "section": section,
            "seq": 0,
            "chunk_id": f"{doc_id}/000000",
            "lang": lang,
            "source": source,
        },
    )


@pytest.fixture
def store(tmp_path) -> VectorStore:
    s = VectorStore(
        persist_dir=str(tmp_path / "chroma"),
        embedder=DeterministicEmbedder(dim=64, n_gram=3),
    )
    s.load()
    s.add_documents(
        [
            _row_chunk(
                "ma::rx__all_prescriptions_clean_json::000000",
                "dispensing",
                "دواء الأسبيرين (ASA) 100 ملغ يُؤخذ قرصاً واحداً يومياً بعد الطعام.",
                "ar",
                "rx/all_prescriptions_clean.json",
            ),
            _row_chunk(
                "ma::rx__all_prescriptions_clean_json::000001",
                "dispensing",
                "باراسيتامول: الحد الأقصى 4 غرام يومياً للبالغين، لا يتجاوز 8 أقراص من 500 ملغ.",
                "en",
                "rx/all_prescriptions_clean.json",
            ),
            _row_chunk(
                "ma::en__medquad_csv::000002",
                "hypertension",
                "الميتفورمين يُبدأ بـ 500 ملغ مرة واحدة يومياً مع الوجبات لتقليل أعراض الجهاز الهضمي.",
                "ar",
                "en/medquad.csv",
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