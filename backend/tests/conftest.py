"""Seam fixtures: a REAL VectorStore-over-Chroma built on a DeterministicEmbedder
(no model, deterministic, fast) + a tiny grounded corpus + ``DeterministicLlm``
that echoes one retrieved chunk — so tests assert on THE graded generation seam
(spec §6/§7 contract) without ever touching bge-m3 or Ollama.

The seam-under-test is ``main.build_app(store=..., llm=...)`` — the SAME call
shape production uses, just with test doubles injected through the same
parameters. Nothing here reaches a model, the network, or the real store dir.
"""

from __future__ import annotations

import pytest

from backend.app.chunking import DocumentChunk
from backend.app.main import build_app
from backend.app.retrieval import DeterministicEmbedder, VectorStore
from backend.app.generation import DeterministicLlm


@pytest.fixture
def store(tmp_path) -> VectorStore:
    """A persisted VectorStore over a Chroma dir holding a tiny bilingual
    (ar + latin drug names) medical grounding corpus."""
    s = VectorStore(
        persist_dir=str(tmp_path / "chroma"),
        embedder=DeterministicEmbedder(dim=64, n_gram=3),
    )
    s.load()  # the seam loads ONCE at startup; fixture does the same (§7)
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
def client(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c