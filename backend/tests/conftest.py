"""Seam fixtures: a REAL VectorStore-over-Chroma built on a DeterministicEmbedder
(no model, deterministic, fast) + a tiny grounded corpus + ``DeterministicLlm``
that echoes one retrieved chunk — so tests assert on THE graded generation seam
(spec §6/§7 contract) without ever touching bge-m3 or Ollama.

The seam-under-test is ``main.build_app(store=..., llm=...)`` — the SAME call
shape production uses, just with test doubles injected through the same
parameters. Nothing here reaches a model, the network, or the real store dir.

Chunk metadata is written in EXACTLY the notebook's shape (cell 5 ``chunk_row``):
``doc_id / collection / section / seq / chunk_id / lang / source`` — the same
key set the backend reads back on ``search``.
"""

from __future__ import annotations

import pytest

from backend.app.chunking import DocumentChunk
from backend.app.main import build_app
from backend.app.retrieval import DeterministicEmbedder, VectorStore
from backend.app.generation import DeterministicLlm


def _row_chunk(
    doc_id: str,
    collection: str,
    section: str,
    text: str,
    lang: str,
    source: str,
) -> DocumentChunk:
    """Build a chunk carrying the notebook's row-chunk metadata contract."""
    return DocumentChunk(
        doc_id=doc_id,
        collection=collection,
        seq=0,
        section=section,
        text=text,
        metadata={
            "doc_id": doc_id,
            "collection": collection,
            "section": section,
            "seq": 0,
            "chunk_id": f"{doc_id}/000000",
            "lang": lang,
            "source": source,
        },
    )


@pytest.fixture
def store(tmp_path) -> VectorStore:
    """A persisted VectorStore over a Chroma dir holding a tiny bilingual
    (ar + latin drug names) medical grounding corpus, shaped like the notebook's
    persisted store output."""
    s = VectorStore(
        persist_dir=str(tmp_path / "chroma"),
        embedder=DeterministicEmbedder(dim=64, n_gram=3),
    )
    s.load()  # the seam loads ONCE at startup; fixture does the same (§7)
    s.add_documents(
        [
            _row_chunk(
                doc_id="ma::rx__all_prescriptions_clean_json::000000",
                collection="medical_docs",
                section="dispensing",
                text="دواء الأسبيرين (ASA) 100 ملغ يُؤخذ قرصاً واحداً يومياً بعد الطعام.",
                lang="ar",
                source="rx/all_prescriptions_clean.json",
            ),
            _row_chunk(
                doc_id="ma::rx__all_prescriptions_clean_json::000001",
                collection="medical_docs",
                section="dispensing",
                text="باراسيتامول: الحد الأقصى 4 غرام يومياً للبالغين، لا يتجاوز 8 أقراص من 500 ملغ.",
                lang="en",
                source="rx/all_prescriptions_clean.json",
            ),
            _row_chunk(
                doc_id="ma::en__medquad_csv::000002",
                collection="medical_docs",
                section="hypertension",
                text="الميتفورمين يُبدأ بـ 500 ملغ مرة واحدة يومياً مع الوجبات لتقليل أعراض الجهاز الهضمي.",
                lang="ar",
                source="en/medquad.csv",
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