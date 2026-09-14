"""M3 graded seam tests: GET /health and POST /query (happy + 422).

Spec §7 — the API contract, tested at the middle-layer generation seam with
test doubles (no bge-m3, no Ollama). Every assertion is external behavior:
status codes, answer presence, typed citations, typed refusal.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_reports_ok_with_store_loaded(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["store"]["loaded"] is True
    assert body["store"]["chunks"] >= 1


def test_query_happy_path_returns_grounded_cited_answer(client) -> None:
    resp = client.post("/query", json={"question": "كيف يُؤخذ الأسبيرين؟", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["refuse"] is False
    assert len(body["answer"]) > 0
    assert len(body["citations"]) >= 1
    cited = body["citations"][0]
    # citation resolves to a real retrieved source (doc-id registry contract)
    assert cited["doc_id"]
    assert cited["quoted"]


def test_query_english_answer_still_grounded(client) -> None:
    resp = client.post("/query", json={"question": "What is the maximum daily paracetamol dose?"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["answer"]) > 0
    assert len(body["citations"]) >= 1


def test_query_empty_question_422(client) -> None:
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 422


def test_query_whitespace_only_question_422(client) -> None:
    resp = client.post("/query", json={"question": "   \n  "})
    assert resp.status_code == 422


def test_query_bad_top_k_422(client) -> None:
    resp = client.post("/query", json={"question": "سؤال", "top_k": 0})
    assert resp.status_code == 422


def test_query_missing_question_422(client) -> None:
    resp = client.post("/query", json={})
    assert resp.status_code == 422


def test_query_out_of_grounding_returns_typed_refusal(tmp_path) -> None:
    """The store has no knowledge about Max Pressure; the answer must be a
    typed refusal (200), never an invented reply or a 500."""
    from backend.app.main import build_app
    from backend.app.retrieval import DeterministicEmbedder, VectorStore
    from backend.app.generation import DeterministicLlm

    empty = VectorStore(
        persist_dir=str(tmp_path / "empty-chroma"),
        embedder=DeterministicEmbedder(dim=64, n_gram=3),
    )
    empty.load()
    app = build_app(store=empty, llm=DeterministicLlm(), min_score=0.35)
    with TestClient(app) as c:
        resp = c.post("/query", json={"question": "ما هي أكبر غابة استوائية في العالم؟"})
    assert resp.status_code == 200, "refusal is a 200-shaped outcome, not an error"
    body = resp.json()
    assert body["refuse"] is True
    assert body["answer"] == ""
    assert body["citations"] == []
    assert "no grounded context" in (body["refuse_reason"] or "")


def test_query_reports_store_meta(tmp_path) -> None:
    """meta.store_chunks mirrors the loaded store count (§7 observability)."""
    from backend.app.main import build_app
    from backend.app.retrieval import DeterministicEmbedder, VectorStore
    from backend.app.generation import DeterministicLlm

    empty = VectorStore(
        persist_dir=str(tmp_path / "meta-chroma"),
        embedder=DeterministicEmbedder(dim=64, n_gram=3),
    )
    empty.load()
    app = build_app(store=empty, llm=DeterministicLlm(), min_score=0.35)
    with TestClient(app) as c:
        resp = c.post("/query", json={"question": "أي سؤال"})
    body = resp.json()
    assert body["meta"]["store_chunks"] == 0