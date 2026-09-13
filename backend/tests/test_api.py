"""M3 graded seam tests: GET /health and POST /query (happy + 422).

Spec §7 — the API contract, tested at the middle-layer generation seam with
test doubles (no bge-m3, no Ollama). Every assertion is external behavior:
status codes, answer presence, typed citations, typed refusal.
"""

from __future__ import annotations


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