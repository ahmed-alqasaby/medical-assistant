"""Frontend client tests — M4 acceptance: env-var-only URL, visible citations,
loading & error states, end-to-end against the real backend seam.

The client is the seam-under-test; the FastAPI app comes from the fixture so
"runs locally against the running backend with a real end-to-end demo
question" is exercised over the very backend the UI talks to (ASGI transport).
The client is async; tests drive it through ``asyncio.run`` — the same
boundary the Streamlit app uses.
"""

from __future__ import annotations

import asyncio

import pytest

from frontend.client import BackendClient, BackendError


def test_from_env_requires_backend_url(monkeypatch) -> None:
    monkeypatch.delenv("BACKEND_URL", raising=False)
    with pytest.raises(BackendError):
        BackendClient.from_env()


def test_from_env_reads_url_from_env(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://localhost:8000")
    assert BackendClient.from_env()._base == "http://localhost:8000"


def test_health_reports_store_loaded(client) -> None:
    body = asyncio.run(client.health())
    assert body["status"] == "ok"
    assert body["store"]["loaded"] is True
    assert body["store"]["chunks"] >= 1


def test_end_to_end_query_returns_cited_answer(client) -> None:
    payload = asyncio.run(client.query("ما الجرعة القصوى للباراسيتامول؟"))
    assert payload["refuse"] is False
    assert payload["answer"]
    assert payload["citations"], "citations must be visible, not hidden"
    c = payload["citations"][0]
    assert c["doc_id"] and c["quoted"] and "score" in c


def test_query_exposes_empty_question_as_backend_error(client) -> None:
    with pytest.raises(BackendError) as exc:
        asyncio.run(client.query("   "))
    assert "422" in str(exc.value)


def test_connection_refused_is_a_backend_error(noclient) -> None:
    with pytest.raises(BackendError):
        asyncio.run(noclient.query("هل تعمل؟"))


def test_typed_refusal_is_a_success_payload_not_an_error(tmp_path) -> None:
    """M4 acceptance: refusal arrives as refuse=True in a 200 response body —
    the UI must render it as an honest 'I don't know', not crash or throw."""
    import httpx

    from backend.app.main import build_app
    from backend.app.retrieval import DeterministicEmbedder, VectorStore
    from backend.app.generation import DeterministicLlm

    empty = VectorStore(
        persist_dir=str(tmp_path / "empty-chroma"),
        embedder=DeterministicEmbedder(dim=64, n_gram=3),
    )
    empty.load()
    app = build_app(store=empty, llm=DeterministicLlm(), min_score=0.35)
    c = BackendClient("http://testserver", transport=httpx.ASGITransport(app=app))
    body = asyncio.run(c.query("ما هي تعاليم الطب الصيني؟"))
    assert body["refuse"] is True
    assert body["answer"] == ""
    assert "no grounded context" in (body.get("refuse_reason") or "")