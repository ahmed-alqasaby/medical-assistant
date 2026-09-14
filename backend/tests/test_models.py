"""models.py — the HTTP/response contract (§7). Deterministic, pure pydantic."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.models import Citation, QueryRequest, QueryResponse


class TestQueryRequest:
    def test_minimal_valid(self) -> None:
        req = QueryRequest(question="هل الأسبيرين آمن؟")
        assert req.question == "هل الأسبيرين آمن؟"
        assert req.top_k is None
        assert req.collection is None

    def test_empty_question_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="")

    def test_whitespace_question_passes_pydantic_but_guarded_in_app(self) -> None:
        # pydantic's min_length counts the whitespace; the app's normalize_query
        # guard is what turns this into a 422 (main.query) — assert the split.
        assert QueryRequest(question="   ").question == "   "

    def test_top_k_bounds(self) -> None:
        assert QueryRequest(question="x", top_k=1).top_k == 1
        assert QueryRequest(question="x", top_k=20).top_k == 20
        with pytest.raises(ValidationError):
            QueryRequest(question="x", top_k=0)
        with pytest.raises(ValidationError):
            QueryRequest(question="x", top_k=21)

    def test_collection_optional(self) -> None:
        assert QueryRequest(question="x", collection="medical_docs").collection == "medical_docs"


class TestCitation:
    def test_requires_all_fields(self) -> None:
        with pytest.raises(ValidationError):
            Citation(doc_id="x", collection="c", section="s")  # missing quoted, score
        Citation(doc_id="x", collection="c", section="s", quoted="q", score=0.9)

    def test_score_is_float(self) -> None:
        c = Citation(doc_id="x", collection="c", section="s", quoted="q", score=0.9)
        assert isinstance(c.score, float)


class TestQueryResponse:
    def test_defaults_are_typed_refusal_off(self) -> None:
        r = QueryResponse(question="q", answer="a", citations=[])
        assert r.refuse is False
        assert r.refuse_reason is None
        assert r.meta == {}

    def test_refusal_is_a_value_not_an_error(self) -> None:
        r = QueryResponse(question="q", answer="", citations=[], refuse=True, refuse_reason="no context")
        assert r.refuse is True
        assert r.refuse_reason == "no context"
        # must serialize as 200-shaped JSON the frontend can render
        body = r.model_dump()
        assert body["refuse"] is True and body["refuse_reason"] == "no context"