"""generation.py — THE graded seam (spec §1 rev / §6.4 / §7).

Tests assert the middle-layer contract: grounding gate -> typed refusal,
the lossless prompt block the LLM may ONLY cite from, REFUSAL handling, and
the mapping of a down/malformed Ollama into a typed refusal (never a 5xx).
OllamaLlm calls are mocked at ``httpx.post`` — no network, no model.
"""

from __future__ import annotations

import json

import httpx
import pytest

from backend.app.generation import (
    Answerer,
    DeterministicLlm,
    GroundingGate,
    LlmUnavailableError,
    OllamaLlm,
    build_prompt,
)
from backend.app.retrieval import RetrievedChunk


def _row(score: float, quoted: str = "الجرعة القصوى 4 غرام يومياً.", doc_id: str = "ma::ar__train_csv::000005") -> RetrievedChunk:
    return RetrievedChunk(
        doc_id=doc_id,
        collection="medical_docs",
        seq=5,
        section="dispensing",
        chunk_id=f"{doc_id}/000005",
        quoted=quoted,
        score=score,
    )


class TestGroundingGate:
    def test_empty_rows_refuse(self) -> None:
        gate = GroundingGate(min_score=0.35)
        assert gate.gate([]) is False
        assert "no grounded context" in gate.reason([])

    def test_below_threshold_refuse(self) -> None:
        assert GroundingGate(min_score=0.35).gate([_row(0.2)]) is False
        assert "below grounding threshold" in GroundingGate().reason([_row(0.2)])

    def test_equal_to_threshold_passes(self) -> None:
        assert GroundingGate(min_score=0.35).gate([_row(0.35)]) is True

    def test_top_above_threshold_passes(self) -> None:
        assert GroundingGate(min_score=0.35).gate([_row(0.8), _row(0.1)]) is True


class TestBuildPrompt:
    def test_ends_with_question_and_answer_markers(self) -> None:
        prompt = build_prompt("ما الجرعة؟", [_row(0.9)])
        assert prompt.rstrip().endswith("ANSWER:")
        assert "QUESTION: ما الجرعة؟" in prompt

    def test_context_block_is_lossless_json_with_typed_keys(self) -> None:
        prompt = build_prompt("سؤال", [_row(0.9)])
        raw = prompt.split("CONTEXT:\n", 1)[1].split("\n\n[quoted]", 1)[0]
        ctx = json.loads(raw)
        assert set(ctx[0].keys()) == {"doc_id", "collection", "section", "chunk_id", "quoted", "score"}
        assert ctx[0]["score"] == 0.9
        assert ctx[0]["chunk_id"].endswith("/000005")

    def test_arabic_system_primary(self) -> None:
        prompt = build_prompt("سؤال", [_row(0.9)], language="ar")
        assert "أجب بالعربية" in prompt
        assert "أسماء الأدوية اللاتينية" in prompt
        assert "لا تخترع" in prompt

    def test_english_system_flips(self) -> None:
        prompt = build_prompt("Question", [_row(0.9)], language="en")
        assert "Answer ONLY from the documents below" in prompt
        assert "never invent" in prompt

    def test_latin_drug_names_preserved_verbatim(self) -> None:
        quoted = "Metformin 500 mg يبدأ مع الوجبات"
        prompt = build_prompt("سؤال", [_row(0.9, quoted=quoted)])
        assert "Metformin 500 mg" in prompt
        assert "[quoted] Metformin 500 mg يبدأ مع الوجبات" in prompt

    def test_every_row_appears_as_quoted(self) -> None:
        prompt = build_prompt("سؤال", [_row(0.9, "أ"), _row(0.8, "ب")])
        assert "[quoted] أ" in prompt and "[quoted] ب" in prompt


class TestDeterministicLlm:
    def test_echoes_first_quoted_chunk(self) -> None:
        llm = DeterministicLlm()
        out = llm.complete(build_prompt("س", [_row(0.9, "الجري بالجرعة الكاملة")]))
        assert out.startswith("الجري بالجرعة الكاملة")
        assert len(out) <= 80

    def test_refuses_when_no_quoted(self) -> None:
        assert DeterministicLlm().complete("فقط تعليمات، لا سياق مقتبس هنا") == "REFUSAL"


class TestAnswerer:
    def test_refuses_below_gate_with_typed_outcome(self) -> None:
        out = Answerer(llm=DeterministicLlm()).answer("س", [_row(0.2)])
        assert out.refused is True
        assert out.answer == ""
        assert out.citations == []
        assert "below grounding threshold" in out.refuse_reason

    def test_refuses_on_empty_context(self) -> None:
        out = Answerer(llm=DeterministicLlm()).answer("س", [])
        assert out.refused is True
        assert out.refuse_reason == "no grounded context retrieved for this question"

    def test_refuses_on_empty_model_output(self) -> None:
        class _Empty:
            def complete(self, prompt, temperature=0.2) -> str:
                return ""

        out = Answerer(llm=_Empty()).answer("س", [_row(0.9)])
        assert out.refused is True
        assert "refused to answer" in out.refuse_reason

    def test_llm_unavailable_becomes_typed_refusal(self) -> None:
        class _Down:
            def complete(self, prompt, temperature=0.2) -> str:
                raise LlmUnavailableError("cannot reach Ollama")

        out = Answerer(llm=_Down()).answer("س", [_row(0.9)])
        assert out.refused is True
        assert out.answer == ""
        assert "generation model unavailable" in out.refuse_reason

    def test_grounded_answer_returns_cited_text(self) -> None:
        out = Answerer(llm=DeterministicLlm()).answer("س", [_row(0.9, "الجرعة القصوى 4 غرام")])
        assert out.refused is False
        assert out.answer  # echoed chunk text
        assert out.citations
        c = out.citations[0]
        assert c.doc_id == "ma::ar__train_csv::000005"
        assert c.collection == "medical_docs"
        assert c.section == "dispensing"
        assert c.score == pytest.approx(0.9)

    def test_refusal_prefix_handled(self) -> None:
        class _RefusalLlm:
            def complete(self, prompt, temperature=0.2) -> str:
                return "REFUSAL: لا معلومات"

        out = Answerer(llm=_RefusalLlm()).answer("س", [_row(0.9)])
        assert out.refused is True
        assert "refused to answer" in out.refuse_reason


class TestOllamaLlm:
    def test_down_ollama_raises_typed_error(self, monkeypatch) -> None:
        def _down(*args, **kwargs):
            raise httpx.ConnectError("refused", request=httpx.Request("POST", "http://x"))

        monkeypatch.setattr(httpx, "post", _down)
        with pytest.raises(LlmUnavailableError):
            OllamaLlm(host="http://localhost:11434", model="command-r7b-arabic").complete("p")

    def test_http_error_status_raises_typed_error(self, monkeypatch) -> None:
        class _R:
            status_code = 500
            text = "boom"

        monkeypatch.setattr(httpx, "post", lambda *a, **k: _R())
        with pytest.raises(LlmUnavailableError, match="Ollama error 500"):
            OllamaLlm().complete("p")

    def test_malformed_json_raises_typed_error(self, monkeypatch) -> None:
        class _RBad:
            status_code = 200

            def json(self):
                raise ValueError("no json")

        monkeypatch.setattr(httpx, "post", lambda *a, **k: _RBad())
        with pytest.raises(LlmUnavailableError, match="malformed"):
            OllamaLlm().complete("p")

    def test_missing_response_key_raises_typed_error(self, monkeypatch) -> None:
        class _RNoKey:
            status_code = 200

            def json(self):
                return {"nope": 1}

        monkeypatch.setattr(httpx, "post", lambda *a, **k: _RNoKey())
        with pytest.raises(LlmUnavailableError, match="malformed"):
            OllamaLlm().complete("p")

    def test_happy_path_returns_response(self, monkeypatch) -> None:
        class _ROk:
            status_code = 200

            def json(self):
                return {"response": "الجواب الطبي"}

        monkeypatch.setattr(httpx, "post", lambda *a, **k: _ROk())
        assert OllamaLlm().complete("p", temperature=0.2) == "الجواب الطبي"