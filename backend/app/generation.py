"""Grounded generation — THE tested seam (spec §1 rev / "test at the
middle-layer generation contract", milestone M3).

Contract (this is what every test asserts against, never internals):
  * ``POST /v1/.../query``-shaped: question -> answer + typed citations,
    where EVERY factual claim in the answer resolves to a real retrieved
    chunk (spec §5 / Citation contract: "answers from the LLM's own
    knowledge instead of the retrieved context" is the #1 pitfall).
  * Grounding gate: if the best retrieved chunk's score < min_score, the
    gate REFUSES with a typed refusal — never an invented answer.
  * Refusal is a first-class typed outcome, never an error response.

The LLM surface is injected (Ollama in production, deterministic stub in
tests) so the seam is testable without a model — the same discipline as an
injected embedder (§6).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from .models import Citation
from .retrieval import RetrievedChunk


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    citations: list[Citation]
    refused: bool = False
    refuse_reason: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.model_dump() for c in self.citations],
            "refused": self.refused,
            "refuse_reason": self.refuse_reason,
        }


class GroundingGate:
    """The trust gate's answer side (spec §4/§7): nothing is said that the
    retrieval layer could not back with a real chunk."""

    def __init__(self, min_score: float = 0.35) -> None:
        self._min_similarity = min_score

    def gate(self, rows: Sequence[RetrievedChunk]) -> bool:
        """True if the top hit clears the gate and there is any context at all.
        An empty store or a sub-threshold top hit -> False (refusal)."""
        if not rows:
            return False
        return rows[0].score >= self._min_similarity

    def reason(self, rows: Sequence[RetrievedChunk]) -> str:
        if not rows:
            return "no grounded context retrieved for this question"
        return f"best hit ({rows[0].score:.2f}) below grounding threshold {self._min_similarity:.2f}"


class LlmSurface(Protocol):
    """The generation surface. Production: an Ollama client (§6 generation
    candidate: Command-R7B-Arabic / Falcon-H1-Arabic both via Ollama).
    Tests: a deterministic stub. Both satisfy: ``complete(prompt) -> str``."""

    def complete(self, prompt: str, temperature: float = 0.2) -> str: ...


class OllamaLlm:
    """Ollama generation client. ``host``/``model`` come from config/env — the
    backend must never hard-code either. Lazy connection on first call so a
    CPU-only box can import the module without touching Ollama."""

    def __init__(self, host: str = "http://localhost:11434", model: str = "command-r7b") -> None:
        self._host = host.rstrip("/")
        self._model = model

    def complete(self, prompt: str, temperature: float = 0.2) -> str:
        import httpx

        resp = httpx.post(
            f"{self._host}/api/generate",
            json={"model": self._model, "prompt": prompt, "stream": False, "temperature": temperature},
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["response"]


class DeterministicLlm:
    """Tests ONLY: echoes the first retrieved citation's quoted text so the
    grounding contract can be asserted WITHOUT a model. NEVER production."""

    def complete(self, prompt: str, temperature: float = 0.2) -> str:
        # Prompt carries a machine-readable context block; find the first
        # [quoted] marker the generator always injects (fragile-by-design:
        # it only exists because the seam asserts the model picked ONE chunk).
        import re

        m = re.search(r"\[quoted\]\s*(.+)", prompt, re.DOTALL)
        if m:
            return m.group(1).strip()[:80]
        return "REFUSAL"


def build_prompt(
    question: str,
    rows: Sequence[RetrievedChunk],
    language: str = "ar",
) -> str:
    """The grounded-generation prompt (spec §6.4 seam contract).

    Context block is a lossless, typed JSON list (doc_id, collection, section,
    quoted, chunk_id, score) the LLM may ONLY cite from — not from its own
    knowledge. Instruction flips with the primary language (Arabic-primary,
    Arabic output; Latin drug names preserved verbatim)."""
    ctx = [
        {
            "doc_id": r.doc_id,
            "collection": r.collection,
            "section": r.section,
            "chunk_id": r.chunk_id,
            "quoted": r.quoted,
            "score": round(r.score, 3),
        }
        for r in rows
    ]
    if language == "ar":
        system = (
            "أنت مساعد طبي ردّك يستند فقط إلى المستندات أدناه. أجب بالعربية، "
            "احفظ أسماء الأدوية اللاتينية كما هي، واذكر المصدر بعد كل معلومة "
            "بصيغة [الجملة المقتبسة] إذا لم تجد المعلومة في المستندات فقل "
            "'لا أملك معلومات كافية' ولا تخترع."
        )
    else:
        system = (
            "You are a medical assistant. Answer ONLY from the documents below. "
            "Keep Latin drug names verbatim. Cite the quoted source after each "
            "claim. If the information is not in the documents, say so — never "
            "invent."
        )
    block = [f"[quoted] {r.quoted}" for r in rows]
    return f"{system}\n\nCONTEXT:\n{json.dumps(ctx, ensure_ascii=False)}\n\n{chr(10).join(block)}\n\nQUESTION: {question}\nANSWER:"


class Answerer:
    """Generation seam implementation: gate -> prompt -> LLM -> typed answer
    with citations. Everything a request handler needs; nothing more."""

    def __init__(self, llm: LlmSurface, gate: GroundingGate | None = None) -> None:
        self._llm = llm
        self._gate = gate or GroundingGate()

    def answer(
        self,
        question: str,
        rows: Sequence[RetrievedChunk],
        language: str = "ar",
        temperature: float = 0.2,
    ) -> GeneratedAnswer:
        if not self._gate.gate(rows):
            return GeneratedAnswer(
                answer="",
                citations=[],
                refused=True,
                refuse_reason=self._gate.reason(rows),
            )
        prompt = build_prompt(question, rows, language=language)
        text = self._llm.complete(prompt, temperature=temperature).strip()
        if not text or text.upper().startswith("REFUSAL"):
            return GeneratedAnswer(
                answer="",
                citations=[],
                refused=True,
                refuse_reason="model refused to answer from the provided grounding",
            )
        cited = [c for c in _citations_from_rows(rows) if c.quoted]
        return GeneratedAnswer(answer=text, citations=cited)


def _citations_from_rows(rows: Sequence[RetrievedChunk]) -> list[Citation]:
    """Citation shape = the retrieved chunk's citation fields (spec §3 doc-id
    registry). The answer's evidence list is exactly the retrieved set the
    gate allowed; the frontend renders these as clickable sources."""
    return [
        Citation(
            doc_id=r.doc_id,
            collection=r.collection,
            section=r.section,
            quoted=r.quoted,
            score=round(r.score, 3),
        )
        for r in rows
    ]
