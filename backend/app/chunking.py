"""Document chunking for the medical corpus (§6 retrieval backbone, milestone).

The grad-release corpus is medical *documents* (guidelines, drug sheets, clinic
notes) — not per-patient dialogue turns — so chunks are heading-anchored
paragraph units instead of turns. Each chunk keeps its section heading as
context and stays under a fixed char budget, so every chunk unit is
independently retrievable AND citable (§3 doc-id registry + Citation contract).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_MAX_CHARS = 800
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_SECTION_FALLBACK = "general"


@dataclass(frozen=True)
class DocumentChunk:
    doc_id: str
    collection: str
    seq: int
    section: str
    text: str
    metadata: dict

    @property
    def chunk_id(self) -> str:
        """Stable, citable chunk id: doc-id + seq (spec §3 doc-id registry)."""
        return f"{self.doc_id}/{self.seq:03d}"

    def citation(self) -> dict:
        """The citation shape a grounded answer must resolve to (models.Citation)."""
        return {
            "doc_id": self.doc_id,
            "collection": self.collection,
            "section": self.section,
            "quoted": self.text,
        }


def chunk_document(text: str, doc_id: str, collection: str = "medical_docs") -> list[DocumentChunk]:
    """Split a normalized medical document into heading-anchored chunks.

    Paragraph-level chunking (never line-level), carrying the current section
    heading onto every chunk. This is the concept-build instantiation of the
    spec's "chunk = semantic unit, not a token window" (§6).
    """
    chunks: list[DocumentChunk] = []
    section = _SECTION_FALLBACK
    pending: list[str] = []
    seq = 0

    for raw in text.splitlines():
        line = raw.rstrip()
        heading = _HEADING_RE.match(line.strip())
        if heading:
            paragraph = " ".join(p.strip() for p in pending if p.strip()).strip()
            if paragraph:
                for piece in _split_long(paragraph):
                    chunks.append(
                        DocumentChunk(doc_id, collection, seq, section, piece,
                                      {"doc_id": doc_id, "section": section, "collection": collection})
                    )
                    seq += 1
            pending = []
            section = heading.group(2).strip()
            continue
        pending.append(line)

    paragraph = " ".join(p.strip() for p in pending if p.strip()).strip()
    if paragraph:
        for piece in _split_long(paragraph):
            chunks.append(
                DocumentChunk(doc_id, collection, seq, section, piece,
                              {"doc_id": doc_id, "section": section, "collection": collection})
            )
            seq += 1
    return chunks


def _split_long(paragraph: str) -> list[str]:
    """Split an over-budget paragraph at token boundaries, not mid-word."""
    if len(paragraph) <= _MAX_CHARS:
        return [paragraph]
    pieces: list[str] = []
    buffer = ""
    for token in paragraph.split(" "):
        token = _hard_split(token)
        if len(buffer) + len(token) + 1 > _MAX_CHARS and buffer:
            pieces.append(buffer.strip())
            buffer = token
        else:
            buffer = f"{buffer} {token}".strip() if buffer else token
    if buffer.strip():
        pieces.append(buffer.strip())
    return pieces


def _hard_split(word: str) -> str:
    """Guard for a single unbroken token over budget (long Latin drug names):
    split it into ≤_MAX_CHARS pieces; each piece becomes its own chunk token."""
    if len(word) <= _MAX_CHARS:
        return word
    return " ".join(word[i : i + _MAX_CHARS] for i in range(0, len(word), _MAX_CHARS))
