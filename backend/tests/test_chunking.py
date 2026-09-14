"""chunking.py — heading-anchored document chunking (§6) + the chunk-id contract
aligned to the notebook's ``doc_id/{seq:06d}`` scheme."""

from __future__ import annotations

import pytest

from backend.app.chunking import DocumentChunk, _MAX_CHARS, chunk_document


class TestDocumentChunk:
    def test_chunk_id_matches_notebook_zero_padding(self) -> None:
        c = DocumentChunk(
            doc_id="ma::ar__train_csv::000042", collection="medical_docs", seq=42,
            section="general", text="نص", metadata={},
        )
        assert c.chunk_id == "ma::ar__train_csv::000042/000042"

    def test_citation_shape_is_models_citation_contract(self) -> None:
        c = DocumentChunk(
            doc_id="rx-1", collection="rx_photos", seq=0, section="dispensing",
            text="النص", metadata={"doc_id": "rx-1"},
        )
        assert c.citation() == {
            "doc_id": "rx-1",
            "collection": "rx_photos",
            "section": "dispensing",
            "quoted": "النص",
        }


class TestChunkDocument:
    def test_small_document_single_chunk(self) -> None:
        chunks = chunk_document("فقرات طبية بدون عناوين.", doc_id="d1")
        assert len(chunks) == 1
        assert chunks[0].section == "general"

    def test_heading_anchors_sticky_sections(self) -> None:
        text = (
            "# الجرعات\n"
            "الجرعة الابتدائية 500 ملغ مرة يوميا.\n"
            "الجرعة القصوى 4 غرام يوميا.\n"
            "# الآثار الجانبية\n"
            "صداع ودوخة خفيفة.\n"
        )
        chunks = chunk_document(text, doc_id="d1")
        # one chunk per section (paragraph-level: lines under a heading join up)
        assert len(chunks) == 2
        assert chunks[0].section == "الجرعات"
        assert chunks[1].section == "الآثار الجانبية"
        assert "الجرعة الابتدائية" in chunks[0].text and "4 غرام" in chunks[0].text

    def test_seq_increments_within_document(self) -> None:
        chunks = chunk_document("# ع\nف1\n\n# ف\nف2", doc_id="d1")
        assert [c.seq for c in chunks] == [0, 1]

    def test_metadata_carries_provenance(self) -> None:
        chunks = chunk_document("# ع\nف", doc_id="d1", collection="medical_docs")
        assert chunks[0].metadata["doc_id"] == "d1"
        assert chunks[0].metadata["collection"] == "medical_docs"
        assert "section" in chunks[0].metadata

    def test_over_budget_paragraph_split_at_word_boundaries(self) -> None:
        words = "كلمة " * 500  # > _MAX_CHARS, all tokens within budget
        chunks = chunk_document(words.strip(), doc_id="d1")
        assert len(chunks) > 1
        assert all(len(c.text) <= _MAX_CHARS for c in chunks)

    def test_hard_split_unbreakable_long_token(self) -> None:
        long_drug = "LONG-LATIN-DRUG-NAME-" * 100  # > budget, no spaces
        chunks = chunk_document(long_drug, doc_id="d1")
        assert all(len(c.text) <= _MAX_CHARS for c in chunks)

    def test_empty_text_yields_no_chunks(self) -> None:
        assert chunk_document("", doc_id="d1") == []
        assert chunk_document("\n\n  \n", doc_id="d1") == []