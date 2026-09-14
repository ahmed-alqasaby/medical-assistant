"""retrieval.py — VectorStore lifecycle + search contract over a persisted
Chroma store (the query side of the notebook's index side). Deterministic
embedder only; never a model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.app.chunking import DocumentChunk
from backend.app.retrieval import DeterministicEmbedder, VectorStore


def _chunk(doc_id: str, text: str, section: str = "general", lang: str = "ar",
           source: str = "ar/train.csv", collection: str = "medical_docs") -> DocumentChunk:
    return DocumentChunk(
        doc_id=doc_id, collection=collection, seq=0, section=section, text=text,
        metadata={
            "doc_id": doc_id, "collection": collection, "section": section, "seq": 0,
            "chunk_id": f"{doc_id}/000000", "lang": lang, "source": source,
        },
    )


_GROUNDED = [
    _chunk(
        "ma::ar__train_csv::000000",
        "دواء الأسبيرين (ASA) 100 ملغ يُؤخذ قرصاً واحداً يومياً بعد الطعام.",
        section="dispensing", lang="ar", source="ar/train.csv",
    ),
    _chunk(
        "ma::ark__train_csv::000001",
        "باراسيتامول: الحد الأقصى 4 غرام يومياً للبالغين.",
        section="dispensing", lang="en", source="ar/train.csv",
    ),
]


class TestDeterministicEmbedder:
    def test_returns_unit_norm_rows(self) -> None:
        e = DeterministicEmbedder(dim=64, n_gram=3)
        out = e.embed(["مرحبا"])
        norms = np.linalg.norm(out, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_is_normalization_invariant(self) -> None:
        e = DeterministicEmbedder(dim=64, n_gram=3)
        a = e.embed(["بَاراسيتا"])
        b = e.embed(["باراسيتا"])
        assert np.allclose(a, b)

    def test_deterministic_across_calls(self) -> None:
        e = DeterministicEmbedder(dim=64, n_gram=3)
        assert np.array_equal(e.embed(["نص"]), e.embed(["نص"]))


class TestVectorStoreLifecycle:
    def test_persist_dir_is_chroma_export_dir(self, tmp_path: Path) -> None:
        s = VectorStore(str(tmp_path / "chroma"), embedder=DeterministicEmbedder(dim=64))
        assert s.collection == "medical_docs"
        assert s.persist() == str(tmp_path / "chroma")

    def test_load_is_idempotent(self, tmp_path: Path) -> None:
        s = VectorStore(str(tmp_path / "chroma"), embedder=DeterministicEmbedder(dim=64))
        s.load()
        s.load()
        assert s.is_loaded() is True
        s.close()
        assert s.is_loaded() is False

    def test_count_after_add(self, tmp_path: Path) -> None:
        s = VectorStore(str(tmp_path / "chroma"), embedder=DeterministicEmbedder(dim=64))
        s.load()
        s.add_documents(_GROUNDED)
        assert s.count() == 2

    def test_add_persists_to_disk_for_fresh_handle(self, tmp_path: Path) -> None:
        s = VectorStore(str(tmp_path / "chroma"), embedder=DeterministicEmbedder(dim=64))
        s.load()
        s.add_documents(_GROUNDED)
        # notebook-style: a FRESH PersistentClient reopens the stored data
        s2 = VectorStore(str(tmp_path / "chroma"), embedder=DeterministicEmbedder(dim=64))
        s2.load()
        assert s2.count() == 2

    def test_reset_drops_collection_for_clean_rebuild(self, tmp_path: Path) -> None:
        s = VectorStore(str(tmp_path / "chroma"), embedder=DeterministicEmbedder(dim=64))
        s.load()
        s.add_documents(_GROUNDED)
        assert s.count() == 2
        s.reset()
        # a rebuild after reset starts from an empty store (no dim lock)
        s.add_documents(_GROUNDED)
        assert s.count() == 2

    def test_reset_missing_collection_is_safe(self, tmp_path: Path) -> None:
        s = VectorStore(str(tmp_path / "chroma"), embedder=DeterministicEmbedder(dim=64))
        s.load()
        s.reset()  # no collection yet -> no-op, not an error
        assert s.count() == 0


class TestSearchContract:
    def test_empty_store_returns_no_rows(self, tmp_path: Path) -> None:
        s = VectorStore(str(tmp_path / "chroma"), embedder=DeterministicEmbedder(dim=64))
        s.load()
        assert s.search("أي سؤال") == []

    def test_search_returns_typed_citable_rows(self, tmp_path: Path) -> None:
        s = VectorStore(str(tmp_path / "chroma"), embedder=DeterministicEmbedder(dim=64))
        s.load()
        s.add_documents(_GROUNDED)
        rows = s.search("الأسبيرين", top_k=2)
        assert rows, "search must find grounded rows"
        row = rows[0]
        assert row.doc_id.startswith("ma::")
        assert row.collection == "medical_docs"
        assert row.chunk_id.endswith("/000000")
        assert 0.0 <= row.score <= 1.0
        assert row.quoted  # the document text is the citable block

    def test_metadata_keys_survive_roundtrip(self, tmp_path: Path) -> None:
        s = VectorStore(str(tmp_path / "chroma"), embedder=DeterministicEmbedder(dim=64))
        s.load()
        s.add_documents(_GROUNDED)
        rows = s.search("باراسيتامول", top_k=1)
        assert rows[0].section in ("dispensing", "general")
        assert rows[0].collection == "medical_docs"

    def test_top_k_respected_and_sorted_desc(self, tmp_path: Path) -> None:
        s = VectorStore(str(tmp_path / "chroma"), embedder=DeterministicEmbedder(dim=64))
        s.load()
        s.add_documents(_GROUNDED)
        rows = s.search("دواء", top_k=1)
        assert len(rows) == 1
        scores = [r.score for r in rows]
        assert scores == sorted(scores, reverse=True)

    def test_similarity_clamp(self) -> None:
        assert VectorStore._to_similarity(0.3) == pytest.approx(0.7)
        assert VectorStore._to_similarity(-0.5) == 1.0  # negative distance -> max
        assert VectorStore._to_similarity(2.0) == 0.0

    def test_collection_uses_cosine_space(self, tmp_path: Path) -> None:
        s = VectorStore(str(tmp_path / "chroma"), embedder=DeterministicEmbedder(dim=64))
        coll = s._coll()
        assert coll.metadata.get("hnsw:space") == "cosine"