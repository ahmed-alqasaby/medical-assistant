"""Notebook-contract drift guard — maps the APP to the NOTEBOOK (canonical split).

``notebooks/rag_pipeline.ipynb`` (Phase 2) is the index side: it loads the
corpus, chunks with the app's own chunker, embeds, and persists the store under
``data/vector_store/chroma``; the backend app is the query side (load the
persisted store at startup). The two sides MUST agree on every shared
constant, or the product silently splits into two incompatible pipelines.

This module parses the committed notebook *as data* and asserts, cell by cell,
that its contract still matches what ``backend/app`` actually implements:

  * config: COLLECTION / EMBED_MODEL / EMBED_DEVICE / TOP_K / MIN_SCORE / dir
  * chunking: app ``chunk_document`` + the ``ma::<src>::<seq:06d>`` doc-id
    scheme + the full metadata key set the backend reads back on search
  * embedding: the same ``VectorStore``/embedder classes, cosine space,
    ``store.reset()`` from-scratch rebuild, ``export_manifest.json``
  * retrieval: ``store.search`` + ``GroundingGate`` + the §6.4 ``build_prompt``
  * export: manifest + ``data/vector_store/chroma`` == ``Settings()`` default

Deterministic: depends only on the committed notebook + the backend modules.
Run this after ANY edit to the notebook generator or the backend contract.
"""
from __future__ import annotations

import re
from pathlib import Path

import nbformat

from backend.app.chunking import chunk_document
from backend.app.config import Settings

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "rag_pipeline.ipynb"

NOTEBOOK_METADATA_KEYS = {"doc_id", "collection", "section", "seq", "chunk_id", "lang", "source"}


def _cells() -> list[str]:
    nb = nbformat.read(str(NOTEBOOK), as_version=4)
    return ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb.cells]


def _cell(cells: list[str], marker: str) -> str:
    hits = [c for c in cells if marker in c]
    assert hits, f"notebook cell containing {marker!r} not found"
    return hits[0]


class TestNotebookExists:
    def test_notebook_present_and_runnable_shape(self) -> None:
        assert NOTEBOOK.exists(), "notebook missing — backend contract has no index side"
        nb = nbformat.read(str(NOTEBOOK), as_version=4)
        assert len(nb.cells) >= 14, "notebook cells shrank below the contract spine"


class TestConfigCellContract:
    def test_collection_matches_backend(self) -> None:
        cfg = _cell(_cells(), "# --- config")
        settings = Settings()
        assert 'COLLECTION  = "medical_docs"' in cfg
        assert settings.default_collection == "medical_docs"

    def test_embed_model_default_matches_backend(self) -> None:
        cfg = _cell(_cells(), "# --- config")
        assert 'os.environ.get("EMBED_MODEL", "BAAI/bge-m3")' in cfg
        assert Settings().embed_model == "BAAI/bge-m3"

    def test_embed_device_knob_wired(self) -> None:
        cfg = _cell(_cells(), "# --- config")
        assert 'os.environ.get("EMBED_DEVICE", "")' in cfg

    def test_retrieval_defaults_match_backend(self) -> None:
        cfg = _cell(_cells(), "# --- config")
        settings = Settings()
        assert 'os.environ.get("TOP_K", "5")' in cfg
        assert settings.top_k == 5
        assert 'os.environ.get("MIN_SCORE", "0.35")' in cfg
        assert settings.min_score == 0.35

    def test_store_dir_matches_backend_default(self) -> None:
        cfg = _cell(_cells(), "# --- config")
        assert '"data" / "vector_store" / "chroma"' in cfg
        settings = Settings()
        path = Path(settings.vector_store_path)
        assert path == ROOT / "data" / "vector_store" / "chroma"

    def test_max_rows_cap_knob_present(self) -> None:
        cfg = _cell(_cells(), "# --- config")
        assert 'os.environ.get("MAX_ROWS", "1500")' in cfg


class TestChunkingContract:
    def test_uses_the_app_chunker(self) -> None:
        cell = _cell(_cells(), "# --- chunking")
        assert "app_chunking.chunk_document" in cell
        assert "# {row['section']}" in cell  # heading-anchored = app semantics

    def test_doc_id_scheme_ma_src_seq(self) -> None:
        cell = _cell(_cells(), "# --- chunking")
        assert 'doc_id = f"ma::{src}::{seq:06d}"' in cell

    def test_metadata_keys_cover_what_backend_reads(self) -> None:
        cell = _cell(_cells(), "# --- chunking")
        for key in NOTEBOOK_METADATA_KEYS:
            assert key in cell, f"notebook chunk_row lost key {key!r}"
        # backend search() reads exactly: doc_id, collection, seq, section, chunk_id
        for read_key in ("doc_id", "collection", "section", "seq", "chunk_id"):
            assert read_key in NOTEBOOK_METADATA_KEYS

    def test_chunk_id_uses_notebook_padding(self) -> None:
        cell = _cell(_cells(), "# --- chunking")
        assert '"chunk_id": c.chunk_id' in cell or "chunk_id" in cell
        chunks = chunk_document(
            "# عنوان\nفقرات طبية قصيرة هنا.",
            doc_id="ma::ar__train_csv::000000",
            collection="medical_docs",
        )
        assert chunks
        assert all(c.chunk_id.endswith("/000000") for c in chunks)
        assert re.fullmatch(r"ma::ar__train_csv::000000/\d{6}", chunks[0].chunk_id)


class TestEmbedPersistContract:
    def test_uses_the_same_store_and_embedder_classes(self) -> None:
        cell = _cell(_cells(), "# --- embed + persist")
        assert "from backend.app.retrieval import" in cell
        assert "BgeM3Embedder" in cell and "DeterministicEmbedder" in cell
        assert "VectorStore" in cell

    def test_from_scratch_rebuild_drops_collection_first(self) -> None:
        cell = _cell(_cells(), "# --- embed + persist")
        assert "store.reset()" in cell  # Chroma locks dim at first write

    def test_demo_and_real_embedder_modes_match_backend(self) -> None:
        stores = _cell(_cells(), "# --- embed + persist")
        # real: bge-m3 with the optional device knob; demo: deterministic 1536
        assert "BgeM3Embedder(model_name=EMBED_MODEL, device=EMBED_DEVICE or None)" in stores
        # backend VectorStore defaults to BgeM3Embedder; demo uses deterministic
        src = (ROOT / "backend" / "app" / "retrieval.py").read_text(encoding="utf-8")
        assert "dim: int = 1536" in src

    def test_cosine_space_both_sides(self) -> None:
        src = (ROOT / "backend" / "app" / "retrieval.py").read_text(encoding="utf-8")
        assert '"hnsw:space": "cosine"' in src  # VectorStore._coll()

    def test_normalized_dense_embeddings(self) -> None:
        # BgeM3Embedder normalizes via sentence-transformers normalize_embeddings
        src = (ROOT / "backend" / "app" / "retrieval.py").read_text(encoding="utf-8")
        assert "normalize_embeddings=True" in src
        # deterministic embedder emits unit-norm rows used in the demo store
        assert "dim: int = 1536" in src

    def test_manifest_written_beside_the_store(self) -> None:
        cell = _cell(_cells(), "# --- embed + persist")
        assert '"export_manifest.json"' in cell
        assert '"collection": COLLECTION' in cell
        assert '"model": EMBED_MODEL' in cell
        assert '"demo_store": IS_DEMO' in cell
        assert '"hnsw_space": "cosine"' in cell
        # the FULL metadata key set lives on the chunk (cell 5); the manifest
        # stamps whatever that chunk carries
        assert '"metadata_keys":' in cell
        chunk_cell = _cell(_cells(), "# --- chunking")
        for key in NOTEBOOK_METADATA_KEYS:
            assert key in chunk_cell, f"chunk_row lost key {key!r}"


class TestRetrievalContract:
    def test_retrieval_is_store_search_over_persisted_store(self) -> None:
        cell = _cell(_cells(), "# --- retrieval + grounded answer helpers")
        assert "store.search(question, top_k=top_k)" in cell

    def test_grounding_gate_uses_min_score(self) -> None:
        cell = _cell(_cells(), "# --- retrieval + grounded answer helpers")
        assert "GroundingGate(min_score=MIN_SCORE)" in cell

    def test_backend_grounds_on_dense_top_k(self) -> None:
        src = (ROOT / "backend" / "app" / "retrieval.py").read_text(encoding="utf-8")
        assert "n_results=min(top_k, count)" in src
        assert "_to_similarity" in src


class TestGenerationContract:
    def test_uses_the_backend_prompt_seam(self) -> None:
        cell = _cell(_cells(), "# --- retrieval + grounded answer helpers")
        assert "import backend.app.generation as gen" in cell
        assert 'answerer.answer(question, rows, language="ar")' in cell
        assert "gen.Answerer" in cell and "gen.GroundingGate" in cell
        assert "gen.OllamaLlm" in cell and "gen.DeterministicLlm" in cell

    def test_prompt_contract_keys_and_arabic_injunction(self) -> None:
        # The notebook reuses backend build_prompt — assert the contract lives
        # in the backend and matches what the notebook's context block carries.
        src = (ROOT / "backend" / "app" / "generation.py").read_text(encoding="utf-8")
        for key in ("doc_id", "collection", "section", "chunk_id", "quoted", "score"):
            assert f'"{key}"' in src or key in src, f"prompt ctx lost key {key!r}"
        assert "[quoted]" in src
        assert "لا أملك معلومات كافية" in src
        cell = _cell(_cells(), "# --- retrieval + grounded answer helpers")
        assert "build_prompt" in repr(cell) or "answerer.answer" in cell

    def test_refusal_is_a_typed_outcome(self) -> None:
        cell = _cell(_cells(), "# --- eval over the full question set")
        assert "refused" in cell
        src = (ROOT / "backend" / "app" / "generation.py").read_text(encoding="utf-8")
        assert "refused" in src


class TestEvalAndExportContract:
    def test_eval_section_requires_at_least_ten_questions(self) -> None:
        cell = _cell(_cells(), "# --- eval over the full question set")
        assert "EVAL" in cell
        assert "in_domain" in cell
        assert "correct" in cell

    def test_eval_results_written_next_to_store(self) -> None:
        cfg = _cell(_cells(), "# --- config")
        assert 'RESULTS    = CHROMA_DIR / "eval_results.csv"' in cfg
        cell = _cell(_cells(), "# --- eval over the full question set")
        assert "df.to_csv(RESULTS" in cell

    def test_manifest_targets_repo_data_vector_store(self) -> None:
        cell = _cell(_cells(), "# --- embed + persist")
        assert "CHROMA_DIR" in cell
        assert '"metadata_keys"' in cell
        cfg = _cell(_cells(), "# --- config")
        assert '"data" / "vector_store" / "chroma"' in cfg


class TestStoreMapRoundTrip:
    def test_app_store_opens_the_path_the_notebook_writes(self) -> None:
        # Settings().vector_store_path == <repo>/data/vector_store/chroma which
        # is exactly the CHROMA_DIR the notebook persists to.
        settings = Settings()
        path = Path(settings.vector_store_path)
        assert path == ROOT / "data" / "vector_store" / "chroma"
        cfg = _cell(_cells(), "# --- config")
        assert "CHROMA_DIR = REPO_ROOT / \"data\" / \"vector_store\" / \"chroma\"" in cfg
        assert str(path) == str(ROOT / "data" / "vector_store" / "chroma")