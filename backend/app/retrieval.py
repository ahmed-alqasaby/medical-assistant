"""Typed vector retrieval over a persisted Chroma store (§6 milestone spine).

The heavy work — embedding the whole medical corpus with bge-m3 and building
the store — runs in the Kaggle notebook for the weak-GPU constraint (§1 rev /
graduation milestone). The backend only needs to:
  1. load the persisted store at startup (§7 "load once at startup, never per request")
  2. embed the single incoming query (bge-m3; ~1 s on CPU)
  3. retrieve top-k typed rows with citation fields + a cosine score the
     grounding gate can refuse on.

Embedding is injected so tests use a deterministic fixture embedder and never
touch a model. The product seam (spec: test at the middle-layer generation
contract) is generation.py — see its module docstring for the seam definition.
"""

from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import chromadb

from .chunking import DocumentChunk
from .normalization import normalize_query


@dataclass(frozen=True)
class RetrievedChunk:
    """A typed, citable retrieval hit (spec §6.5 / Citation contract)."""

    doc_id: str
    collection: str
    seq: int
    section: str
    chunk_id: str
    quoted: str
    score: float  # ~0..1 similarity; grounding gate refuses below threshold

    def citation(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "collection": self.collection,
            "section": self.section,
            "quoted": self.quoted,
        }


def _ngrams(text: str, n: int) -> list[str]:
    norm = normalize_query(text)
    if not norm:
        return []
    return [norm[i : i + n] for i in range(len(norm) - n + 1)]


def _blake(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


class Embedder:
    """Embedding surface. Implementations:
    - ``BgeM3Embedder`` — production (§6 locked): bge-m3 via sentence-transformers.
    - ``DeterministicEmbedder`` — tests ONLY; deterministic, no model download.
    Both embed a list of strings -> float32 matrix with normalized rows.
    """

    def __init__(self, dim: int | None = None) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise NotImplementedError("embedder must declare its dimension")
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class DeterministicEmbedder(Embedder):
    """Hash bag of ngrams for tests ONLY: deterministic, cheap, stable across
    runs, and normalization-invariant because it operates on ``normalize_query``
    output — the same normalization contract as production. NEVER in prod."""

    def __init__(self, dim: int = 1536, n_gram: int = 3) -> None:
        super().__init__(dim)
        self._n_gram = n_gram
        # fixed random rotation decorrelates hash collisions while staying
        # deterministic (same seed -> same matrix, every run)
        rng = np.random.default_rng(7)
        self._rot = np.asarray(rng.normal(size=(dim, dim)), dtype="float32")

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            vec = np.zeros(self.dim, dtype="float32")
            for n in range(1, self._n_gram + 1):
                for gram in _ngrams(text, n):
                    vec[_blake(gram) % self.dim] += 1.0
            vec = (vec @ self._rot)
            denom = float(np.linalg.norm(vec)) or 1.0
            rows.append(vec / denom)
        return np.asarray(rows, dtype="float32")


class BgeM3Embedder(Embedder):
    """bge-m3 via sentence-transformers (§6 locked). Loads LAZILY on first
    embed so a CPU-only box can import this module without pulling the model.
    This exact class is the one canonical embedder in production — the SAME
    model in the Kaggle notebook (index side) and in the backend (query side);
    never two embedders for one product."""

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        super().__init__(dim=None)
        self._model_name = model_name
        self._model = None

    def _lazy(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        model = self._lazy()
        out = model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(out, dtype="float32")


class VectorStore:
    """Typed-collections store over a persisted Chroma dir.

    The SAME class runs on the build side (Kaggle notebook: embed whole corpus
    + persist) and the query side (backend: load persisted store + embed ONE
    query). Only the embedder instance differs (bge-m3 both sides in prod)."""

    def __init__(
        self,
        persist_dir: str,
        embedder: Embedder | None = None,
        collection: str = "medical_docs",
    ) -> None:
        self._dir = persist_dir
        self._embedder = embedder or BgeM3Embedder()
        self._collection = collection
        self._loaded = False
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._loaded = bool(persist_dir and os.path.isdir(persist_dir))

    # -- lifecycle seam (query side, §7 load-once) ---------------------------

    def load(self) -> None:
        """Load the persisted store ONCE, at startup — the spec's "load at
        startup, never per request" (§7). Chroma's PersistentClient reopens the
        on-disk dir lazily, so "loaded" is the flag that says startup ran;
        index data comes back on the first query. Idempotent: calling twice is
        safe."""

    def close(self) -> None:
        """Shutdown seam. PersistentClient persists to disk on its own; close
        just clears the loaded flag and lets the client GC. Never called per
        request."""

    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def collection(self) -> str:
        return self._collection

    def _coll(self) -> Any:
        return self._client.get_or_create_collection(
            name=self._collection,
            metadata={"hnsw:space": "cosine"},  # cosine distance for normalized bge-m3
        )

    # -- build side (Kaggle notebook) ----------------------------------------

    def add_documents(self, chunks: Sequence[DocumentChunk]) -> None:
        coll = self._coll()
        ids = [c.doc_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [dict(c.metadata) for c in chunks]
        embs = self._embedder.embed(docs)
        coll.upsert(
            ids=ids,
            embeddings=[e.tolist() for e in embs],
            documents=docs,
            metadatas=metas,
        )

    def persist(self) -> str:
        """Chromadata lives on disk via PersistentClient; return the export
        dir so the notebook documents the download contract (§7)."""
        return self._dir

    # -- query side (backend) -------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        count = self.count()
        if count == 0:
            return []
        coll = self._coll()
        q = self._embedder.embed([query])[0].tolist()
        res = coll.query(
            query_embeddings=[q],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        rows: list[RetrievedChunk] = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            rows.append(
                RetrievedChunk(
                    doc_id=meta.get("doc_id", ""),
                    collection=meta.get("collection", self._collection),
                    seq=int(meta.get("seq", 0)),
                    section=meta.get("section", ""),
                    chunk_id=str(meta.get("chunk_id", "")),
                    quoted=doc,
                    score=self._to_similarity(dist),
                )
            )
        return rows

    def count(self) -> int:
        try:
            return self._coll().count()
        except Exception:
            return 0

    @staticmethod
    def _to_similarity(distance: float) -> float:
        """Chroma cosine distance (1 - cos) -> similarity; clamp to [0, 1]."""
        return max(0.0, min(1.0, 1.0 - float(distance)))
