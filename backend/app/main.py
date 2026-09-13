"""The graded seam: FastAPI surface of the medical assistant (§7 endpoints /
["POST /query returns a grounded answer with typed citations; empty/"
"whitespace question -> 422"], milestone M4 in the notebook contract).

Deliberately THIN: it owns exactly the HTTP seam and nothing else.

  * startup (lifespan) loads the persisted VectorStore ONCE — the spec's
    "load at startup, never per request" rule (§7 load-once). ``main.py`` is
    the ONLY place the store is loaded for production.
  * ``POST /query`` validates via the pydantic query contract (empty or
    whitespace-only question -> 422, never a hallucinated reply), runs
    retrieval, and hands a typed Question to the middle-layer generation seam.
  * Refusal is a first-class typed outcome (``refuse=True``), NOT a 4xx.

The embedder + LLM are injected when built by test fixtures; production
path builds them from settings (lazy bge-m3 + lazy Ollama).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings
from .generation import Answerer, GroundingGate, OllamaLlm
from .models import QueryRequest, QueryResponse
from .normalization import normalize_query
from .retrieval import BgeM3Embedder, RetrievedChunk, VectorStore
from .config import settings  # noqa: F401  (re-exported so `uvicorn` sees it)


def build_app(
    *,
    store: VectorStore | None = None,
    llm: Any | None = None,
    min_score: float = 0.35,
    top_k: int = 5,
    settings: Settings | None = None,
) -> FastAPI:
    """Seam factory. Production: ``store=None`` -> real store from settings;
    tests: an injected DeterministicEmbedder store + stub LLM. This is the
    single place the store is created/loaded in the running app."""

    cfg = settings or Settings()
    if store is None:
        # Production: bge-m3 (the Kaggle notebook index side). Demo mode
        # (EMBED_MODEL=TEST / "") swaps in the deterministic embedder so a
        # CPU-only box can run the full stack without downloading any model.
        if cfg.is_demo_embedder:
            from .retrieval import DeterministicEmbedder

            embedder: Any = DeterministicEmbedder()
        else:
            embedder = BgeM3Embedder()
        store = VectorStore(persist_dir=str(cfg.vector_store_path), embedder=embedder)

    if llm is None:
        if cfg.is_demo_embedder:
            # Demo mode: extractive grounded answer (top cited chunk), the
            # same deterministic fallback the notebook uses — no Ollama needed.
            from .generation import DeterministicLlm

            llm = DeterministicLlm()
        else:
            llm = OllamaLlm(host=cfg.ollama_url, model=cfg.ollama_model)
    answerer = Answerer(llm=llm, gate=GroundingGate(min_score=min_score))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store.load()  # load ONCE at startup; never per request (§7)
        yield
        store.close()

    app = FastAPI(title="medical-assistant", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.store = store
    app.state.answerer = answerer
    app.state.top_k = top_k

    @app.get("/health")
    def health() -> dict[str, Any]:
        try:
            count = store.count()
        except Exception:
            count = None
        return {
            "status": "ok" if count is not None else "degraded",
            "store": {"loaded": store.is_loaded(), "chunks": count, "collection": store.collection},
        }

    @app.post("/query", response_model=QueryResponse)
    def query(payload: QueryRequest) -> QueryResponse:
        q = normalize_query(payload.question)
        if not q:  # pydantic already rejects empty via min_length; this guards
            # whitespace-only -> typed 422 (spec §7)
            raise HTTPException(status_code=422, detail="question must contain non-whitespace text")
        rows = store.search(q, top_k=payload.top_k or top_k)
        out = answerer.answer(question=payload.question, rows=rows)
        return QueryResponse(
            question=payload.question,
            answer=out.answer,
            citations=out.citations,
            refuse=out.refused,
            refuse_reason=out.refuse_reason,
            meta={"store_chunks": store.count() if store.is_loaded() else None},
        )

    return app


app = build_app()
