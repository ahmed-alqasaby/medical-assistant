"""Response models — the API contract exposed to the frontend and the seam."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question (Arabic or English)")
    top_k: int | None = Field(default=None, ge=1, le=20)
    collection: str | None = None


class Citation(BaseModel):
    doc_id: str
    collection: str
    section: str
    quoted: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    refuse: bool = False
    refuse_reason: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)