"""Build a local, model-free demo store (M5 "stranger clone→run" path).

The Kaggle notebook (§6, full corpus, bge-m3) is the production store the
backend serves. That needs a GPU + a Kaggle session. For anyone who just
clones the repo and wants to SEE the full stack run on a laptop, this builds
a small persistence-compatible store with the SAME store class (and a
deterministic embedder, no model download) matching the backend's demo mode
(EMBED_MODEL=TEST).

Usage:
    python scripts/build_store.py            # default: sample 2k ar + 0.5k en rows
    python scripts/build_store.py --max 2000
Then run the backend with EMBED_MODEL=TEST in the repo-root .env (see README).

The store contract is identical to the notebook's: collection medical_docs,
hnsw cosine, metadata keys doc_id/collection/section/seq/chunk_id.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO))

from backend.app.chunking import DocumentChunk
from backend.app.retrieval import DeterministicEmbedder, VectorStore

COLLECTION = "medical_docs"


def _read_csv(path: Path, cap: int) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))[:cap]
    return [
        {
            "lang": "ar" if "label" in r else "en",
            "source": str(path.relative_to(REPO / "data" / "corpus")),
            "section": (r.get("label") or r.get("focus_area") or "general").strip(),
            "question": (r.get("question") or "").strip(),
            "answer": (r.get("answer") or "").strip(),
        }
        for r in rows
        if (r.get("question") or "").strip() and (r.get("answer") or "").strip()
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=2500, help="rows per source file (cap)")
    ap.add_argument("--out", type=str, default=str(REPO / "data" / "vector_store" / "chroma"))
    args = ap.parse_args()

    corpus_root = REPO / "data" / "corpus"
    rows = []
    for split in ("train", "val", "test"):
        rows += _read_csv(corpus_root / "ar" / f"{split}.csv", args.max)
    rows += _read_csv(corpus_root / "en" / "medquad.csv", args.max)

    seqs: dict[str, int] = {}
    chunks: list[DocumentChunk] = []
    for row in rows:
        src = row["source"].replace("/", "__").replace(".", "_")
        seq = seqs.get(row["source"], 0)
        seqs[row["source"]] = seq + 1
        doc_id = f"ma::{src}::{seq:06d}"
        text = f"{row['question']}\n{row['answer']}".strip()
        chunks.append(
            DocumentChunk(
                doc_id=doc_id,
                collection=COLLECTION,
                seq=seq,
                section=row["section"],
                text=text,
                metadata={
                    "doc_id": doc_id,
                    "collection": COLLECTION,
                    "section": row["section"],
                    "seq": seq,
                    "chunk_id": f"{doc_id}/{seq:06d}",
                    "lang": row["lang"],
                    "source": row["source"],
                },
            )
        )

    store = VectorStore(persist_dir=args.out, embedder=DeterministicEmbedder())
    store.load()
    store.add_documents(chunks)
    store.persist()
    print(f"built demo store: {len(chunks)} chunks -> {args.out}")
    print("run the backend with EMBED_MODEL=TEST (repo-root .env) to serve it.")


if __name__ == "__main__":
    main()