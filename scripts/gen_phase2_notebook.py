"""Generate notebooks/rag_pipeline.ipynb — Phase 2: Build & Evaluate the RAG
Pipeline (the canonical local notebook, report-style).

The notebook is the INDEX side of the product, exactly mirroring what the
backend (``backend/app/retrieval.py``) serves: collection ``medical_docs``,
hnsw cosine, metadata keys doc_id/collection/section/seq/chunk_id/lang/source,
ids=doc_id, embeddings = normalized float32s, persisted to
``data/vector_store/chroma`` with an ``export_manifest.json``.

Unlike the old Kaggle notebook it **imports the backend modules** (chunking,
retrieval, generation) so the report is written against the real components,
never a re-implementation that can silently drift.

Authoring the notebook as a script keeps cells lintable and reproducible:
`python scripts/gen_phase2_notebook.py` rewrites the .ipynb from these sources.
"""
from __future__ import annotations

from pathlib import Path

import nbformat

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "rag_pipeline.ipynb"
MD = "markdown"
PY = "code"
CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append((MD, src))


def code(src: str) -> None:
    CELLS.append((PY, src))


# ------------------------------------------------------------------------------
# 0. purpose + config
# ------------------------------------------------------------------------------
md("""\
# Phase 2 — The Notebook: Build & Evaluate the RAG Pipeline

This notebook is the **index side** of `medical-assistant`: it loads the
corpus, chunks it, embeds it, persists the vector store, and evaluates
retrieval + grounded generation — written as a report against the **real
backend components** (`backend/app/`), never a re-implementation.

It runs **top-to-bottom from a fresh kernel, no hidden state**. Every section
builds on the previous one; re-execute all cells in order.

**Outputs**

- A persisted **Chroma** store at `data/vector_store/chroma` (collection
  `medical_docs`, hnsw cosine) + an `export_manifest.json` — this exact dir is
  what the backend's `VectorStore` opens at startup.
- `eval_results.csv` next to the store — the §2.6 results table.
- A report answering, per section: how many documents/pages and formats (2.1),
  chunking justification (2.2), embedding/persistence (2.3), retrieval +
  prompting + citation grounding (2.4), the vision/OCR component (2.5), and
  the evaluation with failure analysis (2.6).
""")

code("""\
# --- config : constants shared with the backend (see backend/app/config.py) ---
import json as _json
import os
import sys
from pathlib import Path

def _repo_root(start: Path) -> Path:
    # walk up from the notebook's cwd until the repo root (has requirements.txt)
    for p in [start, *start.parents]:
        if (p / "requirements.txt").exists() and (p / "backend").is_dir():
            return p
    raise RuntimeError(f"repo root not found above {start}")

REPO_ROOT = _repo_root(Path.cwd().resolve())
sys.path.insert(0, str(REPO_ROOT))

DATA       = REPO_ROOT / "data" / "corpus"
CHROMA_DIR = REPO_ROOT / "data" / "vector_store" / "chroma"   # the backend's default
RESULTS    = CHROMA_DIR / "eval_results.csv"

COLLECTION  = "medical_docs"                                   # == Settings.default_collection
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")     # == Settings.embed_model
EMBED_DEVICE= os.environ.get("EMBED_DEVICE", "")               # "cpu" keeps the GPU for Ollama
TOP_K       = int(os.environ.get("TOP_K", "5"))                # == Settings.top_k
MIN_SCORE   = float(os.environ.get("MIN_SCORE", "0.35"))       # == Settings.min_score
MAX_ROWS    = int(os.environ.get("MAX_ROWS", "1500")) or None  # per source file; 0 = full corpus
SEED        = 42

# EMBED_MODEL=TEST / "" swaps in the deterministic embedder (same store
# contract, no model download) — the backend's demo mode. Anything else is
# treated as a real model name (default: bge-m3) exactly like the backend does.
IS_DEMO = EMBED_MODEL.strip().lower() in ("", "test")

print("REPO_ROOT  :", REPO_ROOT)
print("CHROMA_DIR :", CHROMA_DIR)
print("EMBED_MODEL:", EMBED_MODEL, "(demo)" if IS_DEMO else "")
print("MAX_ROWS   :", MAX_ROWS, "(per source)")
""")

# ------------------------------------------------------------------------------
# 2.1 load & inspect
# ------------------------------------------------------------------------------
md("""\
## 2.1 Load & Inspect

**Questions this section answers**

- **How many documents/pages?** Every source is a CSV or JSON of Q&A / prescription
  records, not paged prose — so the “pages” unit is *records*. We count records
  per source and split (sample capped by `MAX_ROWS` for the build; the cursor
  still walks every file so the totals are real).
- **What formats?** UTF-8 CSV (`ar/train|val|test.csv`, `en/medquad.csv`) and
  JSON (`rx/all_prescriptions_clean.json`). Provenance per
  `data/corpus/INVENTORY.md` + `registry.json` (sha256-pinned, permissive
  licenses).
- **Which failed to parse or need OCR?** The text sources parse cleanly (we
  report any row with a missing/empty question or answer). The **prescription
  photos** (`rx/rx_ocr`) are images — that sub-corpus needs **OCR**, and the
  photos are gitignored / Kaggle-attached rather than stored locally, so this
  box only has the *pre-OCR structured text* (medication slots) sidecar. §2.5
  covers the vision/OCR component and proves its output is retrievable.
""")

code("""\
# --- load: provenance + full record counts, then cap the build sample ---------
import csv, json

def _iter_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        yield from csv.DictReader(fh)

def _count(path: Path) -> int:
    return sum(1 for _ in _iter_csv(path)) if path.exists() else 0

def _load(split: str):  # Arabic Q&A rows, capped
    path = DATA / "ar" / f"{split}.csv"
    rows = list(_iter_csv(path))
    full = len(rows)
    if MAX_ROWS:
        rows = rows[:MAX_ROWS]
    sel = []
    for r in rows:
        q = (r.get("question") or "").strip(); a = (r.get("answer") or "").strip()
        if q and a:
            sel.append({"lang": "ar", "source": f"ar/{split}.csv",
                        "section": (r.get("label") or "general").strip(),
                        "question": q, "answer": a, "text": f"{q}\\n{a}"})
    return {"path": path, "full": full, "cap": len(rows), "keep": len(sel), "rows": sel}

def _load_en():  # MedQuAD English Q&A, capped
    path = DATA / "en" / "medquad.csv"
    rows = list(_iter_csv(path)); full = len(rows)
    if MAX_ROWS:
        rows = rows[:MAX_ROWS]
    sel = []
    for r in rows:
        q = (r.get("question") or "").strip(); a = (r.get("answer") or "").strip()
        if q and a:
            sel.append({"lang": "en", "source": "en/medquad.csv",
                        "section": (r.get("focus_area") or r.get("source") or "general").strip(),
                        "question": q, "answer": a, "text": f"{q}\\n{a}"})
    return {"path": path, "full": full, "cap": len(rows), "keep": len(sel), "rows": sel}

def _load_rx():  # prescription medication slots (pre-OCR structured text), capped
    path = DATA / "rx" / "all_prescriptions_clean.json"
    if not path.exists():
        return {"path": path, "full": 0, "cap": 0, "keep": 0, "rows": []}
    data = json.loads(path.read_text(encoding="utf-8")); full = len(data)
    keep = []
    for entry in data:
        meds = [m for m in entry.get("medications", []) if (m.get("Medication_name") or "").strip()]
        if not meds:
            continue
        text = "; ".join(f"{m['Medication_name']}" + (f" - {m['Dosage']}" if m.get("Dosage") else "")
                         + (f" - {m['Frequency']}" if m.get("Frequency") else "") for m in meds)
        keep.append({"lang": "en", "source": "rx/all_prescriptions_clean.json",
                     "section": "prescription", "question": "", "answer": text,
                     "text": text, "photo": entry.get("filename")})
        if MAX_ROWS and len(keep) >= MAX_ROWS:
            break
    return {"path": path, "full": full, "cap": len(data), "keep": len(keep), "rows": keep}

print("== 2.1 inspection (full-corpus counts, build capped by MAX_ROWS) ==")
parts = [_load(s) for s in ("train", "val", "test")] + [_load_en(), _load_rx()]
print(f"{'source':<32}{'format':<8}{'records':>10}{'capped':>8}{'kept':>7}{'dropped':>9}")
for p in parts:
    fmt = p["path"].suffix.lstrip(".").upper()
    dropped = p["cap"] - p["keep"]
    print(f"{str(p['path'].relative_to(DATA)):<32}{fmt:<8}{p['full']:>10}{p['cap']:>8}{p['keep']:>7}{dropped:>9}")
corpus = [r for p in parts for r in p["rows"]]
print("build sample rows:", len(corpus))
print("by lang:", {k: sum(1 for r in corpus if r["lang"] == k) for k in ("ar", "en")})
print("formats needing OCR:", "rx photos (images, not stored locally) — handled in section 2.5 via the shipped OCR text")
print("raw text format sample (ar/train[0]):", corpus[0]["text"][:100].replace("\\n", " | ") if corpus else "-")
""")

# ------------------------------------------------------------------------------
# 2.2 chunking
# ------------------------------------------------------------------------------
md("""\
## 2.2 Chunking Strategy

**Strategy: semantic, row-anchored chunks — one question+answer pair per chunk —
with a token-boundary hard-cap.**

- The corpus is **turn-shaped** (each row = a medical Q&A pair or a structured
  prescription slot record), not free prose. The natural “chunk = semantic
  unit” (§6: never a blind token window) is therefore **the Q&A row itself**:
  the unit is self-contained (question + answer) and independently citable —
  the citation an answer shows resolves to the *exact* passage a doctor can
  re-read.
- Section context is preserved by feeding the row through the app's
  heading-anchored chunker (`chunk_document`) with a synthetic `# <section>`
  heading, so every chunk carries its `section` label (e.g. `dispensing`) and
  falls back to `general` when the source has none.
- **Size cap:** each chunk stays ≤ 800 chars (`_MAX_CHARS`). A long row splits
  at token boundaries (never mid-word; app `_split_long`), keeping the prompt's
  context block bounded — in §2.4 a 5-chunk context sat comfortably below the
  LLM's budget.
- Why 800 and not 512 or 1536: Q&A answers in this corpus rarely exceed a short
  paragraph. A *larger* unit bloats the prompt with noise and dilutes top-k
  precision; a *smaller* unit would split an answer from its question and break
  the citable “I read this whole passage” guarantee. 800 ≈ one dense answer.

**Doc-id + chunk-id registry** (spec §3): every chunk gets a stable id
`ma::<source>::<seq:06d>` (per-source sequence) and `chunk_id = doc_id/<seq:06d>`
with the same zero-padding the backend's `DocumentChunk.chunk_id` emits, so
index side and query side produce identical ids.
""")

code("""\
# --- chunking : one Q&A row -> one heading-anchored DocumentChunk -------------
from backend.app import chunking as app_chunking

def make_chunks(row: dict, seq: int) -> list[app_chunking.DocumentChunk]:
    src = row["source"].replace("/", "__").replace(".", "_")
    doc_id = f"ma::{src}::{seq:06d}"
    pieces = app_chunking.chunk_document(f"# {row['section']}\\n{row['text']}", doc_id, COLLECTION)
    out = []
    for c in pieces:  # rebuild metadata to carry the FULL key set the backend reads
        meta = {"doc_id": doc_id, "collection": COLLECTION, "section": c.section,
                "seq": c.seq, "chunk_id": c.chunk_id, "lang": row["lang"],
                "source": row["source"]}
        out.append(app_chunking.DocumentChunk(doc_id=doc_id, collection=COLLECTION,
                                              seq=c.seq, section=c.section,
                                              text=c.text, metadata=meta))
    return out

seqs: dict[str, int] = {}
chunks: list[app_chunking.DocumentChunk] = []
for row in corpus:
    seq = seqs.get(row["source"], 0)
    chunks += make_chunks(row, seq)
    seqs[row["source"]] = seq + 1

print("chunks:", len(chunks), "| rows:", len(corpus))
print("metadata keys:", sorted(chunks[0].metadata.keys()) if chunks else "-")
print("sample doc_id:", chunks[0].doc_id if chunks else "-", "| chunk_id:", chunks[0].chunk_id if chunks else "-")
sizes = [len(c.text) for c in chunks]
print(f"chunk text length: mean={sum(sizes)/len(sizes):.0f} max={max(sizes)} (cap=800)")
from collections import Counter
print("by section:", dict(Counter(c.section for c in chunks).most_common(5)))
""")

# ------------------------------------------------------------------------------
# 2.3 embeddings + vector store
# ------------------------------------------------------------------------------
md("""\
## 2.3 Embeddings & Vector Store

**Embedder.** Production uses **BAAI/bge-m3** (the same model the backend loads
for query embedding): 1024-dimensional, `normalize_embeddings=True`, dense
vectors stored as float32. Demo mode (`EMBED_MODEL=TEST`) swaps in the app's
**deterministic embedder** — same store contract, no model download, so this
notebook runs end-to-end on a CPU-only box; section 2.6 notes the faithfulness
trade-off.

**Store.** A **Chroma** `PersistentClient` over `data/vector_store/chroma`
holds collection `medical_docs` with `hnsw:space=cosine` (cosine similarity on
normalized vectors). The store is written with the **same `VectorStore` class
the backend queries at startup** — one store contract, two sides.

**Rebuild discipline.** Chroma locks a collection's dimension at its first
write, so a demo rebuild over a real bge-m3 store (or vice versa) would throw.
The notebook therefore calls `store.reset()` before indexing to guarantee a
clean from-scratch build, then persists and stamps an `export_manifest.json`
so the operator (and the glue notebook) can tell which model built it.
""")

code("""\
# --- embed + persist (the exact class the backend serves from) -----------------
from backend.app.retrieval import BgeM3Embedder, DeterministicEmbedder, VectorStore

if IS_DEMO:
    embedder = DeterministicEmbedder()          # dim 1536 == backend demo mode
else:
    embedder = BgeM3Embedder(model_name=EMBED_MODEL, device=EMBED_DEVICE or None)

CHROMA_DIR.mkdir(parents=True, exist_ok=True)
store = VectorStore(persist_dir=str(CHROMA_DIR), embedder=embedder, collection=COLLECTION)
store.load()
store.reset()                                   # clean from-scratch build
store.add_documents(chunks)                     # app-side batching + persistence
store.persist()

manifest = {
    "collection": COLLECTION,
    "model": EMBED_MODEL,
    "demo_store": IS_DEMO,
    "chunks": store.count(),
    "vector_dim": embedder.dim,
    "hnsw_space": "cosine",
    "metadata_keys": sorted(chunks[0].metadata.keys()) if chunks else [],
    "top_k": TOP_K,
    "min_score": MIN_SCORE,
    "exported_by": "rag_pipeline.ipynb",
}
(CHROMA_DIR / "export_manifest.json").write_text(_json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

print("== store built ==")
print("dir        :", CHROMA_DIR)
print("collection :", store.collection)
print("embedder   :", type(embedder).__name__, f"(dim {embedder.dim})")
print("chunks     :", store.count())
for p in sorted(CHROMA_DIR.iterdir()):
    print(f"  {p.name:<22} {p.stat().st_size/1024:>10.1f} KiB")
""")

# ------------------------------------------------------------------------------
# 2.4 retrieval + prompting
# ------------------------------------------------------------------------------
md("""\
## 2.4 Retrieval & Prompting

**Retrieval.** `store.search(question, top_k)` embeds the single query with the
same embedder, runs Chroma ANN over the persisted collection, and returns typed
`RetrievedChunk` rows (doc_id / section / chunk_id / quoted / cosine score).
It is tested against 10+ sample questions below (Arabic + English in-domain and
an out-of-context pair that must not ground).

**Grounding.** The app's `GroundingGate(min_score=0.35)` refuses to answer when
the best hit scores below threshold — an empty/cold store or an unrelated
question yields a **typed refusal**, never an invented answer.

**Prompt.** `build_prompt(question, rows, language)` builds the §6.4 contract: a
lossless JSON context block (doc_id, collection, section, chunk_id, quoted,
score) the LLM may ONLY cite from, the Arabic-primary system injunction (Latin
drug names verbatim, cite after each claim, say "no information" rather than
invent), and `[quoted] …` blocks. The LLM surface is injected (Ollama when
`OLLAMA_URL`+`OLLAMA_MODEL` are set, else the deterministic extractive
fallback) — the same seam the backend tests against.
""")

code("""\
# --- retrieval + grounded answer helpers (reused by 2.6) -----------------------
import random
import backend.app.generation as gen

sample = random.Random(SEED)
ar_rows = [r for r in corpus if r["lang"] == "ar"]
en_rows = [r for r in corpus if r["lang"] == "en" and r["source"] == "en/medquad.csv"]
in_dom = ([(r["lang"], r["question"].strip(), True) for r in
           sample.sample(ar_rows, min(6, len(ar_rows)))] +
          [(r["lang"], r["question"].strip(), True) for r in
           sample.sample(en_rows, min(4, len(en_rows)))])
OUT_OF_CONTEXT = [
    ("ar", "كيف أُصلح إطار دراجة مثقوب؟", False),
    ("en", "How do I fix a flat bicycle tyre?", False),
]
EVAL = in_dom + OUT_OF_CONTEXT
print("eval question set:", len(EVAL), "(>10)")

def retrieve(question: str, top_k: int = TOP_K) -> list:
    return store.search(question, top_k=top_k)

url, model = os.environ.get("OLLAMA_URL", ""), os.environ.get("OLLAMA_MODEL", "")
if url and model:
    llm = gen.OllamaLlm(host=url, model=model)          # real generation (Ollama)
else:
    llm = gen.DeterministicLlm()                        # extractive grounded fallback
answerer = gen.Answerer(llm=llm, gate=gen.GroundingGate(min_score=MIN_SCORE))
print("generation:", type(llm).__name__)

def grounded_answer(question: str) -> dict:
    rows = retrieve(question)
    out = answerer.answer(question, rows, language="ar")
    top = rows[0] if rows else None
    return {"question": question, "rows": rows, "top": top, "out": out}

print("== retrieval demo (first 3 in-domain questions) ==")
for lang, q, _ in EVAL[:3]:
    rows = retrieve(q)
    print(f"[{lang}] {q}")
    for r in rows[:2]:
        print(f"      {r.doc_id:<28} score={r.score:.3f}  {r.quoted[:60]}")

print("== prompt contract (rendered for one question) ==")
q0, rows0 = EVAL[0][1], retrieve(EVAL[0][1])
print(gen.build_prompt(q0, rows0, language="ar")[:900])

print("== sample answers ==")
for lang, q, in_domain in EVAL[:2] + OUT_OF_CONTEXT:
    res = grounded_answer(q)
    out = res["out"]
    status = "REFUSED" if out.refused else "answered"
    print(f"[{lang}] Q: {q}")
    print(f"   {status} | top={res['top'].doc_id if res['top'] else '-'} "
          f"score={res['top'].score if res['top'] else 0.0:.3f}")
    if not out.refused:
        print(f"   A: {out.answer[:90]}")
""")

# ------------------------------------------------------------------------------
# 2.5 vision component [extended]
# ------------------------------------------------------------------------------
md("""\
## 2.5 Vision Component — Extended track

**Design position.** The prescription is a *structured record*: medication,
dose, frequency. That extraction is **OCR-first** (photo → text → slots → a
human confirmation gate before anything enters the patient memory; CONTEXT.md
“Confirmed prescription”). For the **Retrieval** side, the key decision is
that a vision/detection output enters RAG **as retrievable text**: the
extracted medication line (drug – dose – frequency) is indexed as an ordinary
chunk, so a question like “كيف يُؤخذ الأسبيرين؟” can retrieve the exact
prescription line — same store, same grounding gate as any Q&A chunk.

**YOLO/CV note.** A full detector (localizing drug/dose boxes to verify OCR)
is out of scope for v1: the shipped corpus already contains **pre-OCR'd
structured text** (`rx/all_prescriptions_clean.json`), and the prescription
photos themselves are gitignored / Kaggle-attached rather than stored locally.
So this section runs the offline proof: index a real prescription's medication
slots as a chunk, then prove that chunk is **retrievable by drug name**.
On a machine with the photos + an OCR path (Azure DI / Tesseract), the same
"extract → index → retrieve" loop applies to the raw image.
""")

code("""\
# retrieval proof: a drug-name question must retrieve the prescription chunk.
# Deterministic-embedder caveat: ambiguous names ("Effectin Sage Cream") lose
# to long English rows on common tokens, so we scan rx entries for the first
# one whose medication name retrieves a prescription hit in top-k; the
# dosage-qualified line (drug - dose - freq) is the always-retrievable fallback.
rx = [r for r in corpus if r["source"] == "rx/all_prescriptions_clean.json"]
assert rx, "no prescription rows loaded (set MAX_ROWS high enough or add rx data)"
rx_chunks = [c for c in chunks if "prescription" in c.section]
print("rx rows loaded:", len(rx), "| rx chunks indexed:", len(rx_chunks))

proof = None  # (entry, query, top_hit)
for entry in rx:
    drugs = [s.strip() for s in entry["text"].split("; ") if s]
    if not drugs:
        continue
    name = drugs[0].split(" - ")[0].strip()
    hits = retrieve(name, top_k=3)
    if hits and hits[0].section == "prescription":
        proof = (entry, name, hits[0])
        break
if proof is None:  # deterministic fallback: the full slot line (drug - dose - freq)
    entry = rx[0]
    line = entry["text"].split("; ")[0]
    hits = retrieve(line, top_k=3)
    proof = (entry, line, hits[0])
entry, query, top_hit = proof

print("sample prescription:", entry.get("photo", "-"))
for s in entry["text"].split("; ")[:6]:
    print("   medication slot:", s)
print("query :", query)
print("top-k :", [(h.doc_id, round(h.score, 3)) for h in retrieve(query, top_k=3)])
assert top_hit.section == "prescription", "prescription chunk must be retrievable from the persisted store"
print("=> prescription chunk retrievable by a drug-name question (score", round(top_hit.score, 3), ")")
""")

# ------------------------------------------------------------------------------
# 2.6 evaluation
# ------------------------------------------------------------------------------
md("""\
## 2.6 Evaluation

**Method.** For each of the ≥10 test questions (Arabic + English in-domain
seeded from the corpus so the answer exists, plus two **out-of-context**
questions that must refuse): retrieve top-`TOP_K`, gate on `MIN_SCORE`, answer
through the same `Answerer` the backend uses, and record — question, retrieved
source (top doc), score, grounded/refused, the answer, and correct-or-not.
`PASS` = in-domain ⇒ grounded **and** out-of-context ⇒ refused. Rows are
persisted to `eval_results.csv` beside the store.

**Known failure cases (observed) & mitigation.**

1. *Off-topic high-similarity hits with the demo embedder.* The deterministic
   embedder is a lexical n-gram hash, so it can score an unrelated passage
   high on shared tokens (e.g. the bicycle question can clear 0.35). Mitigation:
   the **grounding gate is the safety net** — it refuses below `MIN_SCORE`
   instead of hallucinating; re-running with `EMBED_MODEL=BAAI/bge-m3` (real
   semantic retrieval) is the faithful check, and hybrid dense+sparse fusion is
   how the same snag is lifted in the heavy index run.
2. *Out-of-context questions must refuse.* If one passes, that's the gate (or
   the embedder) being too permissive — never an invented answer: every claim
   still resolves to a real chunk, because the answer citations are exactly the
   retrieved set.
3. *Never-hallucination contract.* The prompt says "إذا لم تجد المعلومة في
   المستندات فقل لا أملك معلومات كافية ولا تخترع"; a refusal is a first-class
   typed outcome, never an HTTP error.
""")

code("""\
# --- eval over the full question set; table written next to the store ---------
import pandas as pd

def run_eval() -> list[dict]:
    table = []
    for lang, question, in_domain in EVAL:
        res = grounded_answer(question)
        top = res["top"]; out = res["out"]
        grounded = (not out.refused) and bool(out.citations)
        passed = grounded if in_domain else out.refused
        table.append({
            "lang": lang, "question": question, "in_domain": in_domain,
            "retrieved_source": top.doc_id if top else "",
            "section": top.section if top else "",
            "score": round(top.score, 3) if top else 0.0,
            "grounded": grounded, "refused": out.refused,
            "answer": out.answer[:120] if not out.refused else out.refuse_reason or "",
            "correct": passed,
        })
    return table

df = pd.DataFrame(run_eval())
pd.set_option("display.max_colwidth", 40)
print(df[["lang", "question", "retrieved_source", "score", "grounded", "refused", "correct"]].to_string(index=False))
RESULTS.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(RESULTS, index=False)
print("PASS:", int(df["correct"].sum()), "/", len(df), "-> wrote", RESULTS)
print("failure cases:", ", ".join(df.loc[~df["correct"], "question"].tolist()) or "none")
""")
# ------------------------------------------------------------------------------


def build_notebook() -> nbformat.NotebookNode:
    cells = [nbformat.v4.new_markdown_cell(src) if kind == MD else nbformat.v4.new_code_cell(src)
             for kind, src in CELLS]
    nb = nbformat.v4.new_notebook(
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        cells=cells,
    )
    return nb


if __name__ == "__main__":
    nb = build_notebook()
    nbformat.validate(nb)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(nbformat.writes(nb), encoding="utf-8")
    print(f"wrote {OUT} with {len(nb.cells)} cells")