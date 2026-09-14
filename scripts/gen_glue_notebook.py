"""Generate notebooks/full_stack_glue.ipynb — the local full-stack glue notebook.

The phase-2 notebook (rag_pipeline.ipynb) is the INDEX side: it builds the
persisted store. This notebook is the QUERY side + the glue: it runs the whole
product locally from the repo's REAL modules (`backend.app.*`, `frontend.client`)
— no re-implementation, no servers:

  store (load the exported store, else build a deterministic demo one)
  -> the real FastAPI seam (`main.build_app`) driven over ASGI
  -> the real frontend client (`BackendClient`) over the same seam
  -> an eval that enforces "in-domain must ground, out-of-context must refuse"
  -> the notebook-contract drift guard (`tests/test_notebook_contract.py`)

Authoring it as a script keeps the cells lintable and reproducible:
    python scripts/gen_glue_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "full_stack_glue.ipynb"
MD = "markdown"
PY = "code"
CELLS: list[tuple[str, str]] = []  # (cell_type, source)


def md(src: str) -> None:
    CELLS.append((MD, src))


def code(src: str) -> None:
    CELLS.append((PY, src))


# ------------------------------------------------------------------------------
md("""\
# Medical Assistant — Full-Stack Glue (store → backend → frontend → eval)

One notebook that runs the **entire product** locally with the repo's real
modules — no re-implementation, no servers running.

**What it glues together**

1. **Store** (index side): loads the persisted store the phase-2 notebook
   (`notebooks/rag_pipeline.ipynb`) exported to `data/vector_store/chroma`
   (recognized by its `export_manifest.json`, which must be a non-demo build).
   If it's not there yet, it builds a **deterministic demo store** in
   `data/vector_store/demo_chroma` from a sample of `data/corpus` so the whole
   stack runs offline (GPU not needed) — demo builds never touch the real store
   dir.
2. **Backend** (query side): boots the real FastAPI seam `main.build_app(
   store=..., llm=...)` — the identical call shape production uses — and drives
   `/health` + `/query` over ASGI, no uvicorn.
3. **Frontend client**: drives the same seam through `BackendClient` (the exact
   path the Streamlit UI uses), also over ASGI.
4. **Eval**: in-domain questions must come back **grounded with citations**;
   the out-of-context question must come back as a **typed refusal** — the §7
   contract, not a demo nicety.
5. **Contract check**: runs `tests/test_notebook_contract.py` to prove the app
   and the notebook haven't drifted apart.

Run this from anywhere — it walks up to find the repo root. Env knobs
(`MAX_ROWS`, `EMBED_DEVICE`, `EMBED_MODEL`, `OLLAMA_URL`, `OLLAMA_MODEL`) let
you switch from demo to the real embedder/LLM.
""")

code("""\
# --- 0. locate repo root + import the REAL app modules ------------------------
from pathlib import Path
import os, sys, json

REPO = Path.cwd()
while REPO != REPO.parent and not (REPO / "backend" / "app" / "main.py").exists():
    REPO = REPO.parent
assert (REPO / "backend" / "app" / "main.py").exists(), "could not find the repo root"
sys.path.insert(0, str(REPO))

from backend.app.main import build_app
from backend.app.retrieval import BgeM3Embedder, DeterministicEmbedder, VectorStore
from backend.app.generation import DeterministicLlm, OllamaLlm
from backend.app.chunking import DocumentChunk

print("repo root:", REPO)
""")

code("""\
# --- 1. configuration (same knobs as the phase-2 notebook) --------------------
EMBED_MODEL  = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cpu")   # keep the GPU for Ollama
COLLECTION   = "medical_docs"
TOP_K        = int(os.environ.get("TOP_K", "5"))
MIN_SCORE    = float(os.environ.get("MIN_SCORE", "0.35"))
MAX_ROWS     = int(os.environ.get("MAX_ROWS", "800"))   # demo sample size
SEED         = 42
OLLAMA_URL   = os.environ.get("OLLAMA_URL", "")          # set to use the real LLM
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "command-r7b-arabic")
STORE_DIR    = REPO / "data" / "vector_store" / "chroma"

def _json_load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

REAL_STORE   = (
    (STORE_DIR / "chroma.sqlite3").exists()
    and (STORE_DIR / "export_manifest.json").exists()   # written by phase-2 notebook
    and _json_load(STORE_DIR / "export_manifest.json").get("demo_store") is not True
)
BUILD_REAL   = os.environ.get("BUILD_REAL", "0") == "1"  # build with bge-m3, not the demo
DEMO_DIR     = STORE_DIR.parent / "demo_chroma"          # demo builds never touch the real dir
""")

code("""\
# --- 2. the persisted store: load the real one, else build a demo -------------
def _demo_chunks() -> list[DocumentChunk]:
    # A small sample of data/corpus, chunked EXACTLY like the notebook cell 5:
    # one Q->A row = one chunk, doc_id ma::<src>::<seq:06d>, metadata keys
    # doc_id/collection/section/seq/chunk_id/lang/source.
    import csv, json, itertools

    def read_csv(path: Path):
        with open(path, newline="", encoding="utf-8") as fh:
            yield from itertools.islice(csv.DictReader(fh), MAX_ROWS)

    corpus: list[dict] = []
    ar_dir = REPO / "data" / "corpus" / "ar"
    for split in ("train", "val", "test"):
        p = ar_dir / f"{split}.csv"
        if not p.exists():
            continue
        for r in read_csv(p):
            q = (r.get("question") or "").strip()
            a = (r.get("answer") or "").strip()
            if q and a:
                corpus.append({"lang": "ar", "source": f"ar/{split}.csv",
                               "section": (r.get("label") or "general").strip(),
                               "text": q + "\\n" + a})
    en_p = REPO / "data" / "corpus" / "en" / "medquad.csv"
    if en_p.exists():
        for r in read_csv(en_p):
            q = (r.get("question") or "").strip()
            a = (r.get("answer") or "").strip()
            if q and a:
                corpus.append({"lang": "en", "source": "en/medquad.csv",
                               "section": (r.get("focus_area") or r.get("source") or "general").strip(),
                               "text": q + "\\n" + a})
    rx_p = REPO / "data" / "corpus" / "rx" / "all_prescriptions_clean.json"
    if rx_p.exists():
        for entry in json.loads(rx_p.read_text(encoding="utf-8")):
            meds = [m for m in entry.get("medications", []) if (m.get("Medication_name") or "").strip()]
            if not meds:
                continue
            parts = []
            for m in meds:
                t = m["Medication_name"]
                t += f" - {m['Dosage']}" if m.get("Dosage") else ""
                t += f" - {m['Frequency']}" if m.get("Frequency") else ""
                parts.append(t)
            corpus.append({"lang": "en", "source": "rx/all_prescriptions_clean.json",
                           "section": "prescription", "text": "; ".join(parts)})

    seqs: dict[str, int] = {}
    chunks: list[DocumentChunk] = []
    for row in corpus:
        seq = seqs.get(row["source"], 0)
        seqs[row["source"]] = seq + 1
        src = row["source"].replace("/", "__").replace(".", "_")
        doc_id = f"ma::{src}::{seq:06d}"
        chunks.append(DocumentChunk(
            doc_id=doc_id, collection=COLLECTION, seq=seq,
            section=row["section"], text=row["text"],
            metadata={"doc_id": doc_id, "collection": COLLECTION,
                      "section": row["section"], "seq": seq,
                      "chunk_id": f"{doc_id}/{seq:06d}",
                      "lang": row["lang"], "source": row["source"]},
        ))
    return chunks

if REAL_STORE:
    assert EMBED_MODEL not in ("", "TEST", "test"), (
        "a real store holds bge-m3 vectors; set EMBED_MODEL to the real model "
        "you indexed with (default BAAI/bge-m3)")
    embedder = BgeM3Embedder(model_name=EMBED_MODEL, device=EMBED_DEVICE)
    kind = "loaded (bge-m3 store from the phase-2 notebook)"
    USE_DIR = STORE_DIR
else:
    if BUILD_REAL:
        embedder = BgeM3Embedder(model_name=EMBED_MODEL, device=EMBED_DEVICE)
        kind = "built here with bge-m3 (BUILD_REAL=1)"
    else:
        embedder = DeterministicEmbedder(n_gram=3)  # default dim=1536 == backend demo mode
        kind = "demo store (deterministic embedder, no GPU)"
    if (STORE_DIR / "chroma.sqlite3").exists():
        print("  NOTE: data/vector_store/chroma/chroma.sqlite3 exists WITHOUT an export_manifest.json")
        print("        (stale/partial build) -> ignored; demo store built separately.")
    USE_DIR = DEMO_DIR

store = VectorStore(persist_dir=str(USE_DIR), embedder=embedder, collection=COLLECTION)
store.load()
if not REAL_STORE:
    store.reset()                # demo dir is disposable: rebuild from scratch each run
    store.add_documents(_demo_chunks())

print(f"store ({kind}):")
print("  dir       :", USE_DIR)
print("  collection:", store.collection)
print("  chunks    :", store.count())
if store.count() == 0:
    print("  NOTE: empty store -> every query will come back as a typed refusal until")
    print("  you build the store from notebooks/rag_pipeline.ipynb, or point")
""")

code("""\
# --- 3. boot the REAL FastAPI seam (same call shape production uses) ----------
use_real_llm = bool(OLLAMA_URL and OLLAMA_MODEL)
if use_real_llm:
    llm = OllamaLlm(host=OLLAMA_URL, model=OLLAMA_MODEL)
    print("llm: Ollama", OLLAMA_MODEL, "(if it is down, answers become typed refusals)")
else:
    llm = DeterministicLlm()
    print("llm: DeterministicLlm (set OLLAMA_URL/OLLAMA_MODEL for the real model)")

app = build_app(store=store, llm=llm, min_score=MIN_SCORE, top_k=TOP_K)
""")

code("""\
# --- 4. drive the API seam (ASGI, no server) ----------------------------------
from fastapi.testclient import TestClient

QUESTIONS = [
    ("ar", "ما الجرعة القصوى لباراسيتامول للبالغين؟"),
    ("en", "What is the maximum daily dose of paracetamol?"),
    ("ar", "كيف يُؤخذ الأسبيرين؟"),
]
OUT_OF_CONTEXT = ("en", "How do I fix a flat bicycle tyre?")

with TestClient(app) as c:
    health = c.get("/health").json()
    print("health:", health["status"], "| chunks:", health["store"]["chunks"],
          "| collection:", health["store"]["collection"])
    print()
    for lang, q in QUESTIONS + [OUT_OF_CONTEXT]:
        body = c.post("/query", json={"question": q}).json()
        tag = "REFUSE" if body["refuse"] else "ANSWER"
        print(f"[{lang}] {tag} :: {q}")
        if body["refuse"]:
            print(f"          reason: {body['refuse_reason']}")
        else:
            print(f"          {body['answer'][:100]!r}")
            for ci in body["citations"][:2]:
                print(f"          cite {ci['doc_id']} score={ci['score']:.3f}")
                print(f"            {ci['quoted'][:70]!r}")
        print()
""")

code("""\
# --- 5. eval: in-domain must ground; out-of-context must refuse ---------------
import csv

EVAL = [
    ("ar", "ما الجرعة القصوى لباراسيتامول للبالغين؟"),
    ("ar", "كيف يُؤخذ الأسبيرين؟"),
    ("ar", "متى يؤخذ الميتفورمين؟"),
    ("en", "What is the maximum daily dose of paracetamol?"),
    ("en", "How is metformin started?"),
    ("en", "How is aspirin taken?"),
    ("en", "How do I fix a flat bicycle tyre?"),   # out-of-context -> must refuse
    ("ar", "كيف أُصلح إطار دراجة مثقوب؟"),          # out-of-context -> must refuse
]

def run_eval() -> list[dict]:
    rows: list[dict] = []
    with TestClient(app) as c:
        for lang, q in EVAL:
            body = c.post("/query", json={"question": q, "top_k": TOP_K}).json()
            grounded = bool(body["citations"]) and not body["refuse"]
            in_domain = "دراجة" not in q and "bicycle" not in q
            passed = grounded if in_domain else body["refuse"]
            rows.append({
                "lang": lang, "in_domain": in_domain, "question": q,
                "answer": body["answer"][:60], "grounded": grounded,
                "refused": body["refuse"],
                "top_doc": body["citations"][0]["doc_id"] if body["citations"] else "",
                "top_score": body["citations"][0]["score"] if body["citations"] else 0.0,
                "passed": passed,
            })
    return rows

results = run_eval()
for r in results:
    flag = "PASS" if r["passed"] else "FAIL"
    print(f"[{flag}] [{r['lang']}] in={r['in_domain']} ground={r['grounded']} "
          f"refuse={r['refused']} score={r['top_score']:.3f}")
    print(f"      {r['question']}")
    if r["top_doc"]:
        print(f"      top: {r['top_doc']}")
passed = sum(r["passed"] for r in results)
print(f"PASS {passed}/{len(results)}")

STORE_DIR.mkdir(parents=True, exist_ok=True) if REAL_STORE else USE_DIR.mkdir(parents=True, exist_ok=True)
with open(USE_DIR / "eval_results.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)
print("wrote", USE_DIR / "eval_results.csv")
""")

code("""\
# --- 6. frontend client seam (the exact path the Streamlit UI uses) -----------
import asyncio
import httpx
from frontend.client import BackendClient

async def _demo() -> tuple:
    client = BackendClient("http://testserver", transport=httpx.ASGITransport(app=app))
    h = await client.health()
    b = await client.query("ما الجرعة القصوى للباراسيتامول؟")
    return h, b

h, b = asyncio.run(_demo())
print("client.health():", h["status"], "| chunks:", h["store"]["chunks"])
print("client.query() refuse:", b["refuse"], "| citations:", len(b["citations"]))
if b["citations"]:
    print("  cite[0]:", b["citations"][0]["doc_id"], "score", b["citations"][0]["score"])
""")

code("""\
# --- 7. contract drift guard: app <-> notebook still agree --------------------
import subprocess, sys

r = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "tests/test_notebook_contract.py"],
    cwd=str(REPO), capture_output=True, text=True,
)
for line in r.stdout.strip().splitlines()[-3:]:
    print(line)
assert r.returncode == 0, r.stdout + r.stderr
print("contract check OK")
""")

md("""\
## What's next / how the pieces run for real

- **Index side (build the store)**: run `notebooks/rag_pipeline.ipynb`
  top-to-bottom (set `EMBED_MODEL=BAAI/bge-m3` for the real semantic store;
  unset/`TEST` = deterministic demo store that runs without a GPU). Re-run this
  notebook and it silently switches from the demo store to the real bge-m3
  store when `export_manifest.json` marks it non-demo.
- **Query side (serving)**: `uvicorn backend.app.main:app --host 0.0.0.0
  --port 8000` — the app opens the same `data/vector_store/chroma` at startup.
- **UI**: `streamlit run app.py` (reads `BACKEND_URL`, talks to the backend via
  `frontend.client.BackendClient` — the seam exercised in cell 6).
- **Real LLM**: set `OLLAMA_URL=http://localhost:11434`,
  `OLLAMA_MODEL=command-r7b-arabic` in cell 1 and re-run — answers come from
  the model, every claim still cited or refused.
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