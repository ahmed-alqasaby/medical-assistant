"""Generate notebooks/kaggle_rag_pipeline.ipynb (M2 deliverable).

Authoring the notebook as a script keeps the cells lintable/importable and
reproducible: `python scripts/gen_rag_notebook.py` rewrites the .ipynb from
these cell sources. The notebook is self-contained on Kaggle (GPU, corpus
attached as the medical-assistant-corpus dataset) — it must NOT import from
this repo; it re-implements the exact store contract the backend serves
(collection `medical_docs`, hnsw cosine, metadata keys doc_id/collection/
section/seq/chunk_id, ids=doc_id, embeddings=normalized bge-m3 floats).
"""

from __future__ import annotations

from pathlib import Path

import nbformat

OUT = Path("notebooks") / "kaggle_rag_pipeline.ipynb"
MD = "markdown"
PY = "code"
CELLS: list[tuple[str, str]] = []  # (cell_type, source)


def md(src: str) -> None:
    CELLS.append((MD, src))


def code(src: str) -> None:
    CELLS.append((PY, src))


# ------------------------------------------------------------------------------
md(f"""\
# Medical Assistant — RAG Pipeline (M2, retrieval backbone + extended OCR)

Runs **top-to-bottom from a fresh kernel, no hidden state**. Every cell is
independent; re-execute in order.

**What this notebook does**

1. Load the corpus (Arabic medical Q&A, English MedQuAD, prescription OCR).
2. Chunk at the **turn / semantic level** (one Q->A row = one chunk), with
   provenance (source file + row + section) kept in chunk metadata.
3. Embed with **bge-m3 (dense + sparse)** and persist to a **Chroma** store
   (collection `medical_docs`, hnsw cosine). Dense vectors go to Chroma; the
   sparse (lexical) weights go to a JSON sidecar so ranking is a **hybrid
   dense+sparse** fusion.
4. Retrieve from the **persisted** store (fresh `PersistentClient`, not
   in-memory state).
5. Extended track: OCR a handwritten-prescription photo (Azure Document
   Intelligence if a key is present, else Tesseract, else the Microsoft-OCR
   text shipped with the corpus) and prove its extracted text is retrievable.
6. Grounded, cited Arabic-primary answer (prompt contract: Latin drug names
   preserved verbatim, every answer cites its retrieved source).
7. Eval over >=10 curated Arabic + English questions — including an
   out-of-context question that must refuse/ground, never invent — recorded
   as a results table.
8. **Export** the persisted store that the FastAPI backend loads at startup.

The store contract here exactly matches `backend/app/retrieval.py`:
collection name `medical_docs`, `hnsw:space` = cosine, metadata keys
`doc_id` / `collection` / `section` / `seq` / `chunk_id`, `ids` = `doc_id`,
embeddings = **normalized** float32 vectors.
""")

md("""\
## Environment

- **Accelerator**: GPU (T4/P100/T4 x2 all fine).
- **Internet**: on (installs + model download below).
- Attach the dataset **`<handle>/medical-assistant-corpus`** (push it via
  `scripts/kaggle_connect.sh`) — it contains `ar/`, `en/`, `rx/`, `rx_ocr/`,
  `registry.json`, `INVENTORY.md` at its root.
> Optional env knob `MAX_ROWS` caps the number of corpus rows indexed per
> split — handy for a fast re-run; unset/0 = full corpus.
""")

code("""\
# --- 0. installs (quiet) -----------------------------------------------------
import subprocess, sys

def _quiet_pip(*pkgs: str) -> str:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + list(pkgs),
                       capture_output=True, text=True)
    return (r.returncode, r.stderr[-500:]) if r.returncode else (0, "")

for pkg in ["chromadb==1.5.9", "sentence-transformers==3.4.1", "numpy>=1.26"]:
    rc, tail = _quiet_pip(pkg)
    print(f"install {pkg}: rc={rc} {tail}")
print("install done")
""")

code("""\
# --- 1. configuration (no hidden state; recomputed every run) ----------------
import os, json, hashlib, time, math
from pathlib import Path

MODEL_NAME   = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
COLLECTION   = "medical_docs"
TOP_K        = int(os.environ.get("TOP_K", "5"))
MIN_SCORE    = float(os.environ.get("MIN_SCORE", "0.35"))
MAX_ROWS     = int(os.environ.get("MAX_ROWS", "0")) or None  # None = full corpus
SEED         = 42

KAGGLE_INPUT = Path("/kaggle/input/medical-assistant-corpus")
KAGGLE_OUT   = Path("/kaggle/working")
DATA_ROOT    = KAGGLE_INPUT if KAGGLE_INPUT.exists() else Path("../data/corpus")
OUT_DIR      = KAGGLE_OUT / "vector_store" if str(DATA_ROOT).startswith("/kaggle") else Path("../data/vector_store")
CHROMA_DIR   = OUT_DIR / "chroma"
print("DATA_ROOT :", DATA_ROOT)
print("CHROMA_DIR:", CHROMA_DIR)
assert DATA_ROOT.exists(), f"corpus root missing: {DATA_ROOT}"
""")

code("""\
# --- 2. load corpus with provenance -------------------------------------------
import csv

def _read_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))

def _load_ar(split: str):
    rows = _read_csv(DATA_ROOT / "ar" / f"{split}.csv")
    if MAX_ROWS:
        rows = rows[:MAX_ROWS]
    return [{"lang": "ar", "split": split,
             "source": f"ar/{split}.csv",
             "section": (r.get("label") or "general").strip(),
             "question": (r.get("question") or "").strip(),
             "answer": (r.get("answer") or "").strip(),
             "text": ((r.get("question") or "") + "\\n" + (r.get("answer") or "")).strip()}
            for r in rows if (r.get("question") or "").strip() and (r.get("answer") or "").strip()]

def _load_en():
    rows = _read_csv(DATA_ROOT / "en" / "medquad.csv")
    if MAX_ROWS:
        rows = rows[:MAX_ROWS]
    return [{"lang": "en", "split": "medquad",
             "source": "en/medquad.csv",
             "section": (r.get("focus_area") or r.get("source") or "general").strip(),
             "question": (r.get("question") or "").strip(),
             "answer": (r.get("answer") or "").strip(),
             "text": ((r.get("question") or "") + "\\n" + (r.get("answer") or "")).strip()}
            for r in rows if (r.get("question") or "").strip() and (r.get("answer") or "").strip()]

def _load_rx_ocr():
    # Prescriptions: fine-grained OCR text shipped in the corpus (Microsoft OCR).
    out = []
    rx_json = DATA_ROOT / "rx" / "all_prescriptions_clean.json"
    if rx_json.exists():
        data = json.loads(rx_json.read_text(encoding="utf-8"))
        for entry in data:
            meds = [m for m in entry.get("medications", []) if (m.get("Medication_name") or "").strip()]
            if not meds:
                continue
            text = "; ".join(
                f"{m['Medication_name']}"
                + (f" - {m['Dosage']}" if m.get("Dosage") else "")
                + (f" - {m['Frequency']}" if m.get("Frequency") else "")
                for m in meds)
            out.append({"lang": "en", "split": "rx_ocr", "source": "rx/all_prescriptions_clean.json",
                        "section": "prescription", "question": "",
                        "answer": text, "text": text,
                        "photo": entry.get("filename")})
    return out

corpus = _load_ar("train") + _load_ar("val") + _load_ar("test") + _load_en() + _load_rx_ocr()
print("corpus rows:", len(corpus))
print("by lang:", {k: sum(1 for r in corpus if r["lang"] == k) for k in ("ar", "en")})
print("has photos:", sum(1 for r in corpus if r.get("photo")))
""")

code("""\
# --- 3. turn/semantic chunking with provenance ---------------------------------
# One Q->A row = one chunk (the dataset is already turn-shaped). Each chunk
# carries provenance: doc_id (registry-style), source file, row seq, section.
# Chunk metadata is EXACTLY what the backend reads back on search.
def chunk_row(row: dict, seq: int) -> dict:
    src = row["source"].replace("/", "__").replace(".", "_")
    doc_id = f"ma::{src}::{seq:06d}"
    meta = {
        "doc_id": doc_id,
        "collection": COLLECTION,
        "section": row["section"],
        "seq": seq,
        "chunk_id": f"{doc_id}/{seq:06d}",
        "lang": row["lang"],
        "source": row["source"],
    }
    return {"doc_id": doc_id, "text": row["text"], "metadata": meta}

# stable per-source sequence numbers (doc_id must be unique + stable)
seqs: dict[str, int] = {}
chunks: list[dict] = []
for row in corpus:
    seq = seqs.get(row["source"], 0)
    chunks.append(chunk_row(row, seq))
    seqs[row["source"]] = seq + 1

print("chunks:", len(chunks))
print("metadata keys:", sorted(chunks[0]["metadata"].keys()))
print("sample doc_id:", chunks[0]["doc_id"], "| section:", chunks[0]["metadata"]["section"])
""")

code("""\
# --- 4. embed (bge-m3 dense + sparse) and persist to Chroma --------------------
import numpy as np
import chromadb

print("loading embedder:", MODEL_NAME, flush=True)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(MODEL_NAME)

def embed_dense(texts: list[str]) -> np.ndarray:
    return model.encode(list(texts), normalize_embeddings=True,
                        batch_size=32, show_progress_bar=True)

def embed_sparse(texts: list[str]) -> list[dict[str, float]]:
    # bge-m3 lexical (sparse) weights; falls back to BM25-style TF counts if
    # the version does not expose them, so the cell always runs top-to-bottom.
    try:
        out = model.encode(list(texts), return_sparse=True)
        return [dict(getattr(emb, "at", {})) for emb in out]
    except Exception:
        from collections import Counter
        import re
        res = []
        for t in texts:
            toks = re.findall(r"[a-zA-Z]+|[\\u0600-\\u06FF]+", t.lower())
            res.append(dict(Counter(toks)))
        return res

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
coll = client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"},  # normalized bge-m3 -> cosine similarity
)

B = 256
sparse_index: list[dict] = []
for start in range(0, len(chunks), B):
    batch = chunks[start:start + B]
    texts = [c["text"] for c in batch]
    dense = embed_dense(texts)
    sparse = embed_sparse(texts)
    coll.upsert(
        ids=[c["doc_id"] for c in batch],
        embeddings=[v.tolist() for v in dense],
        documents=[c["text"] for c in batch],
        metadatas=[c["metadata"] for c in batch],
    )
    sparse_index.extend(
        {"doc_id": c["doc_id"], "weights": sp} for c, sp in zip(batch, sparse))

# persist the sparse lens (JSON sidecar; Chroma stores dense vectors)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
(CHROMA_DIR / "sparse_index.json").write_text(
    json.dumps(sparse_index, ensure_ascii=False), encoding="utf-8")

print("persisted chunks:", coll.count())
print("sparse sidecar  :", len(sparse_index))
""")

code("""\
# --- 5. retrieval from the PERSISTED store (fresh client, no in-memory state) --
import json
import numpy as np

client2 = chromadb.PersistentClient(path=str(CHROMA_DIR))   # fresh handle over disk
coll2 = client2.get_collection(COLLECTION)
sparse_index = json.loads((CHROMA_DIR / "sparse_index.json").read_text(encoding="utf-8"))
sparse_lookup = {s["doc_id"]: s["weights"] for s in sparse_index}

def _dense(query: str) -> list[float]:
    v = model.encode([query], normalize_embeddings=True)[0]
    return v.tolist()

def _cos(a, b) -> float:
    a = np.asarray(a, dtype="float32"); bn = np.asarray(b, dtype="float32")
    return float(np.dot(a, bn) / (np.linalg.norm(a) * np.linalg.norm(bn) + 1e-9))

def _sparse_sim(q: dict[str, float], d: dict[str, float]) -> float:
    if not q or not d:
        return 0.0
    dot = sum(q[k] * d.get(k, 0.0) for k in q)
    qn = math.sqrt(sum(v * v for v in q.values()))
    dn = math.sqrt(sum(v * v for v in d.values()))
    return float(dot / (qn * dn + 1e-9)) if dn else 0.0

def hybrid(query: str, top_k: int = TOP_K):
    \"\"\"Hybrid dense+sparse retrieval from the persisted store.

    Dense: Chroma ANN over the persisted collection (authoritative, feeds the
    grounding gate exactly as the backend's `search` does). Sparse: lexical
    weights from the sidecar, fused with the dense score — lifts exact
    medicine-name matches without breaking the backend contract.\"\"\"
    qv = _dense(query)
    res = coll2.query(query_embeddings=[qv], n_results=max(top_k, 50),
                      include=["documents", "metadatas", "distances"])
    qs = embed_sparse([query])[0]
    scored = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        dense_sim = 1.0 - float(dist)              # cosine distance -> similarity
        sp = _sparse_sim(qs, sparse_lookup.get(meta["doc_id"], {}))
        fused = 0.7 * dense_sim + 0.3 * sp
        scored.append((meta["doc_id"], meta.get("section", ""), meta.get("lang", ""),
                       max(0.0, min(1.0, dense_sim)), dense_sim, sp,
                       doc, meta.get("source", ""), fused))
    scored.sort(key=lambda t: t[8], reverse=True)
    return [r[:8] for r in scored[:top_k]]

q = "جرعة الباراسيتامول القصوى في اليوم؟"
for doc_id, section, lang, score, d, s, text, source in hybrid(q)[:3]:
    print(f"[{doc_id}] section={section} dense={d:.3f} sparse={s:.3f}")
    print("   ", text.replace("\\n", " / ")[:140])
""")

md("""\
## Extended-track OCR

The task: OCR **one handwritten-prescription photo** and make its extracted
text retrievable in the pipeline. Strategy (first hit wins, so the cell runs
top-to-bottom everywhere):

1. **Azure Document Intelligence** (`AZURE_OCR_ENDPOINT` + `AZURE_OCR_KEY`
   set) — the OCR engine that built this corpus (Microsoft OCR).
2. **Tesseract** (`pytesseract` + system binary) when available.
3. **Shipped OCR text** — the Microsoft-OCR output already stored in
   `rx/all_prescriptions_clean.json` for the same photo files.

Whichever path runs, the extracted text is added to the store as its own
chunk (provenance kept) and we **prove it is retrievable** by querying for a
drug name present in that photo.
""")

code("""\
# --- 6. OCR one handwritten-prescription photo --------------------------------
photos = sorted(DATA_ROOT.glob("rx/prescriptions/*.jpg"), key=lambda p: p.name)
photo = photos[0] if photos else None
assert photo is not None, "no prescription photo found in corpus"

ocr_text: str | None = None
which = ""

azure_endpoint = os.environ.get("AZURE_OCR_ENDPOINT", "")
azure_key = os.environ.get("AZURE_OCR_KEY", "")

def _try_azure(ph: Path) -> str | None:
    if not (azure_endpoint and azure_key):
        return None
    import base64
    import httpx
    url = azure_endpoint.rstrip("/") + "/formrecognizer/documentModels/prebuilt-layout:analyze?api-version=2023-07-31"
    b64 = base64.b64encode(ph.read_bytes()).decode()
    r = httpx.post(url, headers={"Ocp-Apim-Subscription-Key": azure_key,
                                 "Content-Type": "application/json"},
                   json={"base64Source": b64}, timeout=60)
    r.raise_for_status()
    op = r.headers.get("operation-location") or r.headers.get("Operation-Location")
    if not op:
        return None
    for _ in range(30):
        st = httpx.get(op, headers={"Ocp-Apim-Subscription-Key": azure_key}, timeout=30)
        j = st.json()
        if j.get("status") == "succeeded":
            return "\\n".join(l["content"] for p in j["analyzeResult"]["pages"]
                              for l in p.get("lines", []))
        if j.get("status") == "failed":
            return None
        time.sleep(1.0)
    return None

def _try_tesseract(ph: Path) -> str | None:
    try:
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(ph))
    except Exception:
        return None

def _ship_text(ph: Path) -> str | None:
    data = json.loads((DATA_ROOT / "rx" / "all_prescriptions_clean.json").read_text(encoding="utf-8"))
    entries = [e for e in data if e.get("filename") == ph.name]
    if not entries:
        return None
    meds = [m for m in entries[0].get("medications", []) if m.get("Medication_name")]
    return "; ".join(m["Medication_name"] for m in meds) if meds else None

if azure_endpoint and azure_key:
    try:
        ocr_text = _try_azure(photo)
        which = "azure"
    except Exception:
        ocr_text = None
if not ocr_text:
    try:
        ocr_text = _try_tesseract(photo)
        which = "tesseract"
    except Exception:
        ocr_text = None
if not ocr_text:
    ocr_text = _ship_text(photo)
    which = "shipped"

assert ocr_text, f"OCR produced no text for {photo}"
print(f"OCR via: {which} | photo: {photo.name}")
print("OCR text:", ocr_text[:400])

# --- index the OCR'd text as a retrievable chunk (provenance kept) ------------
import chromadb
client3 = chromadb.PersistentClient(path=str(CHROMA_DIR))
coll3 = client3.get_collection(COLLECTION)
doc_id_ocr = f"ma::ocr::{photo.name}::{hashlib.sha1(photo.name.encode()).hexdigest()[:8]}"
coll3.upsert(
    ids=[doc_id_ocr],
    embeddings=[_dense(ocr_text)],
    documents=[ocr_text],
    metadatas=[{"doc_id": doc_id_ocr, "collection": COLLECTION, "section": "ocr",
                "seq": 0, "chunk_id": f"{doc_id_ocr}/000000",
                "lang": "en", "source": f"rx/prescriptions/{photo.name}", "ocr": which}],
)

# append the OCR chunk to the sparse sidecar so retrieval covers it too
import re
from collections import Counter
sw = dict(Counter(re.findall(r"[a-zA-Z]+|[\\u0600-\\u06FF]+", ocr_text.lower())))
sf = CHROMA_DIR / "sparse_index.json"
si = json.loads(sf.read_text(encoding="utf-8"))
si.append({"doc_id": doc_id_ocr, "weights": sw})
sf.write_text(json.dumps(si, ensure_ascii=False), encoding="utf-8")
print("indexed OCR chunk:", doc_id_ocr)
""")

code("""\
# --- 7. prove the OCR text is RETRIEVABLE --------------------------------------
# Ask for one of the drug names the OCR picked up and require it in top-k.
first_drug = ocr_text.split(";")[0].strip()
token = first_drug.split()[0] if first_drug else "drug"
rows = hybrid(f"medication {token}", top_k=TOP_K)
hits = [r for r in rows if doc_id_ocr in r[0]]
print("query  :", f"medication {token}")
print("retrieved:", [(r[0], round(r[3], 3)) for r in rows[:3]])
print("OCR chunk retrieved in top-k:", bool(hits))
assert hits, "OCR text must be retrievable from the persisted store"
""")

code("""\
# --- 8. grounded, cited Arabic-primary answer (prompt contract) ----------------
# Prompt contract (§6.4): Arabic-primary output, Latin drug names verbatim,
# every answer cites the retrieved source. Without an LLM key the notebook
# renders the deterministic grounded answer (top cited chunk + refusal path),
# which proves the contract shape; with OLLAMA_URL it calls the real model.
import json as _json

def build_prompt(question: str, rows) -> str:
    ctx = [{"doc_id": r[0], "collection": COLLECTION, "section": r[1],
            "chunk_id": f"{r[0]}/000000", "quoted": r[6], "score": round(r[3], 3)} for r in rows]
    system = ("أنت مساعد طبي ردّك يستند فقط إلى المستندات أدناه. أجب بالعربية، "
              "احفظ أسماء الأدوية اللاتينية كما هي، واذكر المصدر بعد كل معلومة "
              "بصيغة [الجملة المقتبسة] إذا لم تجد المعلومة في المستندات فقل "
              "'لا أملك معلومات كافية' ولا تخترع.")
    block = "\\n".join(f"[quoted] {r[6]}" for r in rows)
    return f"{system}\\n\\nCONTEXT:\\n{_json.dumps(ctx, ensure_ascii=False)}\\n\\n{block}\\n\\nQUESTION: {question}\\nANSWER:"

def grounded_answer(question: str, rows) -> dict:
    if not rows or rows[0][3] < MIN_SCORE:
        return {"answer": "", "citations": [], "refused": True,
                "refuse_reason": f"best hit below grounding threshold {MIN_SCORE:.2f}"}
    prompt = build_prompt(question, rows)
    ollama_url = os.environ.get("OLLAMA_URL", "")
    ollama_model = os.environ.get("OLLAMA_MODEL", "")
    if ollama_url and ollama_model:
        import httpx
        r = httpx.post(f"{ollama_url}/api/generate",
                       json={"model": ollama_model, "prompt": prompt,
                             "stream": False, "temperature": 0.2}, timeout=120)
        r.raise_for_status()
        text = r.json()["response"].strip() or rows[0][6][:120]
    else:
        text = rows[0][6][:120]  # deterministic grounded fallback: top cited chunk
    return {"answer": text, "refused": False,
            "citations": [{"doc_id": r[0], "collection": COLLECTION, "section": r[1],
                           "quoted": r[6], "score": round(r[3], 3)} for r in rows]}

q = "ما الجرعة القصوى لباراسيتامول للبالغين؟"
out = grounded_answer(q, hybrid(q))
print("refused:", out["refused"])
print("refuse_reason:", out.get("refuse_reason"))
print("answer:", out["answer"][:200])
if out["citations"]:
    print("cite[0]:", out["citations"][0]["doc_id"], "| score", out["citations"][0]["score"])
""")

code("""\
# --- 9. evaluation over >=10 curated questions ----------------------------------
# Arabic + English, including an out-of-context question that MUST refuse
# (never invent). Every row records: question, lang, top hit, dense score,
# grounded?, and a PASS = (in-domain => grounded; out-of-domain => refused).
# In-domain questions are seeded from the corpus itself so retrieval has a
# known-good answer; the out-of-context pair guarantees the refusal path.
import random
rng = random.Random(SEED)
ar_rows = [r for r in corpus if r["lang"] == "ar"]
en_rows = [r for r in corpus if r["lang"] == "en" and r["split"] == "medquad"]
sampled = [(r["lang"], r["question"].strip(), True) for r in
           rng.sample(ar_rows, 6) + rng.sample(en_rows, 4)]
OUT_OF_CONTEXT = [
    ("ar", "كيف أصلح إطار الدراجة المثقوب؟", False),
    ("en", "How do I fix a flat bicycle tyre?", False),
]
EVAL = sampled + OUT_OF_CONTEXT
print("eval questions:", len(EVAL))

def run_eval() -> list[dict]:
    table = []
    for lang, question, in_domain in EVAL:
        rows = hybrid(question, top_k=TOP_K)
        out = grounded_answer(question, rows)
        top = rows[0] if rows else None
        grounded = not out["refused"] and len(out["citations"]) > 0
        if in_domain:
            passed = grounded
        else:
            passed = out["refused"]
        table.append({
            "lang": lang, "question": question, "in_domain": in_domain,
            "top_doc": top[0] if top else "", "dense": round(top[3], 3) if top else 0.0,
            "grounded": grounded, "refused": out["refused"], "passed": passed,
        })
    return table

import pandas as pd
df = pd.DataFrame(run_eval())
df.to_csv(str(OUT_DIR / "eval_results.csv"), index=False)
print(df.to_string(index=False))
print("PASS:", int(df["passed"].sum()), "/", len(df))
""")

code("""\
# --- 10. export the persisted store (what the backend loads at startup) --------
# The backend's VectorStore(persist_dir=...) opens CHROMA_DIR directly and
# queries collection 'medical_docs'. Summarise + write a manifest.
manifest = {
    "collection": COLLECTION,
    "model": MODEL_NAME,
    "chunks": coll.count(),
    "vector_dim": 1024,
    "hnsw_space": "cosine",
    "metadata_keys": ["doc_id", "collection", "section", "seq", "chunk_id", "lang", "source"],
    "exported_by": "kaggle_rag_pipeline.ipynb",
    "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
}
(CHROMA_DIR / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("== EXPORT ==")
print("store dir :", CHROMA_DIR)
print("chunks    :", coll.count())
for p in sorted(CHROMA_DIR.iterdir()):
    print(f"  {p.name:<22} {p.stat().st_size/1024:.1f} KiB")
print()
print("On Kaggle, download /kaggle/working/vector_store/ and place it at")
print("REPO_ROOT/data/vector_store/ so the backend serves it (it loads")
print("repo-root data/vector_store/chroma at startup).")
""")


# ------------------------------------------------------------------------------
def build_notebook() -> nbformat.NotebookNode:
    cells = [nbformat.v4.new_markdown_cell(src) if kind == MD else nbformat.v4.new_code_cell(src)
             for kind, src in CELLS]
    nb = nbformat.v4.new_notebook(
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"accelerator": "GPU", "isInternetEnabled": True,
                       "sourceType": "notebook", "language": "python3"},
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