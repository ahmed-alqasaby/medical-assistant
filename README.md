# Medical Assistant (مساعد طبي)

A graduation build: an Arabic-primary medical assistant that answers
**grounded, cited** questions over real medical corpora, running end-to-end
from retrieval to generation.

- **Frontend** — Streamlit chat app (question → cited answer on screen).
- **Backend** — FastAPI RAG service (retrieval + grounding gate + generation).
- **Model pipeline** — a runnable Kaggle notebook that builds the vector store
  (bge-m3 dense + sparse) from the corpora and exports the store + eval.
- **Corpus** — Arabic medical Q&A (`ar/`), MedQuad English (`en/`), Egyptian
  prescription OCR images (`rx/`, `rx_ocr/`).

```
┌────────────┐   ┌────────────────────┐   ┌───────────────────────────┐
│  Streamlit │──▶│  FastAPI backend   │──▶│  Chroma store (cosine)    │
│  chat app  │   │  RAG + trust gate  │   │  bge-m3 dense + sparse    │
└────────────┘   │                    │   └───────────────────────────┘
                 │  Gate: citations   │   ┌───────────────────────────┐
                 │  from store only   │──▶│  Ollama (command-r7b)     │
                 └────────────────────┘   │  grounded cited answer   │
                                         └───────────────────────────┘
```

## Stack

| Layer | Tech |
| --- | --- |
| Chat UI | Streamlit |
| API | FastAPI + Uvicorn |
| Vector store | Chroma (hnsw:space = cosine), `medical_docs` collection |
| Embeddings | BAAI/bge-m3 (dense + sparse), deterministic embedder for CPU demo |
| Generation | Ollama (`command-r7b-arabic`), extractive fallback in demo mode |
| Data | Arabic medical QA, MedQuad (EN), Egyptian Rx OCR datasets |
| Tests | pytest (38) — seam-tested with deterministic doubles, never real models |

## Setup (clone → run)

Python 3.12. Commands from the repo root, using `uv` (or your venv of choice):

```bash
uv venv
uv pip install --python .venv/bin/python -r requirements.txt \
  -r backend/requirements.txt -r frontend/requirements.txt
```

### Quickstart (CPU-only, no model downloads)

1. Build a small demo store from the corpus (deterministic embedder —

   ~10k chunks, a few seconds):

   ```bash
   .venv/bin/python scripts/build_store.py
   ```

2. Start the API in demo mode:

   ```bash
   EMBED_MODEL=TEST .venv/bin/uvicorn backend.app.main:app --app-dir . --port 8000
   ```

3. Point the frontend at it and launch the chat:

   ```bash
   export BACKEND_URL=http://localhost:8000
   .venv/bin/streamlit run frontend/app.py
   ```

Open http://localhost:8501, ask a medical question, and get a **cited** answer.
The full stack runs with zero GPU and zero model downloads.

### Production path (bge-m3 + Ollama)

1. Run `notebooks/kaggle_rag_pipeline.ipynb` on Kaggle (full corpus, bge-m3)
   and download the exported Chroma store + `sparse_index.json` to
   `data/vector_store/chroma` (Kaggle creds in `~/.kaggle/access_token`,
   see `scripts/kaggle_connect.sh`).
2. Create a repo-root `.env` (see `backend/.env.example`) and start Ollama with
   the model:
   ```bash
   ollama pull command-r7b-arabic
   ```
3. Start backend and frontend as above (without `EMBED_MODEL=TEST`).

## API reference

`GET /health`

```json
{"status":"ok","store":{"loaded":true,"chunks":9986,"collection":"medical_docs"}}
```

`POST /query`  `{ "question": "...", "top_k": 5 }`

```json
{
  "answer": "...",
  "refuse": false,
  "refuse_reason": null,
  "citations": [
    {"doc_id":"ma::ar__train_csv::002039","section":"أمراض-الغدد-الصماء",
     "score":0.791,"text":"..."}
  ]
}
```

- Empty/whitespace `question` → `422`.
- Unanswerable/out-of-domain questions or a down generation model → `refuse`
  with a reason; the answer is **never** LLM free-invention.

| Env var | Default | Meaning |
| --- | --- | --- |
| `BACKEND_URL` | — (required) | Frontend → API base URL; never hard-coded |
| `VECTOR_STORE_PATH` | `data/vector_store/chroma` | Chroma persist dir |
| `EMBED_MODEL` | `BAAI/bge-m3` | `TEST`/`` → deterministic demo embedder |
| `OLLAMA_URL` / `OLLAMA_MODEL` | `http://localhost:11434` / `command-r7b-arabic` | Generation |
| `MIN_SCORE` | `0.35` | Grounding-gate citation threshold |

## Evaluation

The notebook evaluates the pipeline over ≥10 curated questions — in-domain
Arabic and English samples plus out-of-context refusals — writing
`eval_results.csv` (grounding, refusal behavior, citation scores). Results are
exported with the store and reported in `docs/adr/` / the M6 presentation.

## Tests

```bash
.venv/bin/python -m pytest -q      # 38 tests: env contract, corpus, backend API, frontend client
```

## Screenshots

Captured from the running app during the M6 demo (see `docs/screenshots/` and
`scripts/demo.md`); they show the question → API → retrieval → grounded-cited
answer arc in the chat UI.

## Docs

- `spec/system-design.md` — v1 system design & decision order
- `docs/adr/` — architecture decision records
- `CONTEXT.md` — domain glossary