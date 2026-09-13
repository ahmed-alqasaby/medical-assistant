# Frontend — practice chat UI (M4)

Streamlit app that talks to the FastAPI backend (`POST /query`) and renders
the grounded answer **with its citations visible** — doc-id, section, score,
and the quoted source text.

## Run (needs the backend up)

```bash
export BACKEND_URL=http://localhost:8000     # backend URL via env ONLY (never hard-coded)
streamlit run app.py                         # default port 8501 (backend CORS allows it)
```

Backend            → `cd .. && uvicorn backend.app.main:app --port 8000` (or the prod stack)
`BACKEND_URL` unset → the app shows a visible config error instead of failing silently.

## What the UI guarantees

- Question → **cited answer**: citations rendered on screen, never hidden.
- **Loading state**: spinner shown while the backend answers.
- **Error state**: dead/timeout/5xx backend → a visible error message, never a silent hang.
- Every failure is coerced to a typed `BackendError` in `client.py` — the test seam.

## Tests

```bash
python -m pytest tests -q                 # from back/ or repo root; all 5 client tests
```

The client tests run against the **real** FastAPI seam app (fixture `build_app(store, llm)`)
over an ASGI transport — no live server, no bge-m3, no Ollama.