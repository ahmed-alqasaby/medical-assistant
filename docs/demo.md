# M6 — Live demo script, recording instructions, and presentation outline

The final deliverable: a recorded video of the full arc —
**question → API → retrieval → LLM → grounded cited answer on screen** —
plus the closing presentation.

## 1. Before you record

```bash
# (optional, exact) rebuild the demo store from a fresh clone
.venv/bin/python scripts/build_store.py

# terminal A — API in demo mode (no GPU, no Ollama needed for the walk)
export EMBED_MODEL=TEST
.venv/bin/uvicorn backend.app.main:app --app-dir . --port 8000
# verify: curl http://localhost:8000/health
#   -> {"status":"ok","store":{"loaded":true,"collection":"medical_docs"}}

# terminal B — chat UI
export BACKEND_URL=http://localhost:8000
.venv/bin/streamlit run frontend/app.py
# open http://localhost:8501
```

Capture with your desktop recorder (OBS recommended). If only a terminal is
available, `ffmpeg` can record the window:

```bash
ffmpeg -f x11grab -i :0 -r 24 demo.mp4   # whole screen; crop as needed
```

## 2. Demo script (≈3 min)

1. **Open the app** — show the chat input and the sidebar health check
   (store loaded, chunk count, collection name).
2. **Arabic question** — type:
   `ما هي أعراض السالمونيلا في الدم وهل هي خطيرة؟`
   - Observe the loading state, then the answer.
   - **Expand the citations** and call out: doc-id, section, match score,
     quoted source text — the answer is *grounded*, never invented.
3. **Second in-domain question** (English) — `What is the outlook for
   childhood ependymoma?` — shows the pipeline is bilingual, same cited flow.
4. **Out-of-context / low-context question** — e.g. `هل القطط تطير؟` — show
   the trust gate's refusal path (cited-answer guarantee; the gate refuses
   rather than free-invent).
5. **Under the hood (~60s)** — pane-in the backend terminal: curl `GET /health`,
   then `POST /query` with `jq` to show the JSON contract (answer/citations/
   refuse). Then pane-in `scripts/build_store.py` output to show the *index*
   step that feeds the whole chain.

## 3. Recording checklist

- [ ] Terminal focused, README open in background (first screen)?
- [ ] Loading spinner visible on the Arabic question (M4 acceptance).
- [ ] Citations **expanded on screen** for at least one answer.
- [ ] `POST /query` JSON visible with citations array (M6 "API" beat).
- [ ] Camera intro ≤10s, title card: *Medical Assistant — grounded Q&A*.
- [ ] Export H.264 MP4, 1080p, ≤90 MB (Kaggle/GradCAP-friendly).

## 4. Presentation outline (final talk)

1. **Problem** — doctors need grounded answers mid-consult; LLM free-invention
   is unacceptable in medicine. Trust gate, not confidence.
2. **Stack in one line** — Chroma (cosine) + bge-m3 + Ollama behind a FastAPI
   seam, a Streamlit chat, and a Kaggle-orchestrated pipeline.
3. **Demo live** — the 5-step script above.
4. **Safety** — every answer cites retrieved evidence; below-threshold and
   out-of-domain inputs are refused; generations happen only over store data
   (`backend/app/generation.py`, grounding gate).
5. **Scale / next steps** — Kaggle full-corpus store, sparse hybrid, ASR +
   OCR session capture, patient memory (v2; see `CONTEXT.md`).