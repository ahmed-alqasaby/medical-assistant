# medical-assistant — System Design (v1)

*Status: ready-for-agent. Synthesized from the wayfinder map and locked research decisions (see `research/*.md`). Terms follow `CONTEXT.md`; the four research decision docs are normative where this spec compresses them.*
*rev. 2026-09-13 — review-pass refinements incorporated: CAG session-primary hybrid, chunk-enrichment + retrieval-index choices, generation-model candidates, tool-trace audit.*
*rev. 2026-09-13 (2) — risk register added (§9) + parameter ledger (`research/parameter-ledger.md`): every tunable is `[prior]` and unmeasured until its eval lands; RAPTOR summary tier explicitly unimplemented-by-design.*
*rev. 2026-09-13 (3) — tool contract pinned (§6.4): single `retrieve_history` tool, ≤2 calls, structural enforcement, citable result shape; doc-id registry + FileStore linkage (§3); `…/records/{doc_id}/artifact` endpoint (§7) so every citation resolves to the real file.*

---

## Milestone: Graduation Project — the medical assistant's first release

There is **one product** in this repo: the per-patient medical memory assistant of §1–§9. The graduation milestone is its **first release**, conformed to the Level 2 Summer Training / Graduation Project assignment while *still being the same idea* — not a different product. The assignment's notebook/embedding/retrieval work IS spec §6; the FastAPI backend IS the §7 ask-surface spine; the frontend IS the practice chat UI. ARCH-* tickets (§1–§9) deepen this same spine to the full v1 product after submission.

**Assignment** (student guide): build a complete AI product, raw documents → deployed RAG web app, on GitHub. **6-day build / 4-day deadline**. **Track: Extended** — Core RAG pipeline + a CV component. Our Microsoft OCR prescription-photo path satisfies the CV component and IS the seed of the ARCH-7 prescription route.

The build reuses the research decisions directly: bge-m3 + typed collections + turn-level chunking discipline (§6), the citation-resolves-to-a-real-source contract (§5), Arabic-primary (§5), index/query normalization (§6), Azure DI / Microsoft OCR for the handwritten-prescription images (ARCH-7 precursor). What the milestone does **not** implement yet (lands in ARCH-*): per-patient memory + grants (trust gate), CAG session-primary hybrid, multi-collection typed indexes as a service, the `retrieve_history` tool contract, diarized session audio, on-prem topology. The paper/pipeline shape stays: notebook → persisted vector store → FastAPI → Streamlit → GitHub.

**Deliverables checklist** (grading-driven; the grounding/citation contract IS the #1 pitfall guard — *"answers from the LLM's own knowledge instead of the retrieved context"* loses points, and this spec was built to forbid exactly that):

- [ ] `notebooks/rag_pipeline.ipynb` — runs top-to-bottom (Kernel → Restart & Run All), chunking, embeddings, retrieval testing, evaluation results table
- [ ] `backend/` — FastAPI with `GET /health` + `POST /query`, `.env.example`, pinned `requirements.txt`, passing pytest (happy + 422)
- [ ] `frontend/` — chat UI (Streamlit/Gradio) showing the cited answer; backend URL from env var, never hard-coded; loading + error states
- [ ] Persisted vector store produced by the notebook, served by the backend (no rebuild at request time)
- [ ] Root README.md a stranger can clone and run (overview, architecture, stack, setup, API reference, evaluation, screenshots)
- [ ] Public GitHub repo, clean history (no `.venv`, `.env`, corpus dump, oversized store)
- [ ] End-to-end demo: question → API → retrieval → LLM → grounded cited answer on screen
- [ ] Live demo + recorded video (final presentation)

**Phase map** (assignment → tickets on the tracker, M-*):

| Phase | Assignment says | Ticket |
|---|---|---|
| P0/1 | Env + domain + data collection (medical corpus; Extended: rx-image set) | M1 |
| P2 | Notebook: load, chunk, embed, Chroma, retrieve, prompt, eval (≥10 questions + table), export | M2 |
| P3 | FastAPI backend: `/health`, `POST /query`, startup load, CORS, ≥2 tests | M3 |
| P4 | Frontend (Streamlit/Gradio): chat, citations, env-driven URL, loading/error | M4 |
| P5 | Publish: `.gitignore`, public repo, README | M5 |
| Presentation | Live demo + recorded video | M6 |

---

## Problem Statement

Doctor–patient sessions happen **in person** in Egyptian/GCC grassroots clinics. The content of a session — the conversation, the handwritten prescription (روشتة), the lab results — lives on paper and in the doctor's memory. There is no per-patient digital record a doctor can read before a consult, nothing to interrogate during one, and nothing the patient can carry across practices. A telehealth platform cannot fix this: the session has no platform channel at all — it is a room and a paper ticket.

The consequence is that medical history is re-explained at every visit, prescriptions are handwritten and legible only to their author, and a patient changing clinics restarts from zero. The project exists to build the missing layer: a privacy-first per-patient memory with a machine over it that reads, retrieves, and explains.

## Solution

A **patient memory** middle layer, v1 scope:

- **Capture, automatically and in-room**: a diarized **turn-level transcript** `(speaker, start, end)` of the **session recording**; a **confirmed prescription** captured from the handwritten ticket; **lab results** as structured text.
- **Memory**: per-patient, patient-owned, **patient-granted and revocable** via the **trust gate**. Only **confirmed data** ever enters memory as fact — unconfirmed OCR drafts and ASR text are pipeline state, never retrievable as fact.
- **Middle layer** over that memory: the **pre-visit brief** (doctor, async), the **live grounded Q&A** (doctor, real-time, cited), and the **plain-language lens** (patient).

Five modules: **ingestion** (diarize-first ASR + prescription HITL + lab results), the **patient memory store**, **retrieval** (structured + hybrid vectors + thin agentic orchestration), **generation** (the three surfaces), and the **trust gate** (authz + audit). Arabic-primary, English **required**. Cloud-designed for development with **synthetic data only**; a public deployment is **on-prem capable** so real patient data never leaves the practice's control.

The entire v1 system is **tested through one seam**: the middle-layer generation contract (see Testing Decisions).

---

## User Stories

**Patient**

1. As a patient, I want my in-room session captured automatically into my memory, so that I don't re-explain my history at every visit.
2. As a patient, I want to approve, at check-in, which connected practice may see my memory, so that my record stays private by default.
3. As a patient, I want access revocable at any time and effective immediately, so that a practice I no longer trust loses access now.
4. As a patient, I want my memory to follow me when I move to another connected practice, so that I don't start from zero.
5. As a patient, I want to see a plain-language explanation of my own records and prescriptions, so that they make sense to me.
6. As a patient, I want to know what a doctor saw of my memory, so that access is transparent rather than assumed.
7. As a patient, I want the assistant to answer only from data I granted access to, never from invention, so that I can trust the machine.
8. As a patient, I want my session recording kept alongside the transcript, so that anything the machine says can be checked against reality.
9. As a patient, I want my memory used for my own care only — never cross-patient analytics or profiling.
10. As a patient, I want unconfirmed prescription drafts never surfaced to me or anyone as fact, so that I'm never told I'm taking a drug I'm not.
11. As a patient, I want my English drug names findable whether written in Latin or Arabic script, so that my record is searchable either way.

**Doctor**

12. As a doctor, I want a pre-visit brief synthesized from the patient's entire granted memory, so that I walk into the consult prepared.
13. As a doctor, I want the brief to lead with the patient's current confirmed drug list, so that I can spot interactions and contraindications.
14. As a doctor, I want to ask live questions against the patient's memory during the consult, so that I'm not paging through paper files.
15. As a doctor, I want every answer to carry citations to the exact turn `(speaker, timestamp)` it came from, so that I can verify the machine.
16. As a doctor, I want the transcript labeled doctor vs patient, so that I can see who said what at a glance.
17. As a doctor, I want a clean, explicit refusal when a question falls outside the patient's grant, so that refusal isn't mistaken for a system failure.
18. As a doctor, I want Arabic-primary output with Latin-script drug names preserved, so that I can act on it with pharmacy and systems alike.
19. As a doctor, I want to free-text search a patient's turns and records, so that I can pull up a specific episode without reading everything.
20. As a doctor, I want the prescription the memory records to be what I actually prescribed, so that the brief and Q&A never contradict the ticket.

**Practice staff**

21. As the pharmacy staff, I want to confirm or correct the OCR prescription draft on a short screen before it enters memory, so that a misread dose or drug never becomes a fact.
22. As the pharmacy staff, I want formulary autocomplete to catch drug-name and dot-level OCR errors, so that corrections are quick and guided.
23. As the practice reception, I want access requests to surface at check-in for the patient to approve, so that granting is part of the visit, not paperwork.

**Operator**

24. As the operator, I want development to run on synthetic data only, so that no real patient data touches the cloud.
25. As the operator, I want the entire runtime swappable to on-prem for public deployment, so that real PHI stays in the practice's control.
26. As the operator, I want the trust gate to be a service, not a frontend concern, so that no surface or API can bypass authorization.
27. As the operator, I want an audit of every retrieval a doctor performed (what, when, from which turns), so that the trust gate is provable.
28. As the operator, I want an accuracy gate before any model is relied on in production — WER on real session audio, HTR on real tickets, retrieval on a code-switch eval set — so that every published claim is measured on our data.
29. As the operator, I want lab results recorded as structured text, so that they are directly queryable without free-text noise.
30. As the operator, I want the v1 module list to contain no vision component, so that scan image understanding is honestly deferred to a future version.

**Middle layer (the assistant itself)**

31. As the assistant, I want every claim to resolve to a retrieved record before I answer, so that grounding is a property of the pipeline, not a hope.
32. As the assistant, I want my tools scoped per patient and per grant, so that cross-patient retrieval is structurally impossible, not just discouraged.
33. As the assistant, I want to answer in the patient's and doctor's primary language (Arabic) while preserving Latin drug names, so that bilingual clinical reality is represented.

---

## Implementation Decisions

### 1. Module inventory and scope

| Module | Responsibility | Locked by |
|---|---|---|
| **Ingestion** | session recording → turns (diarize-first ASR); prescription photo → confirmed prescription (HTR + HITL); lab result text → structured record | ASR decision (Cohere primary); prescription decision (OCR-primary + HITL + formulary) |
| **Patient memory store** | per-patient relational store; the only place facts live; every record traceable to a session. This is the lightweight **MAG (Memory-Augmented Generation)** layer — the "graph without a graph DB" (§6.1) | data model (§3) |
| **Retrieval** | structured queries + hybrid vectors + reranker; agentic tool layer; per-patient scoped | embeddings decision (bge-m3) |
| **Generation (middle layer)** | the three surfaces; grounding + citation + refusal contracts | §5 |
| **Trust gate** | access grants, revocation, audit, identity key, roster | §4 |

No vision module exists in v1. Scans are stored **outside the retrieval memory** and are not captioned or embedded.

### 2. Ingestion pipeline

**Audio → turns (diarize-first, locked):**

1. **VAD / noise gate** — mandatory; the ASR model transcribes silences eagerly, so the gate is a correctness feature, not a nicety.
2. **Diarization** — pyannote `speaker-diarization-community-1`, `min/max_speakers = 2`, on-prem. License verified commercial-safe (CC-BY-4.0; gated access).
3. **Per-turn transcription** — `CohereLabs/cohere-transcribe-arabic-07-2026` (Apache-2.0), `language="ar"` default, `"en"` fallback on English-dominant turns (the model has no language auto-detection — always specified). Long turns auto-split; map chunk indices back to the turn.
4. **Role pass (A/B → doctor/patient)** — "doctor opens and holds longer turns"; validated against ~10 real sessions before downstream retrieval trusts the labels.
5. **Output shape** — the turn below; this is the RAG chunk boundary and the audit unit.

**Capture SOP:** baseline = single in-room mic + software diarization; **target = dual-lapel two-channel** (ch1 = doctor, ch2 = patient) where diarization collapses to channel mapping (DER ≈ 0). Do **not** let the diarizer downmix stereo.

**Prescription photo → confirmed prescription (locked chain):**

1. **Capture conditioning** — guided photo SOP (flat, well-lit, full frame, no shadows): the biggest free quality lever on realistic hand-written tickets.
2. **HTR engine** — Azure Document Intelligence (the only hosted API claiming Arabic handwriting) **or** a fine-tuned trOCR-class model; **the engine choice is held open behind our own eval** (no public Arabic medical-prescription dataset or benchmark exists — v1 concept build IS the benchmark).
3. **Formulary-constrained post-processing** — every candidate token resolves against the clinic's medicine list (Latin names + Arabic transliterations) plus digit normalization (Western `8` = Eastern `٨`). Free-form spell-correction is **banned** (it mutilates out-of-vocabulary drug names); post-OCR manual correction is encouraged.
4. **Slot extraction** — drug / dose / frequency / duration with per-slot confidence.
5. **Human confirmation gate — non-negotiable** — the pharmacy/staff confirm or correct the draft in a short screen; **only the confirmed prescription enters the memory**. Unconfirmed slots are pipeline state, never retrievable as fact.

**Lab results:** text-based (typed entry or printed-sheet OCR), structured record per session.

**Accuracy-gate milestones (before any model is trusted in production):** ~20–30 real in-room sessions measured for WER (every published ASR number is from a proxy corpus); N real prescriptions (photo + keyed ground truth) before committing the HTR engine; role labels hand-validated on ~10 sessions.

### 3. Patient memory data model (schema shape — from the data-model ticket, synthesized here)

Entities per `CONTEXT.md`; relations are explicit typed edges (session → turns / confirmed prescriptions / lab results → patient) so a knowledge graph is derivable later **without re-modeling** (GraphRAG-deferred requirement).

```
Patient            { id: identity-key, practices: ConnectedPractice[] }   // id stable across practices
ConnectedPractice  { id, name, status }
MedicalSession     { id, patient, practice, occurred_at, recording_ref, status }
Turn               { patient, session, seq, speaker: "doctor"|"patient",
                     start_ms, end_ms, text, lang: "ar"|"en", asr_confidence }
ConfirmedPrescription { id, session, drug, dose, frequency, duration, instructions,
                     confirmed_by, confirmed_at, photo_ref }        // ONLY post-HITL; photo_ref = provenance
LabResult          { id, session, panel, value, unit, ref_range, recorded_at, sheet_ref }
AccessGrant        { patient, practice, granted_at, granted_from, revoked_at?, scope }
AuditEvent         { patient, actor, action, turn_ref?, frame?, tool_call_seq, occurred_at }
                     // frame = the context frame served (e.g. whole session S); tool_call_seq = ordered
                     // tool invocations for that actor/request (agentic paths are non-deterministic —
                     // the audit must be provable, so the trace is recorded, not reconstructed)
ScanRecord         { id, session, stored_ref, source_turn? }   // metadata ONLY; never embedded
                     // source_turn = FK to the transcript turn that discussed the scan → scans are
                     // findable transitively through the turn, never interpreted by a model in v1
```

Rules enforced by the store, not by convention:

- Only **ConfirmedPrescription** rows are created from the prescription pipeline; the OCR draft never occupies a memory slot.
- Every Turn, ConfirmedPrescription, and LabResult is reachable from exactly one MedicalSession, which is reachable from exactly one Patient.
- An AccessGrant's scope is expressed over this schema (which sessions / fields), never as free text.
- No global or cross-patient query surface exists in the storage layer.
- **Scans are metadata-only records.** A ScanRecord holds the stored file reference and the source turn that discussed it; it is never embedded, captioned, or retrieved as content. The prescription **photo** behaves the same — audit/provenance link, never embedded (mirror of the unconfirmed-draft rule).
- ConfirmedPrescription and LabResult rows additionally carry a **normalized text template** (§6.2) so structured records participate in retrieval both as fields and as indexed text.
- **Doc-id registry:** every retrievable row exposes a stable composite id `patient/session/type/seq` (e.g. `pat_7/sess_12/turn/034`) — it is the vector-index id, the citation key, and the artifact-resolver key, all one key. The embedded copy is **disposable**: **never serve content from the index**; the canonical row is re-read at answer time, and its original artifact (recording WAV, rx photo, lab sheet, scan file) lives in a per-premises **FileStore** referenced by `recording_ref` / `photo_ref` / `sheet_ref` / `stored_ref`. Blobs are never embedded.

### 4. The trust gate

- **Request → approve at check-in:** a connected practice requests access; the patient approves at check-in before the consult. Access is earned, not automatic.
- **Revocation:** patient-initiated, effective immediately; the grant row is closed, and retrieval + generation paths re-check it on every request (no cached authorization).
- **Audit:** every generation request a doctor makes is an AuditEvent. The event records the **context frame served** (for the CAG path: the whole session loaded; for the RAG path: the turn list returned by each tool call) plus the **ordered tool-call sequence** — agentic paths are non-deterministic, so the trace is logged at request time, not reconstructed later. This is the unit the trust gate is provable by.
- **Identity key:** the Patient id is the single key linking a patient across connected practices; roster (federation vs single-operator) is flagged as a sub-decision under the gate/shared design, but is not allowed to change the schema above.
- **Deliberately out of scope:** voiceprint/biometric identity is a conscious future decision, not an ASR byproduct — the v1 system never derives identity signals from session audio.

### 5. Generation (the middle layer contracts)

Three surfaces over the same store, three distinct contracts:

1. **Pre-visit brief** (`POST /v1/patients/{id}/brief`) — async, over the patient's **full granted memory**. v1 synthesis is **direct long-context** over all granted turns/records (both generation candidates hold ≥128K context; a full patient history fits); a turn-cluster summary tier is the documented **scale-escape-hatch** only when a patient's granted history exceeds the generation context — **provisional, unimplemented-by-design** (§9/R3). The brief carries a source list that resolves to the underlying records — a summary is never cited *instead of* its source turns. Stylized around the confirmed drug list.
2. **Live grounded Q&A** (`POST /v1/patients/{id}/ask`) — real-time, **session-primary (CAG)**: the **active session's** turns are loaded **directly into the prompt** as a deterministic, cached context frame (no retrieval-miss risk within the session; zero hope that ranking surfaces it); only a question that needs something **outside today's session** routes to the retrieval layer (§6). Every claim still carries a citation resolving to a real turn in the frame.
3. **Plain-language lens** (`POST /v1/patients/{id}/explain`) — same store, simple-terms generation for the patient. Defaults to full granted memory; accepts a session context to reuse the CAG frame for "what happened today" questions. Never uses the summary tier (must stay grounded to records).

Cross-cutting contracts:

- **Confirmed-data-only grounding:** retrievable facts = confirmed prescriptions, turn-level transcript, lab results. An OCR draft prescription is **never** surfaced by any surface.
- **Citation format:** `(doc-id, speaker, start_ms, end_ms)` with the quoted turn text; the seam asserts this shape (see Testing). The CAG frame satisfies this by construction — the cited turn is literally in the prompt.
- **Refusal is a contract:** a question outside the patient's grant (or off-record scope) produces a clean refusal, typed as a refusal — not an invented answer, not an error, not a guess.
- **Session-primary router:** the `ask` surface holds one tool — *history* retrieval. In-session questions never invoke it; the frame is context, not a tool, which is where deterministic behavior is bought back from the agent. The router's in-frame / needs-history split is part of what the audit logs as `tool_call_seq`. The CAG frame-packing threshold (how a session is packed near context limits) is a `[prior]` ledger item (§9/R1).
- **Language:** Arabic-primary output; English permitted; Latin-script drug names preserved verbatim within Arabic output.
- **Summary tier (RAPTOR-style) is brief-only.** Summarized history is generated text and is **never** an input to live Q&A or the lens — both remain bound to the citation-resolves-to-a-real-turn contract.

### 6. Retrieval (layered)

**Chunking (the judgment against our pipeline):**

- **Turn-level is locked and correct** — dialogue chunks are semantic units (`speaker, start, end`), not token windows; this is the chunk boundary *and* the audit unit, so retrieval and audit share one shape.
- **Short-turn context loss is real** (patients answer "نعم", "تمام") → index-time **contextual enrichment**: each turn's index entry is prefixed with a short, generation-LLM-written context (same normalization contract; the prefix is **index-only**, never returned as answer content or citation text). Implemented with our own generation model — no license or extra-encoder cost. **Late chunking (Jina-style) is rejected for v1**: the canonical implementation rides `jina-embeddings-v3` (CC-BY-NC — the same license trap the generation model §7 avoids) and inserts a second encoder into the loop; bge-m3 has no native late-chunk path — its strength is hybrid sparse+dense within its 8192-token window. Revisit only if the embedder changes. Enrichment-on vs. off is an eval variant on the code-switch set (§ Testing); the prefix budget is a `[prior]` ledger item (§9/R1).
- **Hierarchical retrieval is folded into the CAG session-primary hybrid** (§5): "match at turn level, expand to the current session" collapses into "the current session is already in the prompt (CAG); only history questions route to vectors." Same access pattern, one mechanism, no extra index.

1. **Structured patient memory (MAG layer)** — relational fields (sessions, confirmed prescriptions, lab results, access grants) = "a graph without a graph DB"; relations explicit for future derivation.
2. **Hybrid vector RAG — typed collections, not one flat index:**
   - **`turns`** — bge-m3 dense+sparse over turn text, enriched per §6-chunking; top-k + `bge-reranker-v2-m3` cross-encoder.
   - **`lab_results`** and **`confirmed_prescriptions`** — the same embeddings over their **normalized text templates** (e.g. a prescription renders as a drug/dose/frequency/duration sentence), so structured records are hit by hybrid retrieval *and* by direct field queries. Templating follows the §6.3 normalization contract.
   - **`scans`** — a **metadata-only** collection: never embedded, never captioned; found only transitively via the `source_turn` pointer when the discussing transcript turn is retrieved. Prescription **photos** are audit/provenance links, never embedded.
3. **Hybrid fusion** — bge-m3's three internal signals (dense / sparse / multi-vector) are **weighted by query type**, not uniformly: Latin-token queries (drug names) boost sparse; long-form semantic Arabic boosts dense; short queries use late-interaction. Cross-collection results fuse with **Reciprocal Rank Fusion** (default), A/B-tested against a simple weighted convex combination on our eval set before lock-in — fusion mode and the per-type weights are `[prior]` ledger items (§9/R1).
4. **Thin agentic orchestration — the tool contract.** Exactly one tool exists: `retrieve_history(query, collection?, top_k?)`, executed by service code, never by the model. The loop is `model → tool_call → service executes → result appended → model continues`; hard cap of **≤2 tool calls per request**; the ordered sequence (plus the served frame) is written to the AuditEvent at request time, never reconstructed. Enforcement is structural, not prompted:
   - the tool is deterministic code scoped per patient and per grant; every invocation re-checks the grant (no cached authz) — a revoked grant is a hard no-op;
   - it is **read-only** — it cannot write, and argument validation runs before execution;
   - it returns typed results with citation fields (`doc_id`, `collection`, `quoted`, `speaker`, `start_ms`, `end_ms`, `lang`, `asr_confidence`), never raw internals;
   - refusal outside the grant is a typed refusal, never a tool error.

**Normalization contract (non-negotiable):** identical preprocessing on the **index** and **query** sides — remove diacritics, unify alef variants (أ/إ/آ → ا), teh-marbuta (ة → ه), yaeh (ى → ي), strip tatweel/kashida. ASR output is undiacritized, so this is load-bearing for retrieval quality. Applied once for both the bge-m3 inputs and the template strings.

### 7. End-points and stack inventory

Public contract (all behind the trust gate):

| Endpoint | Surface | Notes |
|---|---|---|
| `POST /v1/access/request` | gate | practice requests access to a patient |
| `POST /v1/access/grant` · `revoke` | gate | patient approves / closes a grant |
| `POST /v1/patients/{id}/brief` | generation | pre-visit brief |
| `POST /v1/patients/{id}/ask` | generation | live grounded Q&A, citations |
| `POST /v1/patients/{id}/explain` | generation | plain-language lens |
| `GET /v1/patients/{id}/audit` | gate | what a doctor saw |
| `GET /v1/patients/{id}/records/{doc_id}/artifact` | gate | the real file behind a citation: turns → WAV snippet at `[start,end]`; rx → original photo; scan → stored file (not interpretable) |

Runtime picks (locked): Cohere Transcribe Arabic 07-2026 (transformers/vLLM, ≤8 GB GPU class), pyannote `community-1` (k=2), `bge-m3` + `bge-reranker-v2-m3` (sentence-transformers / TEI / vLLM), the prescription HTR engine **held behind our eval** (Azure DI or fine-tuned trOCR), the patient memory store = the relational schema of §3.

**Open decisions carried forward (flagged, not locked):**

- **Generation model** — **CAG-capable and RAG-faithful, Arabic-primary, on-prem capable, selectable.** Two eval candidates, both run on the code-switch eval set before lock-in, same discipline as the bge-m3/Swan comparison:
  - `Command R7B Arabic` (`CohereLabs/c4ai-command-r7b-12-2024`) — **quality ceiling**: 7B, 128K ctx, the model with direct Arabic **RAG-faithfulness** benchmarking (AfricaNLP 2025: instruction-following, RAG, contextual faithfulness). **License is CC-BY-NC** — usable for research/concept, but a public deployment requires a **separate commercial license from Cohere**.
  - `Falcon-H1-Arabic 7B Instruct` (TII, 2026-01) — **commercially usable out of the box**: 256K ctx, dialect coverage incl. Egyptian, `falcon-llm-license`. Same size class, no license gate for public deployment.
  - Locking rule: whichever candidate passes the eval on faithfulness + refusal + Arabic quality; if Command R7B wins, the lock includes the commercial license decision. Both ≥128K ctx → the CAG session frame (a 45–90 min consult ≈ 8–25K tokens) fits comfortably in either.
- The **roster model** — federation vs single-operator (sub-decision under the trust gate).
- **Scale targets** — patients, sessions/day, retention (feed storage sizing later).

### 8. Deployment topology

- **Development:** cloud, **synthetic data only**; the full stack runs identically to production (swappable-component topology).
- **Public deployment:** **on-prem for real data**; every locked component (Cohere ASR, pyannote, bge-m3, reranker) is on-prem capable by construction. The one license wrinkle is the generation model — see §7 — commute it before public go-live (commercial license for the CC-BY-NC candidate, or the Falcon candidate). Nothing in the v1 runtime requires a cloud API; the only cloud-touching pieces are the model-authorization gates that fetch gated weights at deploy time.
- **Data boundary:** real PHI never crosses the premises boundary in a public deployment; the boundary is enforced by topology, not by policy.
- **The trust gate is a service** — authorization lives between every surface and the store; no request reaches retrieval without a grant check.
- v2+ captioner layer (scans) must slot into this topology later **without re-architecting** — it is a new ingestion path over the same store, gated by the same trust gate.

---

## 9. Risk register & parameter ownership

The architecture is design-time reasoning. **No parameter has been measured against real data yet** — the accuracy-gate program (in-room WER, real-ticket HTR, role-label validation, the code-switch retrieval set) is the project's actual risk owner, not any architectural choice. Until a number is measured it is tagged `[prior — unmeasured]` in `research/parameter-ledger.md` and is **not a lock**.

- **R1 — Validated vs. measured.** Every tunable in the retrieval/chunking/generation stack is a reasonable prior, not a measured value: enrichment-prefix budget, enrichment on/off, top-k, reranker candidate window, fusion (RRF vs weighted-convex), per-query-type signal weights, CAG frame-packing threshold, generation context headroom. The parameter ledger owns each; no `[prior]` value may be treated as a specification until its eval test records a value.
- **R2 — Permissive-license scarcity for Arabic-specialized models.** This is the project's standing constraint, not a chain of four decisions. Every layer has hit it: Swan-Large ✗ (CC-BY-NC), jina-embeddings-v3 ✗ (CC-BY-NC), the generation model ⚠ (Command R7B Arabic carries a commercial-license gate; Falcon-H1-Arabic 7B does not), the HTR engine ⚠ (Azure DI is vendor-hosted; trOCR is the self-hosted fallback). The layers that won did so on license as much as quality (Cohere ASR Apache-2.0, bge-m3/reranker MIT). Standing pattern: joint **quality × license** selection with a per-layer candidate pool — eligibility is decided together, not licensed later during compliance review.
- **R3 — Provisional designs are unimplemented-by-design.** The RAPTOR-style **summary tier** (§5) is the only provisional design. **Trigger:** a measured patient whose granted history exceeds the generation context. **Rule:** no implementation until the eval demonstrates the trigger; if ever built, it stays brief-only and never feeds live Q&A or the lens.

---

## Testing Decisions

**One seam.** The entire feature is tested at the **middle-layer generation contract**: the `ask` / `brief` / `explain` surfaces taken as a black box over a fixture patient memory and fixture grants. Driving the feature through this seam exercises ingestion output, the store, retrieval, grounding, confirmed-data-only rules, citation shape, refusal behavior, and the trust gate end to end. Uniform/module tests below the seam (a diarization-VAD unit, a formulary-constraint unit, a grant-revocation unit) are acceptable internal tests; they are not where the feature is proven.

**What makes a good test (at the seam):**

- External behavior only: assert on answers, citations, refusals, language — never on retrieval internals, prompts, or chunking.
- A fixture patient memory is built from **synthetic** sessions: turns (doctor/patient, Arabic + English, Latin drug names), confirmed prescriptions, lab results, and **deliberately one unconfirmed OCR draft** — the draft is the observable trap that proves confirmed-data-only grounding.
- Citation assertions: every factual claim is backed by a `(doc-id, speaker, start_ms, end_ms)` citation that resolves to a real fixture turn.
- Refusal assertions: a question about a non-granted patient, or a non-granted field, yields the typed refusal — and never a fabricated answer.
- Language assertions: answers default Arabic-primary; Latin drug names pass through unchanged.
- **Normalization-invariance:** the same query in diacritized vs undiacritized / alef-variant forms retrieves the same evidence (proves the normalization contract).
- **Code-switch retrieval set:** the 50–100-query Arabic+English medical eval set (the only evidence for that slice — no public benchmark exists) is a first-class fixture used at this seam. It carries the layer-level A/B variants as scored assertions: contextual-enrichment on vs off, RRF vs weighted-convex fusion, and signal-weighting by query type.
- **Session-primary (CAG) assertions:** a purely in-session question is answered with **zero history tool invocations** — the `AuditEvent.tool_call_seq` for that request proves no history retrieval happened, and every citation resolves to a turn inside the session frame. A cross-session question (e.g. "what did he take in March") must route to history and cite real prior-session turns.
- **Audit assertions:** every ask emits an AuditEvent recording frame + tool_call_seq; a revoked grant produces no event with an upstream grant.

**Modules tested:** generation (at the seam), retrieval pipeline (behind the seam via the same fixtures), ingestion stages (unit-level), trust gate (unit-level: grant lifecycle + revocation immediacy + audit completeness), store (relational invariants: no facts without a session; no draft prescriptions).

**Prior art:** none — this is a greenfield repo; the seam-first convention is established by this spec, and the fixture-patient-memory pattern above is the prior art every future slice builds on.

---

## Out of Scope

- **Scan image understanding** (v2+); scans are stored outside retrieval memory, never captioned or embedded in v1.
- **GraphRAG** (deferred by decision); relations stay explicit so it can come later without re-modeling.
- **Cross-patient / global retrieval and analytics** — the storage layer has no such surface, by design.
- **Telehealth** streaming or platform integration — in-person sessions are the product.
- **Production compliance certification** — architecture stays privacy-first; certification is a future destination redraw, not a v1 deliverable.
- **Procedure-note ingestion** — v1 embeds transcript, confirmed prescriptions, and text-based lab results only.
- **Streaming ASR** and non-Arabic/English languages.
- **Voiceprint/biometric identity.**
- **Autonomous (no-HITL) prescription HTR.**

---

## Further Notes

- *The failure the trust gate exists to stop:* hallucinated ASR text quoted into a patient record. Mitigations compound — VAD gate, med-list handling, slot-review on prescriptions, confirmed-data-only grounding, per-claim citation — but the seam test is the proof they all hold together.
- Every locked number in this spec is from a **proxy corpus**; the accuracy-gate milestones are explicitly the missing-evaluation problem the concept build converts into evidence: in-room WER, real-ticket HTR, the code-switch retrieval set.
- Role labels (doctor vs patient) start as a heuristic and are **hand-validated on ~10 sessions** before retrieval consumes them; DER and WER are different quantities, and label correctness is verified by hand, not assumed.
- Capture hygiene (dual-lapel two-channel, prescription photo conditioning) is product-level quality, cheaper than any model change.
- Where a lock is held open (HTR engine, generation model, roster model), it is tagged in this spec; build sessions must not silently substitute a model free of those criteria. The generation-model choice carries a **license** dimension alongside quality — the eval scores and the deployment license gate are decided together, not separately (systemic: risk register §9/R2).
- The 2026-09-13 review-pass refined chunking/retrieval/generation without re-opening locked research decisions: turn-level stays; CAG session-primary removes retrieval risk where it was most likely to matter; the summary tier is confined to the async brief and is explicitly **unimplemented-by-design** (§9/R3); late chunking is rejected on the same license logic already applied to the generation model. Every parameter the pass touched is `[prior]` until the eval program (spec §9/R1, `research/parameter-ledger.md`) measures it.
- Next build session: build against this spec, test at the middle-layer seam, and use the fixture-patient-memory pattern from the start.
- **Milestone ↔ ARCH mapping:** tickets M1–M5 (grad release) and ARCH-1…ARCH-10 (full v1) are the **same product**. M2 is §6's retrieval backbone; M3 is the §7 ask-surface spine; M4 is the chat UI; the ARCH tickets deepen this spine — per-patient memory, trust gate, CAG, tool contract, diarized ingest, HITL prescription confirm (seeded by M2's Microsoft OCR), on-prem topology. No generic document assistant exists anywhere in the plan.