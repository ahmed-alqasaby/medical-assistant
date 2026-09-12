# medical-assistant — System Design (v1)

*Status: ready-for-agent. Synthesized from the wayfinder map and locked research decisions (see `research/*.md`). Terms follow `CONTEXT.md`; the four research decision docs are normative where this spec compresses them.*

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
| **Patient memory store** | per-patient relational store; the only place facts live; every record traceable to a session | data model (§3) |
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
                     confirmed_by, confirmed_at }                          // ONLY post-HITL
LabResult          { id, session, panel, value, unit, ref_range, recorded_at }
AccessGrant        { patient, practice, granted_at, granted_from, revoked_at?, scope }
AuditEvent         { patient, actor, action, turn_ref?, occurred_at }     // what a doctor saw
```

Rules enforced by the store, not by convention:

- Only **ConfirmedPrescription** rows are created from the prescription pipeline; the OCR draft never occupies a memory slot.
- Every Turn, ConfirmedPrescription, and LabResult is reachable from exactly one MedicalSession, which is reachable from exactly one Patient.
- An AccessGrant's scope is expressed over this schema (which sessions / fields), never as free text.
- No global or cross-patient query surface exists in the storage layer.

### 4. The trust gate

- **Request → approve at check-in:** a connected practice requests access; the patient approves at check-in before the consult. Access is earned, not automatic.
- **Revocation:** patient-initiated, effective immediately; the grant row is closed, and retrieval + generation paths re-check it on every request (no cached authorization).
- **Audit:** every retrieval a doctor performed is an AuditEvent tied to the turn list actually served; this is the unit the trust gate is provable by.
- **Identity key:** the Patient id is the single key linking a patient across connected practices; roster (federation vs single-operator) is flagged as a sub-decision under the gate/shared design, but is not allowed to change the schema above.
- **Deliberately out of scope:** voiceprint/biometric identity is a conscious future decision, not an ASR byproduct — the v1 system never derives identity signals from session audio.

### 5. Generation (the middle layer contracts)

Three surfaces over the same store, three distinct contracts:

1. **Pre-visit brief** (`POST /v1/patients/{id}/brief`) — async, over the patient's **full granted memory**; output = the brief + the source list it was built from. Stylized around the confirmed drug list.
2. **Live grounded Q&A** (`POST /v1/patients/{id}/ask`) — real-time; answer **only from retrieved evidence**; every claim carries a citation; no-invention is a pipeline property.
3. **Plain-language lens** (`POST /v1/patients/{id}/explain`) — same store, simple-terms generation for the patient.

Cross-cutting contracts:

- **Confirmed-data-only grounding:** retrievable facts = confirmed prescriptions, turn-level transcript, lab results. An OCR draft prescription is **never** surfaced by any surface.
- **Citation format:** `(doc-id, speaker, start_ms, end_ms)` with the quoted turn text; the seam asserts this shape (see Testing).
- **Refusal is a contract:** a question outside the patient's grant (or off-record scope) produces a clean refusal, typed as a refusal — not an invented answer, not an error, not a guess.
- **Language:** Arabic-primary output; English permitted; Latin-script drug names preserved verbatim within Arabic output.

### 6. Retrieval (layered)

1. **Structured patient memory** — relational fields (sessions, confirmed prescriptions, lab results, access grants) = "a graph without a graph DB"; relations explicit for future derivation.
2. **Hybrid vector RAG** — `bge-m3` dense+sparse over **turn-level** chunks (chunks are diarized turns, not arbitrary splits), top-k + `bge-reranker-v2-m3` cross-encoder on the candidate window (8192-token context noted).
3. **Thin agentic orchestration** — tools = query patient structured fields / vector-search patient turns; every tool call is scoped per patient and per grant; every claim cited; refusal outside the grant.

**Normalization contract (non-negotiable):** identical preprocessing on the **index** and **query** sides — remove diacritics, unify alef variants (أ/إ/آ → ا), teh-marbuta (ة → ه), yaeh (ى → ي), strip tatweel/kashida. ASR output is undiacritized, so this is load-bearing for retrieval quality.

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

Runtime picks (locked): Cohere Transcribe Arabic 07-2026 (transformers/vLLM, ≤8 GB GPU class), pyannote `community-1` (k=2), `bge-m3` + `bge-reranker-v2-m3` (sentence-transformers / TEI / vLLM), the prescription HTR engine **held behind our eval** (Azure DI or fine-tuned trOCR), the patient memory store = the relational schema of §3.

**Open decisions carried forward (flagged, not locked):**

- The **generation model** for the three surfaces (an Arabic-capable on-prem-capable LLM) — the conversation has not yet selected it. Selection criteria: on-prem capable, Arabic-primary quality, refuse-able/honest behavior; candidate validation goes on the code-switch eval set before lock-in.
- The **roster model** — federation vs single-operator (sub-decision under the trust gate).
- **Scale targets** — patients, sessions/day, retention (feed storage sizing later).

### 8. Deployment topology

- **Development:** cloud, **synthetic data only**; the full stack runs identically to production (swappable-component topology).
- **Public deployment:** **on-prem for real data**; every locked component (Cohere ASR, pyannote, bge-m3, reranker) is on-prem capable by construction. Nothing in the v1 runtime requires a cloud API; the only cloud-touching pieces are the model-authorization gates that fetch gated weights at deploy time.
- **Data boundary:** real PHI never crosses the premises boundary in a public deployment; the boundary is enforced by topology, not by policy.
- **The trust gate is a service** — authorization lives between every surface and the store; no request reaches retrieval without a grant check.
- v2+ captioner layer (scans) must slot into this topology later **without re-architecting** — it is a new ingestion path over the same store, gated by the same trust gate.

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
- **Code-switch retrieval set:** the 50–100-query Arabic+English medical eval set (the only evidence for that slice — no public benchmark exists) is a first-class fixture used at this seam.

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
- Where a lock is held open (HTR engine, generation model, roster model), it is tagged in this spec; build sessions must not silently substitute a model free of those criteria.
- Next build session: build against this spec, test at the middle-layer seam, and use the fixture-patient-memory pattern from the start.