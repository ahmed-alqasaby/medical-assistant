# CONTEXT.md

Glossary for `medical-assistant`. Devoid of implementation detail — this file defines domain terms only. Decisions live in `docs/adr/`.

## Terms

### Patient memory
The per-patient consolidated store of everything from their medical sessions. The core conceptual unit of the product. A doctor with access views it; a patient owns it.

### Connected practice
A clinic on the system that serves a patient and can request access to that patient's memory.

### Medical session
One doctor–patient encounter. Contains: a session recording (audio), a transcript, a prescription, and any lab results produced or received during the encounter.

### Session recording
The in-room audio captured during a medical session. In-person only — telehealth is explicitly out of scope (a telehealth platform is the thing this project exists to replace).

### Transcript
The ASR-derived text of a session recording, diarized by speaker (doctor/patient) where the technology allows.

### Prescription (روشتة)
The medication, dose, and instructions issued at the session. Stored as a structured record **only after human confirmation** — the OCR/HITL draft is not part of the memory. (Capture = OCR-primary with a formulary-constrained confirmation gate.)

### Confirmed prescription
The human-approved version of a prescription that actually enters the patient memory. Unconfirmed OCR drafts are pipeline state, never retrievable as fact.

### Lab result
A text-based test result (hematology, biochemistry, etc.) belonging to a session. Enters the memory as structured text.

### Scan
A medical image (X-ray, MRI, ultrasound) produced at a session. **v1: stored outside the retrieval memory — image understanding is out of scope (v2+) and scans are not captioned/embedded in this version.**

### Scan caption
**Deprecated (v1) — v2+ concept only.** A vision-model text description of a scan, used so scans participate in text RAG.

### Middle layer
The RAG + LLM surface positioned between the doctor and the patient over the patient memory. What the product *is*.

### Pre-visit brief
The AI-synthesized image of the patient generated from their full memory before a consult.

### Live grounded Q&A
Real-time interrogation of the patient memory during a consult. Answers must be grounded in retrieved evidence, never free invention.

### Plain-language lens
The patient-facing RAG view of their own memory, explaining their records in simple terms. Same store, different generation contract.

### Trust gate
The patient-granted, revocable access control over their memory. No doctor sees anything without the patient's approval — access is earned, not automatic.