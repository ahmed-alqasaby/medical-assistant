# CONTEXT.md

Glossary for `medical-assistant`. Devoid of implementation detail — this file defines domain terms only. Decisions live in `docs/adr/`.

## Terms

### Patient memory
The per-patient consolidated store of everything from their medical sessions. The core conceptual unit of the product. A doctor with access views it; a patient owns it.

### Connected practice
A clinic on the system that serves a patient and can request access to that patient's memory.

### Medical session
One doctor–patient encounter. Contains: a session recording (audio), a transcript, a prescription, and any scans produced during the encounter.

### Session recording
The in-room audio captured during a medical session. In-person only — telehealth is explicitly out of scope (a telehealth platform is the thing this project exists to replace).

### Transcript
The ASR-derived text of a session recording, diarized by speaker (doctor/patient) where the technology allows.

### Prescription (روشتة)
The medication, dose, and instructions issued at the session. The capture channel (digital e-form vs. handwritten OCR) is an open decision.

### Scan
A medical image (X-ray, MRI, ultrasound) produced at a session. Stored raw and also captioned for text retrieval.

### Scan caption
A vision-model text description of a scan, used so scans participate in text RAG.

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