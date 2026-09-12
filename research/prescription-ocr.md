# Research: Handwritten Arabic Prescription OCR vs. Structured E-Form Capture

**Ticket:** Medical RAG project — decide the capture channel for the prescription (روشتة), the concept-build channel.
**Date:** 2026-09-12
**Mode:** Research only. Primary/high-trust sources cited inline; everything below traces to a cited source.

**Question:** Is OCR of handwritten Arabic pharmacy tickets feasible today for structured extraction (drug, dose, instructions), or should the capture channel instead be a digital e-form at the clinic?

**Answer up front:** OCR of *legible* handwritten Arabic is possible, but structured extraction from messy, real-world handwritten prescriptions is **not reliable enough to be the primary capture channel in a concept build**. The pragmatic path is a **structured e-form capture at the clinic (digital), with an optional photo/handwriting scan kept as a fallback** — exactly the channel the existing market (Saudi Wasfaty, GCC e-prescription) has already standardized on. Reasoning follows.

---

## 1. State of Arabic handwriting recognition (HTR)

### 1.1 What exists

| Model / resource | Type | Best published result (clean benchmark data) |
|---|---|---|
| **HADARA80P** dataset (Pantke et al., 2014) | Historical Arabic book word-spotting/HTR corpus (~1,336 lines) | Self-ONN + deformable CNN: CER 7.98%, WER 31.99% on HADARA80P |
| **trOCR** (Microsoft, 2023) | Transformer OCR (ViT encoder + text decoder); the base arch for most Arabic HTR work | Fine-tuned trOCR on KHATT: CER 13.82%, WER 35.66% |
| **HATFormer** (2024/25) | trOCR-based Arabic HTR, historical focus | CER 8.6% on Muharaf; 15.4% on KHATT |
| **CRNN baselines** (2025–26) | CNN-BiLSTM-CTC + language model | CER 7.76–8.45% on KHATT; 10.11% on Muharaf |
| **Arabic digit/numeral recognition** (CNNs, 2021) | Isolated Arabic + Hindi numerals | ~99.8% on clean isolated digit images |
| **LLMs / VLMs** (GPT-4o, PaddleOCR 3.0, Mistral OCR, 2025) | Zero-shot Arabic HTR | GPT-4o: CER 39%, WER 70% on OnlineKHATT — *worse* than fine-tuned small models |
| **Tesseract** (open-source) | General OCR | Handwritten Arabic out-of-the-box: ~20–30% accuracy (CER ≈ 32%, WER ≈ 66%) |

Sources: HADARA80P numbers — Mohammed et al. 2022 (arXiv:2207.08139); trOCR / HATFormer numbers — HATFormer (arXiv:2410.02179); KHATT/Muharaf CRNN numbers — arXiv:2601.16713 and King Saud Univ. JKSU-CIS 2026 (10.1007/s44443-026-00970-6); numeral recognition — Applied Sciences 2021 (MDPI 2076-3417/11/4/1573); VLM zero-shot comparison — Alsaedi et al. 2025 (10.1109/dese68208.2025.11367891); Tesseract handwritten Arabic — Faizullah et al. 2024 (IJACSA 10.14569/ijacsa.2024.01510120) and Wagler 2026 (10.48448/qymj-j303).

### 1.2 The numbers you must not over-trust

Two structural caveats dominate every headline number:

1. **Benchmark-vs-deployment gap is enormous.** HTR accuracy collapses when you move across domains. Cross-dataset results (trained on KHATT → tested on Muharaf) show CER climbing to **40–44%**, and reverse transfer to 16–28% — i.e. a model fine-tuned on one corpus style degrades 2–5× on unseen handwriting (HATFormer, arXiv:2410.02179). **A model trained on Arabic handwriting in general has never been transferred to Arabic *medical prescriptions* and shown to hold accuracy.**
2. **No public Arabic prescription-handwriting dataset exists.** The benchmarks everyone cites (KHATT, Muharaf, MADCAT, HADARA80P, IFN/ENIT, AHCD) are general prose or historical manuscripts. None contains drug names, doses, abbreviations, or the Arabic/Latin script mixing that prescriptions have (Muharaf paper, arXiv:2406.09630, surveys the whole dataset landscape — no medical dataset among them).

### 1.3 The specific models/datasets named in the ticket

- **HADARA80/HADARA80P** — a *historical manuscript* word-spotting corpus (80 pages of a 1430-AD book). Useful to evaluate HTR machinery, **irrelevant as training data for modern prescriptions** (Pantke et al., ICFHR 2014).
- **Ar-PHOCD** — no standard resource by this exact name was found in primary sources. The closest verified items are the PHOC (Pyramidal Histogram of Characters) word-embedding line of work and the "Arabic Ch4k" / AHCR character datasets. **Unable to verify — treat as an ambiguous reference, not a buildable asset.**
- **trOCR** — verified and real; the strongest *starter* architecture for Arabic HTR and the active base for fine-tuned community Arabic models, but only reaches usable accuracy on-domain (per §1.2).
- **"Google/Dar Elcorpus"** — ambiguous/unverified as stated. What is *verifiable* is that Google's own commercial OCR (**Cloud Vision `DOCUMENT_TEXT_DETECTION`**) **does not support handwriting for the Arabic script at all** — its handwriting-script support table lists only Bengali, Cyrillic, Devanagari, Greek, Chinese, Japanese, Korean, Latin and Vietnamese; `ar` appears only for *printed* text ("Modern Standard"). Google Cloud Vision is therefore **not a viable Arabic-handwriting capture path today** (Google Cloud Vision OCR language-support & handwriting docs).
- **"KFUP MCI"** — unverified as stated. The closest KFUPM-adjacent verified assets are the AHDB/AHDB-FTR Arabic handwriting corpora and the KHATT/prescription-adjacent line of work cited above. Flagged, not buildable.

---

## 2. Commercial API support for Arabic handwriting

| API | Arabic handwriting support | Notes |
|---|---|---|
| **Google Cloud Vision** (`DOCUMENT_TEXT_DETECTION`) | ❌ **Not supported** | `ar` is printed-only; Arabic script absent from the handwriting-scripts table. |
| **Google Document AI** | Printed Arabic yes; Arabic *handwriting* not listed as supported | Same script constraint class as Vision. |
| **Azure AI Document Intelligence (Read v4)** | ✅ **Supported** (`ar` added in the 2023-10-31-preview, GA after) | This is the strongest commercial off-the-shelf option for Arabic handwriting — but Microsoft publishes **no accuracy numbers** for Arabic handwriting, only support claims. |
| **Azure Computer Vision Read (classic)** | ❌ Not for Arabic | Handwritten table lists only Latin/CJK/Korean/Portuguese/Spanish etc. |
| **Tesseract / EasyOCR / PaddleOCR** | Weak out-of-box; needs fine-tuning | Fine-tuned Arabic handwritten Tesseract still lands at ~22% CER (≈78% error); EasyOCR exact-match on cursive prescriptions ≈2.6% (see §3). |

Sources: Google Cloud docs — docs.cloud.google.com/vision/docs/languages ("Handwriting scripts" table) and docs.cloud.google.com/vision/docs/handwriting; Azure — learn.microsoft.com/azure/ai-services/document-intelligence/language-support/ocr ("Read: handwritten text" includes `ar`), and Azure Vision language-support page (handwritten table has no Arabic); Tesseract fine-tune — Wagler 2026 (10.48448/qymj-j303).

**Implication:** Among hosted APIs, only **Azure Document Intelligence** currently exposes Arabic handwriting, and its accuracy on medical content is unbenchmarked. Everything else requires self-hosting a fine-tuned model.

---

## 3. Realistic accuracy on messy handwritten prescriptions

### 3.1 English prescriptions (the best-documented case)

A 2026 independent benchmark of modern OCR engines on **cropped handwritten English prescription words** (CPU-default, no medical fine-tuning):

- **GLM-OCR:** CER 0.328 (≈33% chars wrong), exact-word match **32.6%**
- **PP-OCRv5:** WER 0.789, exact-word match **21.4%**
- **Tesseract:** exact-word match **2.5%**; **EasyOCR:** **2.6%**

The write-up states the clinical conclusion plainly: *"roughly two words out of three were not reproduced perfectly. A system making errors at that rate cannot safely determine medicines, strengths or dosing instructions without human verification."* And this benchmark **did not even test page-finding, layout, or drug↔dose↔instruction relationship preservation** — it used pre-cropped words. (Arabian Post, "Modern OCR narrows gap on doctors' handwriting", 2026-08-17, summarizing the 2026-04 benchmark.)

Published "clean handwriting" figures look better because they are **not clinical**: TrOCR fine-tuned on English prescriptions reaches CER 8.7% / WER 12.5% / exact-match 81.3% (JCBi 2025) — but the same paper documents failure modes on messy samples: `400mg/60mg` → `400mg16mg`, `1+1+1` → `1H+1`, `40mg` → `40vg`. **A dose read wrong is a dose issued wrong.**

### 3.2 Arabic prescriptions (the actual target)

- **No published end-to-end accuracy benchmark exists** for Arabic handwritten prescriptions (verified: no such dataset or benchmark appears in the Arabic HTR literature surveyed in §1.3).
- The credible Arabic prescription-OCR projects that do exist are **heavy multi-engine ensembles with a human-in-the-loop review stage**, which tells you what's required to make it work at all:
  - **omni-medical-suite** (DrAbdulmalek, GitHub): OCR **router across 7 engines** (EasyOCR + PaddleOCR + Tesseract + trOCR + QARI + Nougat + Qwen), then Arabic spell-correction ("Jais LLM proofread"), then medical NER for drug/dose/date/diagnosis, plus a human review/feedback loop that re-ingests corrections for retraining.
  - **OCR-For-Medical-Prescriptions** (David-Magdy, GitHub): YOLO text-region detection → trOCR fine-tuned on KHATT/prescription data → SymSpell against a medicine dictionary → medicine/instruction pairing. States it "performs best with clear, legible handwriting" and requires a curated drug database.
  - **Pen-To-Pill** (Machathon 6.0): YOLOv8 → trOCR → **mBART** to reconstruct `(medicine, dosage)` pairs.
  - All three are **student/hackathon-grade prototypes, not clinically validated**. None reports clinical-grade accuracy numbers on a real messy corpus.
- Every one of these pipelines would need a **medical lexicon** and likely **per-clinic doctor-seed calibration** — i.e., exactly the data flywheel a concept build does not have yet.

### 3.3 The structural reasons Arabic prescriptions are a worst case for OCR

1. **Script**: cursive, context-dependent ligatures, diacritics that change meaning (ب/ت/ث differ only by dots) — the dominant Arabic-HTR error classes are exactly *dot-related character confusion* and *word-boundary errors* (JKSU-CIS 2026, 10.1007/s44443-026-00970-6). In drug names, a one-dot error is a different medication.
2. **Bilingual mixing**: drug names are usually Latin/brand script mixed with Arabic instructions; OCR engines trained monolithically on Arabic or Latin alone are the norm and degrade on mixed lines.
3. **Abbreviations**: `قرص`, `تب`, `cap`, `أص` instructions and sig abbreviations are clinic-specific and unstandardized.
4. **Numbering systems**: doses may be in Western Arabic digits (`8`) **or** Eastern Arabic/Hindi numerals (`٨`) — two different digit sets in the same ticket. (The 2021 Appl. Sci. system that hits 99.8% is *clean isolated digits only*, not in-context dose digits inside handwriting.)
5. **Domain OOV**: drug names are out-of-vocabulary for any general Arabic LM that post-processing would use — so spell-correction either "corrects" real drug names into nothing, or must be gated behind a local medical dictionary (the exact approach the prototypes above take).
6. **No data**: no annotated Arabic medical-prescription corpus exists to fine-tune on (§1.2). Fine-tuning a trOCR-class model on KHATT gives you prose, not prescriptions; the cross-domain collapse we measured in §1.2 (CER 28–44% across corpora) is the risk you'd be shipping.

### 3.4 Downstream structured extraction (drug / dose / instructions)

- Arabic **clinical NER on clean text** is workable but already the weakest link: CAMeLBERT+CRF F1 ≈ 90% on disease/symptom/organ; ABioNER ≈ 85% on disease/treatment; John Snow's licensed `ner_clinical` model exposes PROBLEM/TEST/TREATMENT in Arabic.
- **None of these NER models has been evaluated on OCR-noisy prescription text.** OCR errors propagate straight into the NER layer, and the entities that matter in prescriptions (drug *names*, numeric doses, sig abbreviations) are precisely the ones Arabic biomedical NER struggles with — the papers above annotate *diseases/symptoms/organs*, not dose+frequency schedules (IJACSA 2025; BioMed Research Int. ABioNER 2021; John Snow Labs `ner_clinical`).
- Consequence: **even with a perfect OCR, structured (drug, dose, instructions) extraction for Arabic is a model-engineering project in itself, not a solved primitive.**

---

## 4. The pragmatic concept-build recommendation

### Recommendation: **Structured e-form capture at the clinic (digital), with a camera-photo fallback.**

1. **A structured e-form is the capture channel.** At session end the doctor (or staff via the doctor's note) fills a short structured prescription — drug (autocomplete from a local medicine list), dose, frequency, duration, instructions, plus the "script as free text" field mirrored to the patient. This yields clean, structured data by construction, with **zero HTR risk**, and it is exactly what the system digitizes for the patient-memory RAG anyway.
   - **Proof it works at national scale:** Saudi Arabia's **Wasfaty** e-prescription link (doctor → pharmacy, patient gets an SMS code) launched 2018; by Dec 2021 it had served ~5M patients across 228 hospitals / 2,069 primary-health centers / 3,100 community pharmacies in 174 cities; multiple peer-reviewed satisfaction/adoption studies (2022–2025) document routine use, including for chronic-disease refills (NUPCO/MOH; Almaghaslah et al. *Healthcare* 2022; Tobaiqy et al. *Int J Gen Med* 2023; Alsahali et al. *Pharmacy* 2023; *BMC Health Serv Res* 2025). GCC neighbors run equivalents (Dubai e-ClaimLink). **The e-form channel is not hypothetical — it's the incumbent.**
2. **Keep a photo fallback as a *secondary* channel** — a photograph of the handwritten ticket attached to the session — but store it as an **image with a human-reviewed transcription**, not as an autonomous structured-extraction path. That preserves the handwritten record for the patient memory without betting the concept on HTR.
3. **Defer any Arabic HTR investment** to the post-concept phase, and only if a corpus accumulates: photo fallbacks + human corrections are the natural seed data for a future fine-tuned trOCR-style model + medical-lexicon post-processing (the exact pattern the successful prototypes follow). Until then, Arabic HTR stays a **demo capability, not a capture capability**.

### Why not the other direction (HTR-primary)?

- Hosted Arabic handwriting OCR is one unbenchmarked API (Azure); Google Vision doesn't do Arabic handwriting at all.
- Where benchmarks do exist (English prescriptions, clean subsets), messy-handwriting exact-match is 21–33% on cropped words — clinical floor is far above that.
- The prescription-specific HTR stack costs: 7-engine routers, medical lexicons, per-clinic calibration, HITL review — none of which a concept build can staff or evaluate.

---

## 5. What breaks if the non-recommended path (HTR-primary) is used

1. **Medication identity breaks.** Dot/ligature errors in Arabic drug names alias one medication to another (ب/ت/ث family); no Arabic medical lexicon exists to catch it, and dictionary-free "spell correction" mangles OOV drug names.
2. **Dose and frequency break.** Digit confusion (Western vs. Eastern numerals, `mg` mangled to `vg`, `1+1+1` → `1H+1`) makes the dose field a liability instead of an asset — the single most safety-critical field.
3. **The RAG trust contract breaks.** The product's core is a *grounded* patient memory (CONTEXT.md: "answers must be grounded in retrieved evidence, never free invention"). A store seeded with 2-in-3-words-corrupt OCR text poisons retrieval: the patient-memory RAG would faithfully retrieve *wrong* medications and doses. Silent wrongness is worse than absence.
4. **Structured extraction breaks.** (drug, dose, instructions) is not a solved Arabic primitive even on clean text — layering it onto ≤78%-accurate handwriting OCR compounds two unreliable stages.
5. **Build economics break.** To make HTR-primary work you must assemble the full ensemble stack (multi-engine router + medical dictionary + NER + HITL review + retraining loop). That is the product now, not the RAG product the project is building.

**Net:** HTR-primary works only as a sad demo; e-form-primary works immediately and is the incumbent market pattern.

---

## 6. One-line evidence/confidence note

**High confidence** that e-form capture is the pragmatic concept-build channel: state-of-the-art Arabic HTR only reaches 8–16% CER on *clean in-domain* data and collapses to 28–44% CER cross-domain (HATFormer arXiv:2410.02179; JKSU-CIS 2026); messy-prescription OCR lands at 21–33% exact-word on *pre-cropped* words (2026 engine benchmark); Google Vision has **no** Arabic-handwriting support while Azure lists it **without published accuracy** (vendor docs); Arabic prescription-structured extraction has **no public benchmark or dataset**; Saudi Arabia's Wasfaty e-prescription (2018, ~5M patients by 2021) already operates the recommended channel at national scale (Almaghaslah 2022; Tobaiqy 2023). Unverified ticket references (Ar-PHOCD, "Google/Dar Elcorpus", "KFUP MCI") are flagged as ambiguous rather than relied on.

---

## Sources

1. Mohammed, Malik, Al-Madeed, Kirayaz — "2D Self-Organized ONN Model for HTR" (HADARA80P results), arXiv:2207.08139, 2022.
2. Saeed et al. — "Muharaf: Manuscripts of Handwritten Arabic Dataset", NeurIPS D&B 2024 / arXiv:2406.09630.
3. "HATFormer: Historic Handwritten Arabic Text Line Recognition with Transformers", arXiv:2410.02179 (v1/v2) — CER 8.6% Muharaf / 15.4% KHATT / cross-dataset 16–44%.
4. arXiv:2601.16713 — CRNN Arabic-script HTR, CER 8.45% KHATT / 10.11% Muharaf.
5. "A controlled study of CTC, attention, hybrid models for Arabic HTR", JKSU Computer & Information Sciences (Springer), 10.1007/s44443-026-00970-6, 2026 — decoder/LM study, dominant error classes = dots + word boundaries.
6. Al Ali, Almansoori, Elnagar — "Arabic HTR Using TrOCR with LLM-Based Post-Processing", IEEE 2026, 10.1109/icetes68504.2026.11518724 — trOCR-on-KHATT CER 13.82% / WER 35.66%.
7. Alsaedi et al. — "Investigating LMMs' Potential in Arabic Handwriting Recognition", IEEE 2025, 10.1109/dese68208.2025.11367891 — GPT-4o CER 0.39 / WER 0.70 on OnlineKHATT.
8. Al-Maamari et al. — "Integrating CNN and transformer architectures for Arabic printed/handwritten character classification", Scientific Reports 2025, 10.1038/s41598-025-12045-z — 98–99% character-level (isolated chars, not lines).
9. "Recognition of Handwritten Arabic and Hindi Numerals", Applied Sciences 11(4):1573, 2021 — ~99.8% on clean isolated digits.
10. Faizullah et al. — "LSTM-Enhanced OCR for Arabic Handwritten Manuscripts", IJACSA 2024, 10.14569/ijacsa.2024.01510120 — Tesseract Arabic-handwriting accuracy 23.30% out-of-box.
11. AFCEA 2026 / Wagler — "Reducing OCR Error Rates for Chinese and Arabic Handwritten Text (Tesseract + Tesstrain)", 10.48448/qymj-j303 — baseline CER 61% → 22% after fine-tune.
12. Google Cloud — "OCR language support: Cloud Vision" (handwriting-scripts table; `ar` printed only) and "Detect handwriting in images", docs.cloud.google.com/vision/docs/languages, /vision/docs/handwriting.
13. Microsoft Learn — "Language support: document analysis (Azure AI Document Intelligence)" — Read handwritten table includes `ar`; and Azure Vision "Language support" — handwritten table has no Arabic; "What's new" (2023-10-31-preview) — "Language Expansion for Handwriting: Russian, Arabic, Thai."
14. "Modern OCR narrows gap on doctors' handwriting" — Arabian Post, 2026-08-17 — summary of the April-2026 prescription-word benchmark (GLM-OCR 32.6% exact / PP-OCRv5 21.4% / Tesseract 2.5% / EasyOCR 2.6%).
15. JCBi 2025/2026 — "AI-driven OCR using TrOCR + Roboflow for handwritten prescriptions" — CER 8.7% / WER 12.5% / exact-match 81.3% on *clean* handwritten English samples; documented dose/symbol failure modes on messy ones.
16. GitHub — DrAbdulmalek/omni-medical-suite; David-Magdy/OCR-For-Medical-Prescriptions; HabibaYossre/Pen-To-Pill — Arabic+English handwritten-prescription OCR pipelines (architecture evidence only; no clinical validation).
17. Almaghaslah et al. — "Patients' Satisfaction with E-Prescribing (Wasfaty) in Saudi Arabia", Healthcare 2022, 10.3390/healthcare10050806 — ~5M patients / 228 hospitals / 2,069 PHCs / 3,100 pharmacies by Dec 2021.
18. Tobaiqy et al. — "Prescription Transfer and Medicines Collection Through Wasfaty", Int J Gen Med 2023, 10.2147/ijgm.s432075.
19. Alsahali et al. — "Community Pharmacists toward the National E-Prescribing Service (Wasfaty), Qassim", Pharmacy 2023, 10.3390/pharmacy11050152.
20. "Exploring family physicians' experiences with Wasfaty", BMC Health Services Research 2025, 10.1186/s12913-025-13217-3.
21. John Snow Labs — `ner_clinical` Arabic, nlp.johnsnowlabs.com/2023/10/06/ner_clinical_ar.html; Gannoune et al. IJACSA 2025 (CAMeLBERT+CRF F1 90%); Boudjellal et al. BioMed Research Int. 2021 (ABioNER F1 85%) — Arabic biomedical NER on clean clinical text only.

*Unverified ticket references, flagged not used:* "Ar-PHOCD", "Google/Dar Elcorpus", "KFUP MCI" — no primary source found under these exact names.