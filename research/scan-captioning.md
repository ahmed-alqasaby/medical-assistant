# Scan Captioning Models — Research Findings

**Ticket:** [#4 — Medical scan captioning models](https://github.com/user/medical-assistant/issues/4)
**Date:** 2026-09-12
**Status:** Research complete

---

## ⚠️ SCOPE DECISION (2026-09-12) — DEFERRED (V2+, out of scope for this build)

**Decision (ticket #4):** scan image understanding is **OUT of scope for version 1.** Rationale (product
owner): scans can't be meaningfully explained by the v1 design; scan interpretation joins as "another
layer at another version." Version 1 embedded data = **session transcript + confirmed prescriptions +
text-based lab results**; scans leave the retrieval index entirely.

- This doc is retained as **V2+ design inventory** (the research is still valid evidence for the later layer).
- Ingestion (#10) is consequently **untied from the vision-model dependency**; `scan-captioning` no
  longer blocks it. Any V2 design note: the swappable-captioner-interface requirement survives; the
  trusted/unverified split (modality/region/view trusted vs findings unverified) applies when the layer
  is built; MedGemma Health-AI ToU verification stands as an open item for that version.

---

## What This Covers

We need a model that produces **caption text from medical scans** (X-ray, MRI, ultrasound) — not diagnosis, just a text description useful for **RAG retrieval** (embedding the caption so scans are findable via text search). The caption needs enough discriminative content (modality, body region, gross findings) to be useful for retrieval.

**Definition from CONTEXT.md:** *"Scan caption — A vision-model text description of a scan, used so scans participate in text RAG"*

---

## Evidence Summary

### General VLMs on Medical Imaging

| Model | Study | Finding |
|---|---|---|
| GPT-4V | European Radiology 2024 | Modality recognition ~100%, body region ~87%, but pathology accuracy only ~35%; hallucination rate 46.8% overall (X-ray 19.6%, US 60.6%, CT 51.5%) |
| GPT-4o | Jiang et al. NAACL 2025 Findings | Fails at radiology report generation; "hallucination, incorrect findings, and omissions" across X-ray, CT, MRI |
| GPT-4o | Diagnosing Chest X-ray Study (PMC 2026) | Primary diagnosis accuracy 42%; misidentifies laterality; image-only interpretation poor |
| GPT-4.1 | Pediatric CXR pneumonia (2026) | Most balanced performance: 84% sensitivity, 76% specificity |
| Claude 3.7 Sonnet | CXR nodule detection (Frontiers 2026) | Accuracy 0.651; high specificity (0.933) but very low sensitivity (0.316); best among proprietary single-model LLMs for this task |
| Claude 3.7 Sonnet | Japan Diagnostic Radiology Board (Hirano et al. 2025) | 55% accuracy (vision); low legitimacy scores from radiologists (median 3/5) |
| o3 | Japan Diagnostic Radiology Board (2025) | 72% accuracy (vision); high legitimacy scores (median 4-4.5/5) |
| Gemini 2.5 Pro | Japan Diagnostic Radiology Board (2025) | 70% accuracy (vision); highest legitimacy scores from radiologists (median 4-5/5); greatest improvement with image input |
| Gemini 2.5 Pro | Pediatric CXR pneumonia (2026) | 97% sensitivity but only 25% specificity — overcalls positives |
| GPT-4o, Gemini 1.5 Pro | CXR interpretation study (Diagnostics 2026) | Gemini 1.5 Pro highest detection rate overall; GPT-4o second; both struggle with small lesions and mediastinal findings |

**Key takeaway:** All general VLMs produce hallucinated findings at significant rates (19-60% depending on modality). None reaches clinical reliability for diagnosis. But for **retrieval captions** (modality + region + gross findings), the modality/body-region recognition is strong (~87-100%).

### Medical-Specific Models

| Model | Availability | Coverage | Notes |
|---|---|---|---|
| **Med-PaLM M** | NOT public (Google Research) | Multi-modal medical | SOTA CXR report gen (53.56% micro-F1); 0.25 clinically significant errors/report — but unusable for product |
| **LLaVA-Med** | Open (microsoft/LLaVA-Med on HF) | Research-only, English only | ~50.2% relative to GPT-4 on VQA; hallucination issues; not suitable for production |
| **RadFM** | Open (chaoyi-wu/RadFM on HF) | 2D + 3D radiology (all modalities) | 14B LLaMA-based; beats GPT-4V on RadBench medical VQA (2.87 vs 2.13); MIT license; 2023 architecture |
| **CheXagent** | Open (StanfordAIMI/CheXagent-8b on HF) | **CXR only** — not MRI/US | Beats GPT-4V/RadFM/LLaVA-Med on CheXbench; chest X-ray only |
| **MedGemma** | Open weights (google/medgemma on HF, gated) | CXR, dermatology, ophthalmology, pathology, CT, MRI | 4B multimodal + 27B multimodal; SigLIP medical encoder; approaches task-specific SOTA; Health AI Developer Foundations ToU |

### RAG Retrieval Specifics

- **Caption text is embedded** for semantic retrieval (ACL 2026 LVLM-aware multimodal retrieval for medical RAG). Captions only need discriminative terms (modality, region, gross findings) for retrieval to work.
- **Hallucinated findings in captions pollute patient memory** — a fabricated finding in a caption becomes "truth" in the RAG system.
- **Structured caption format** (modality, body part, view, laterality, key findings) improves retrieval quality vs free-form descriptions.

---

## Concept-Build Recommendation

**Cloud API (development): GPT-4o via OpenAI API**

Rationale:
- Mature HIPAA BAA available (updated July 2026); modified zero data retention for API
- Strong modality/body-region recognition (100%/87%)
- Wide ecosystem, easy integration
- o3 and GPT-4.1 show further improvements but GPT-4o is stable and well-documented
- Claude showed low sensitivity on CXR tasks and low legitimacy scores in radiologist review; Gemini overcalls positives

**Local fallback (production/on-prem): MedGemma 4B multimodal**

Rationale:
- Open weights on HuggingFace (gated access, Health AI Developer Foundations ToU)
- Medical-tuned SigLIP encoder trained on radiology images
- Approaches task-specific model performance on CXR and CT/MRI
- Runs on modest GPU hardware (4B parameters)
- No data leaves premises

**Architecture note:** Design the captioner as a swappable interface. Validate cloud captions against local MedGemma captions early to ensure consistency. The public deployment must be on-prem capable (ticket #9).

---

## Privacy Posture

| Provider | HIPAA BAA | Data Retention | Image Processing |
|---|---|---|---|
| OpenAI API | Available (signed BAA required) | Modified zero data retention (API only) | Images processed in OpenAI cloud; PHII leaves premises |
| Anthropic Claude API | Available (signed BAA required) | Zero data retention option | Images processed in Anthropic cloud; PHII leaves premises |
| Google Vertex AI / Gemini | GCP BAA covers Vertex AI | Configurable retention | Images processed in Google cloud; PHII leaves premises |

**All cloud options require:** BAA execution, PHI de-identification for development, and acknowledgment that images leave the local network.

---

## Caveats

1. **No model is clinically reliable** — all evidence shows significant hallucination rates, especially on fine-grained pathology findings. Captions are for retrieval, not diagnosis.
2. **Caption quality varies by modality** — X-ray performs best; MRI and ultrasound have higher error rates across all models.
3. **Medical-specific models underperform general VLMs** on several benchmarks (Liu et al. 2025) — the field is converging toward general-purpose models.
4. **RadFM** (2023) is the only open radiology-specific model with broad modality coverage but its architecture is dated.
5. **CheXagent** is excellent for CXR but useless for MRI/US — not suitable as primary captioner.

---

## Sources

- European Radiology 2024: GPT-4V on radiology images (modality/body region/pathology)
- Jiang et al. NAACL 2025 Findings: GPT-4o failing at radiology report generation
- Hirano et al. 2025 (PMC 12769535): Japan Diagnostic Radiology Board — 8 models, vision + text-only
- Frontiers in Digital Health 2026: CXR pulmonary nodule detection — 9 models
- Diagnostics 2026 (MDPI 2075-4418/16/3/376): CXR interpretation by Gemini/GPT models
- Pediatric CXR pneumonia benchmark 2026: GPT-4.1/Claude 4/Gemini 2.5 Pro/Grok 2
- Liu et al. 2025 (arXiv 2507.11200): Comprehensive medical VLM benchmark
- MedGemma model card (Google, May 2025)
- OpenAI HIPAA BAA guide (July 2026)
- Anthropic Claude API BAA (January 2026)
- GCP HIPAA BAA (cloud.google.com/terms/hipaa-baa)
- ACL 2026: LVLM-aware multimodal retrieval for medical RAG
