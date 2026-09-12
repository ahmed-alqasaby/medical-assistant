# ASR + Speaker Diarization for Conversational Medical Audio (Arabic / English)

Research ticket: which ASR path handles in-room doctor–patient audio in Arabic (primary) and
English (secondary) with speaker diarization, suited to a **concept build**. Research only — no
build work. Every claim below is cited to the primary source (official docs, model cards, papers).

---

## TL;DR — recommendation

| Role | Pick | Why |
|---|---|---|
| **Concept-build pick (self-hosted)** | **faster-whisper `large-v3` + WhisperX** (VAD → batched Whisper → wav2vec2 align → **pyannote `speaker-diarization-community-1`**) | One open, runnable model transcribes both Arabic and English in a single pass; Arabic word-timestamp alignment is built in (default `wav2vec2-large-xlsr-53-arabic`); whole pipeline is MIT/Apache, runs on one ≤8 GB GPU, stays on-prem (matters for patient data + the project's trust-gate/patient-memory model). Keep `num_speakers=2`. |
| **Runner-up (open)** | **NVIDIA NeMo: Parakeet-RNNT-1.1B-multilingual + Sortformer diarizer** (NIM `diarizer=sortformer`) | Only mainstream open stack with Arabic **and** English plus integrated end-to-end diarization in one artifact; streaming-capable. Heavier setup (NeMo/NIM containers), less hallucination control than WhisperX. |
| **Fallback / zero-infra API** | **AssemblyAI Universal-3.5 Pro** (batch) or U-3.5 Pro Realtime, `speaker_labels=true` + role-based **Speaker Identification** (`Doctor` / `Patient`) + keyterm prompting for med names | Lowest effort, best "roles not just A/B" diarization, domain prompting built in. ~$0.21/hr batch, $0.45/hr streaming. Sends audio off-site — a privacy gate for real patient data. |

**Watch items before trusting transcripts:** Whisper-family models hallucinate (incl. fabricated
medication names) at ~1% of segments; conversational Egyptian Arabic is far harder than MSA for
every model tested; and "who said what" quality depends heavily on 2-speaker constraint + mic
placement, not just which diarizer you pick.

---

## 1. What the input actually looks like

From the project domain model (`medical-assistant/CONTEXT.md`): a **medical session** is one
in-person doctor–patient encounter whose **session recording** is in-room audio; the **transcript**
is ASR output "diarized by speaker (doctor/patient) where the technology allows". Implications:

- Conversation (two alternators), not narration → diarization is 2-speaker, near-ideal for today's diarizers.
- Room audio → crosstalk, reverb, mic distance; VAD quality is first-order.
- Code-switching Arabic↔English happens naturally in clinical talk → favor **bilingual one-pass** models.
- Prescriptions (روشتة) depend on med-names in the transcript → hallucination control is a hard requirement.

---

## 2. ASR landscape

### 2.1 Whisper large-v3 / large-v3-turbo (open weights, strongest ecosystem)

- Architecture + license facts from the model cards: `large-v3` is a 32-decoder-layer seq2seq,
  1.5B params, 99 languages; `large-v3-turbo` is the same model pruned to 4 decoder layers (0.81B
  params), ~2× faster with minor accuracy cost.
  Sources: https://huggingface.co/openai/whisper-large-v3 , https://huggingface.co/openai/whisper-large-v3-turbo
- Opens' own long-form guidance: sequential decoding is up to 0.5% WER better than chunked; chunked
  is faster for a single file (same sources).
- **Arabic accuracy (be honest about it):**
  - FLEURS (read speech, MSA): large-v2 = **16.0% WER** (OpenAI paper, Table 13) →
    large-v3 is 10–20% better than v2 → realistically ~13–15%.
  - Independent 2026 paired FLEURS run: large-v3-turbo Arabic = **15.75% WER** (Whisper Notes).
  - MSA broadcast (FLEURS test): zero-shot large-v3 ≈ **8.3–8.5% WER** (dev-ahmedhany benchmark).
  - **Egyptian broadcast** (MGB-2, TV): large-v3 = **16.2% WER** (Octopus paper, arabicnlp 2025) / **16.26%** (Cohere AR leaderboard blog).
  - **Conversational Egyptian** (Casablanca calls): zero-shot large-v3 = **40–50% WER** — this is the
    closest public proxy to doctor–patient dialogue, and it is poor. QLoRA dialect fine-tunes cut
    that to ~24% on a mixed Egyptian set at the cost of ~2 pp on MSA.
  - Net: Whisper is *good on MSA, usable on Egyptian broadcast, weak on conversational Egyptian
    out of the box*. Expect concept-build WER on room dialogue ≥15–30% regardless of model unless a
    dialectal/domain fine-tune or a strong vocabulary bias is applied.
- GPU/runtime: faster-whisper (CTranslate2) runs large-v2 in **<8 GB VRAM** with beam=5 (WhisperX README); int8/fp16 cuts this further. WhisperX VAD pre-chunking also *reduces* hallucination (below).

### 2.2 NVIDIA NeMo / Parakeet (open weights)

- **Parakeet-RNNT-1.1B-multilingual**: 25 languages/variants incl. **ar-AR** and en-US/en-GB, ~90k h
  training, automatic language detection, streaming + offline, word timestamps.
  Sources: https://build.nvidia.com/nvidia/parakeet-1_1b-rnnt-multilingual-asr/modelcard , NGC catalog page.
- **Canary-1b**: 26 languages incl. Arabic+English, offline, ASR + bidirectional translation. Not in the push.
- GPU: 1.1B params → similar class as Whisper large-v3 on a single consumer GPU; NIM serves it in containers.

### 2.3 Arabic-specialized / newer open options

- **Google USM** (Zhang et al. 2023, 2B Conformer, 100+ langs incl. Arabic): the technical basis for
  "USM"; **weights are not publicly runnable** (YouTube-internal; described in a paper).
  Source: https://sites.research.google/usm/ , arXiv:2303.01037.
- **ArabicBench** (Abdelali et al., 2025): head-to-head of USM vs Whisper on Arabic datasets — the
  specialized Arabic system beats both, USM edges Whisper. Confirms zero-shot multilingual
  underperforms specialized Arabic ASR. Source: reduced discussion in arXiv:2506.02627.
- **Meta MMS-1b-all** (1162 langs incl. Arabic, wav2vec2 architecture): open weights, CC-BY-NC
  (non-commercial — a licensing blocker for a product). Source: https://huggingface.co/facebook/mms-1b-all
- **Meta Omnilingual** (2026): Wav2Vec2-based multilingual family, fairseq2 checkpoints, community HF conversions. Newer, less battle-tested for Arabic.
- **Cohere Transcribe Arabic** (open, Jul 2026): currently *lowest avg WER on the HF Arabic ASR
  leaderboard* (25.87 avg; Whisper-large-v3 = 36.86 avg; OmniASR-LLM-7B = 28.32). Still weak on
  Maghrebi (Casablanca 49.7) but clearly better than Whisper on dialect-heavy sets. It is an
  Arabic-only model — English is a separate model — so add it only if a two-model fallback is fine.
  Source: https://cohere.com/blog/transcribe-arabic
- **Qwen3-ASR-1.7B** (Qwen, 2025): 30 languages incl. **Arabic + English**, offline/streaming, claims
  SOTA among open models; independent run shows Arabic FLEURS 14.14 vs 15.75 (Whisper turbo) — a
  coin-flip on Arabic, and *worse* than Whisper on conversational Egyptian (Casablanca 57.5 vs 50.4).
  Sources: https://huggingface.co/Qwen/Qwen3-ASR-1.7B , Whisper Notes benchmark.
- **Egyptian fine-tunes of Whisper**: e.g. `dev-ahmedhany/whisper-large-v3-arabic-ft-v3`
  (MSA 10.5 / Egyptian 23.9 / Levantine 30.6 / Gulf 41.5 WER on mixed conversational+broadcast),
  `MAdel121/whisper-medium-egy` (18% WER on its own Egyptian set). Signal: a ~20–40 h dialectal
  QLoRA on large-v3 is the single cheapest accuracy lever for Egyptian after the base model.
- **Dialog-domain fine-tunes of Whisper exist** (e.g., earlier `nececive/arabic-whisper-specialized-v2` is a whisper-large-v2 Arabic fine-tune); verify any HF fine-tune on *your own* session audio before trusting its WER claims — several public fine-tune cards report inflated numbers from test contamination.

### 2.4 APIs

- **AssemblyAI (fallback pick)**: Universal-3.5 Pro — 18 languages incl. Arabic+English with native
  mid-sentence code-switching; marketed explicitly for "medical scribes needing clinical grade
  transcription accuracy"; keyterm prompting (up to ~1,500 words) to bias med vocabulary;
  diarization across 95 languages; $0.21/hr batch, $0.45/hr streaming.
  Sources: https://www.assemblyai.com/docs/getting-started/models , /languages/arabic , /docs/pre-recorded-audio/supported-languages
- (Not chased in depth but present per docs search: Deepgram, Azure/Google — all support Arabic; the
  differentiator for this project is the medical-scribe positioning + role diarization, which is why
  AssemblyAI is called out.)

---

## 3. Speaker diarization

Goal: label each turn **doctor vs patient**, i.e. known 2-speaker conversation.

### 3.1 pyannote.audio (used by WhisperX)

- **`pyannote/speaker-diarization-3.1`**: pure-PyTorch (no onnxruntime), needs HF token + model
  access agreements; `num_speakers=2`/`min|max_speakers` supported → ideal for forcing doctor+patient.
  Source: https://huggingface.co/pyannote/speaker-diarization-3.1
- **`pyannote/speaker-diarization-community-1`** (now the default WhisperX diarizer): better speaker
  counting/assignment, works offline, ~0.3–4 pp DER better than 3.1 on every academic, and notably
  **RAMC** (a medical-consultation corpus) = 20.8 DER vs 22.2 (3.1). Source:
  https://huggingface.co/pyannote/speaker-diarization-community-1
- **`pyannote/speaker-diarization-precision-2`**: much better still (RAMC 10.5 DER) but **hosted API only**.
- Effort: pyannote is the low-effort, well-trodden path; quality on a 2-speaker room recording is
  good but degrades with crosstalk and similar-pitch voices. WhisperX wires VAD → ASR → align →
  diarize → per-word speaker labels in ~30 loc.
  Source: https://github.com/m-bain/whisperX (plus `whisperx/alignment.py` shows the default
  Arabic align model `jonatasgrosman/wav2vec2-large-xlsr-53-arabic`).

### 3.2 NVIDIA NeMo diarization

- Cascaded pipeline: MarbleNet VAD → TitaNet embeddings → clustering + **MSDD** (multi-scale
  diarization decoder).
- End-to-end **Sortformer** diarizer, and NIM `diarizer=sortformer` profiles that fuse ASR+diarization
  (available on Parakeet CTC variants and RNNT Multilingual — the Arabic-capable one).
  Sources: https://docs.nvidia.com/nemo-framework/user-guide/25.09/nemotoolkit/asr/speaker_diarization/models.html ,
  https://docs.nvidia.com/nim/speech/latest/asr/
- Effort: higher (config plumbing, manifest files, or Docker/NIM runtime), but it is the only fully
  integrated open "bilingual ASR + diarization in one artifact" path and adds streaming.

### 3.3 Tradeoff summary

| Option | DER-ish quality (RAMC = medical) | Effort | Notes |
|---|---|---|---|
| pyannote community-1 (via WhisperX) | RAMC 20.8 | Low | Forced 2-speaker; cheap; per-word speaker labels |
| pyannote precision-2 (hosted) | RAMC 10.5 | API | Best, but cloud-only |
| NeMo Sortformer / MSDD+TitaNet | solid on meetings; RAMC not public | Medium–high | Integrated, streaming, all-on-NIM |
| per-utterance **role** mapping | n/a | Low | Post step: heuristics (who talks first / most) or AssemblyAI Speaker Identification assigns "Doctor"/"Patient" by role |
| AssemblyAI `speaker_labels` + role ID | good on calls | Very low | Roles not just A/B; 30-s-per-speaker guidance |

For doctor vs patient specifically: two-speaker constraint + a role assignment pass (doctor usually
speaks more and opens) is enough for a concept build; don't pay for precision-2 yet.

---

## 4. Recommended path for the concept build

### Primary: `faster-whisper large-v3` + WhisperX + pyannote community-1 (num_speakers=2)

1. **Transcribe** bilingual Arabic/English in one pass (`faster-whisper large-v3`, fp16 or int8, on a
   single 8 GB-class GPU). Use `language="ar"` if you want to hard-lock primary (English code-switch
   still decodes), or leave auto.
2. **Word-align** with WhisperX default Arabic align model (`wav2vec2-large-xlsr-53-arabic`) — built in.
3. **Diarize** with WhisperX's pyannote pipeline (community-1), `min_speakers=2, max_speakers=2`.
4. Assign roles by simple rule (first speaker who holds long turns ≈ doctor) or a tiny
   classifier/LLM pass; or ship "Speaker A/B" for v0.
5. **Hallucination guard**: keep WhisperX's VAD churn *on*; set `no_speech_threshold≈0.6`,
   `logprob_threshold≈-1.0`, `compression_ratio_threshold≈1.35`, temp fallback; never soft-transcribe
   long silences.

Rationale: open (MIT/Apache), bilingual one-pass, on-prem (fits patient-memory privacy), proven
combo, ~30 loc, minute-level setup. Tradeoff vs turbo: **large-v3** buys the last few Arabic WER
points; if GPU time matters, drop to `large-v3-turbo` (half the decoder, ~2× speed, ~equal Arabic
FLEURS).

### Runner-up: NeMo `parakeet-rnnt-1.1b-multilingual` + Sortformer diarizer

Same GPU class, Arabic+English, integrated ATO diarization, streaming capable, fully on-prem. Pick
this when you want one vendor + one artifact (NIM) and don't need Whisper's per-segment
hallucination knobs. Note Parakeet-RNNT is also sold as an API for later scale-out.

### Fallback / fastest path: AssemblyAI Universal-3.5 Pro

`speaker_labels=true` + `speaker_identification` with roles `Doctor`/`Patient` + keyterm prompt of
the clinic's med list. ~$0.21/hr. Ship-in-a-day, but cloud-only → revisit for real patient data.

### Accuracy reality-check to calibrate expectations

| Setting | Expected WER (Arabic) |
|---|---|
| MSA read / broadcast | ~8–11% (whisper l3) |
| Egyptian broadcast (MGB) | ~16% |
| Conversational Egyptian (room-like) | ~40–50% zero-shot → ~24% with a 20–40 h Egyptian QLoRA |
| English secondary | 2–6% clean; 8–12% real-world |

First concept milestone should be **collecting 20–30 real in-room sessions**, transcribing with the
primary pick, measuring WER on your own clinical vocabulary — every published number above is from a
proxy corpus.

---

## 5. Hallmarks to watch (medical-specific)

1. **Whisper hallucinations are real and dangerous here** — ~1% of segments contain entire invented
   sentences; a documented case fabricated **a made-up drug name and a claim of taking it**; the
   effect scales with silence/pauses (aphasia/disfluent speakers worse), i.e., exactly what a slow
   patient consult has. Another audit counted **7M+ doctor visits transcribed by a Whisper-based
   scribe (Nabla)** — so this category of risk has already gone to production.
   Sources: Koenecke et al., *Careless Whisper* (FAccT'24), arXiv:2402.08021; Mei et al. (FAccT'26),
   doi:10.1145/3805689.3812320; Burke & Schellmann reporting on Nabla.
2. **Medication names** are a low-frequency token class → ASR mishears them silently (correct-sounding
   wrong word). Mitigations: keyterm/prompt bias toward the practice's drug list (AssemblyAI keyterms;
   Whisper `prompt`/`initial_prompt`), post-ASR spell-check against a pharmacy formulary, and
   **slot-level human review on the prescription field** — never let RAG below this ingest a
   hallucinated med as ground truth.
3. **Diarization errors ≠ WER** — DER doesn't measure transcription accuracy; a clean DER can still
   attach the right words to the wrong speaker. For concept, verify doctor/patient label alignment
   on ~10 sessions by hand before trusting downstream RAG.
4. **Never merge ASR output straight into the patient memory** — tag confidence, keep audio, and make
   the "Live grounded Q&A" cite the transcript segment; hallucinated text quoted into a patient record
   is the exact failure mode the Trust gate is meant to stop.

---

## 6. Sources (primary)

- Whisper large-v3 / large-v3-turbo model cards + long-form docs — https://huggingface.co/openai/whisper-large-v3 , https://huggingface.co/openai/whisper-large-v3-turbo
- OpenAI Whisper FLEURS WER (Table 13, Appendix D) — arXiv:2212.04356 (Robust Speech Recognition via Large-Scale Weak Supervision)
- WhisperX — https://github.com/m-bain/whisperX ; align models incl. Arabic — `whisperx/alignment.py`
- pyannote speaker-diarization-3.1 / -community-1 / precision-2 — https://huggingface.co/pyannote/speaker-diarization-3.1 , .../speaker-diarization-community-1 (RAMC, DER tables)
- NeMo ASR + diarization docs — https://docs.nvidia.com/nim/speech/latest/asr/ , https://docs.nvidia.com/nemo-framework/user-guide/25.09/nemotoolkit/asr/speaker_diarization/models.html
- Parakeet-RNNT-1.1B-multilingual (Arabic in 25 langs) — https://build.nvidia.com/nvidia/parakeet-1_1b-rnnt-multilingual-asr/modelcard
- Qwen3-ASR model card — https://huggingface.co/Qwen/Qwen3-ASR-1.7B ; independent FLEURS runs — https://whispernotes.app/blog/qwen3-asr-vs-whisper
- Cohere Transcribe Arabic (HF Arabic ASR leaderboard numbers for Whisper / OmniASR / Cohere) — https://cohere.com/blog/transcribe-arabic (2026-07)
- Octopus Arabic speech-LLM paper (MGB-2 WER for Whisper-large-v3) — ACL Anthology 2025.arabicnlp-main.35
- Arabic fine-tune benchmarks (zero-shot vs QLoRA conversational WER) — https://github.com/dev-ahmedhany/whisper-arabic-dialects
- Google USM — https://sites.research.google/usm/ , arXiv:2303.01037
- Hallucination harms — arXiv:2402.08021 (FAccT'24); doi:10.1145/3805689.3812320 (FAccT'26)
- AssemblyAI docs (models, languages, diarization, speaker identification) — https://www.assemblyai.com/docs/getting-started/models , https://www.assemblyai.com/docs/pre-recorded-audio/label-speakers , https://www.assemblyai.com/docs/speech-understanding/speaker-identification