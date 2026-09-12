# Embedding Model Selection: Mixed Arabic + English Medical Text Retrieval

**Ticket:** Research — concept-build embedding model for a medical RAG system where documents are primarily Arabic with Latin-script drug names and English clinical terminology (code-switched). Public deployment must be on-premise-capable.  
**Date:** Sep 12 2026  
**Status:** Research complete; recommendation below.

---

## TL;DR Recommendation

| Role | Model | Why |
|------|-------|-----|
| **Concept-build pick** | `BAAI/bge-m3` (open-source, MIT) | Hybrid dense+sparse+multivector handles Latin drug names via lexical matching *and* Arabic semantics; best-verifiable open-source Arabic retrieval (MIRACL-ar nDCG@10 0.801 hybrid, 0.829 with bge-reranker-v2-m3); 8192-token context for long clinical notes; 100+ languages; runs fully on-prem via vLLM/TEI/Ollama. |
| **On-prem runner-up** | `intfloat/multilingual-e5-large-instruct` (MIT) | Best publicly available model overall on MMTEB 500-task multilingual benchmark (ICLR 2025); strong but weaker than Arabic-centric on Arabic-specific tasks; 512-token limit is a constraint. |
| **Arabic-quality ceiling (test alongside)** | `Swan-Large` (ArMistral-based; NAACL 2025) | ArabicMTEB medical-domain score 81.64 — best published Arabic medical embedding quality; test English + cross-lingual retrieval quality before committing. |
| **API best (if cloud permitted)** | `Cohere embed-multilingual-v3.0` or `OpenAI text-embedding-3-large` | Strong ArabicMTEB domain scores (Swan paper: Cohere-v3 domain avg 73.76; OpenAI-3-large 82.20); Cohere offers on-prem air-gapped deployment. |

**One-line evidence/confidence note:** BGE-M3 hybrid is the strongest open Arabic retriever in a reproducible MIRACL-arabic evaluation (arabic-retrieval-lab, 2026: 0.829 nDCG@10), has MIT license, runs on-prem; Swan-Large has best published Arabic medical embedding quality (81.64 on ArabicMTEB medical domain, Swan et al., NAACL 2025); multilingual-e5-large-instruct is best multilingual overall (MMTEB, ICLR 2025, 63.2 avg). **Confidence: High for BGE-M3 as safe default; medium for Swan (limited English evidence); high for mE5-instruct as multilingual baseline.** Key gap: no public benchmark covers code-switched Arabic+English medical retrieval specifically.

---

## ⚠️ ADDENDUM (2026-09-12) — decision confirmed; pipeline context added

**Decision (ticket #5):** **bge-m3 confirmed** as the concept-build embedder for the Arabic-primary /
English-required pipeline. Runner-up `multilingual-e5-large-instruct`; Arabic-ceiling test candidate
`Swan-Large` deferred (retain §6 eval-set plan).

**Cohere Transcribe Arabic does NOT replace a text embedder.** It is a Conformer *acoustic* encoder +
decoder trained by cross-entropy on output tokens — no contrastive/siamese objective, no pooled
semantic embedding head. Its hidden states measure audio similarity, not document relevance. The RAG
embedder is a separate text model (bge-m3) operating on ASR text.

**Normalization contract (make it the SAME on index & query sides):** ASR output is undiacritized, so
the retrieval layer MUST strip and normalize consistently — remove diacritics, unify alef variants
(أ/إ/آ → ا), teh-marbuta (ة → ه), yaeh (ى → ي), strip tatweel/kashida. Applied at index time (turns,
prescription records, lab results, captions-if-later) and at query time (user questions). Single
cheapest accuracy lever for Arabic embedding (§3, diacritics).

**RAG paradigm (feeds #11):** layered architecture, not one paradigm:
1. **Structured patient memory** — relational fields (sessions, confirmed prescriptions, lab results,
   access grants, scan records) = the "graph without a graph DB".
2. **Hybrid RAG** — bge-m3 dense+sparse over turn-level chunks, top-k + `bge-reranker-v2-m3` cross-encoder.
3. **Thin agentic orchestration** — tools = query patient structured fields / vector-search patient
   turns; cite every claim to `(speaker, timestamp, doc-id)`; refuse anything outside the patient's
   granted memory.
4. **GraphRAG deferred** to a later version, but the data model must keep relations explicit (typed
   edges session→prescription→condition/lab) so a KG is derivable without re-modeling. Cross-patient
   graph queries remain out of scope (private per-patient retrieval).

**Code-switch reality:** drug names in Latin script inside Arabic sentences match via bge-m3's sparse
head; the code-switch internal eval set (50–100 queries, §6) is a first-order milestone — it is the only
evidence for this exact slice, no public benchmark exists.

---

---

## 1. Candidate Model Landscape

### 1.1 Open-Source / Local

| Model | Params | Dims | Max Tokens | License | Arabic Support | Key Evidence |
|-------|--------|------|------------|---------|----------------|--------------|
| **bge-m3** (BAAI) | 568M | 1024 | 8192 | MIT | 100+ langs incl. Arabic | MIRACL SOTA at release; hybrid dense+sparse+multivector; MIRACL-ar dev: 0.801 dense-only, 0.829 with bge-reranker-v2-m3 blend ([arabic-retrieval-lab](https://github.com/yinli-systems/arabic-retrieval-lab/blob/main/TECHNICAL_REPORT.md)). ruMTEB: best <1B on Russian IR ([arXiv:2504.12879](https://arxiv.org/pdf/2504.12879)). |
| **multilingual-e5-large** | 560M | 1024 | 512 | MIT | 94 langs incl. Arabic | MMTEB rank 4 overall; MIRACL avg competitive. `[HF model card](https://huggingface.co/intfloat/multilingual-e5-large)` |
| **multilingual-e5-large-instruct** | 560M | 1024 | 512 | MIT | 93 langs incl. Arabic | **MMTEB best publicly available model overall** (ICLR 2025, 63.2 avg across 500 tasks, 250+ langs); in truly low-resource settings, consistently outperforms 7B LLM-based embedders ([MMTEB, ICLR 2025](https://openreview.net/forum?id=zl3pfz4VCV)). |
| **jina-embeddings-v3** | 570M | 1024 | 8192 | **cc-by-nc-4.0** ⚠️ | 89 langs (tuned 30 incl. Arabic) | "Best multilingual model <1B on MTEB" per vendor (Sep 2024); beats multilingual-e5-large-instruct on all multilingual tasks ([arXiv:2409.10173](https://arxiv.org/pdf/2409.10173)). **Non-commercial license — not usable for public deployment without commercial license.** |
| **jina-embeddings-v2-base-multilingual** | ~161M | ? | 8192 | Apache-2.0 | Bilingual variants (zh, de, code); multilingual via API | Apache-2.0 for commercial use; but smaller, less strong than v3 or bge-m3. |
| **Swan-Small** (ARBERTv2) | ~133M | 1024 | 512 | Research | Arabic-focused (MSA+dialects) | ArabicMTEB: outperforms multilingual-e5-base on Arabic tasks. Domain: medical 70.86 ([Swan/ArabicMTEB, NAACL 2025](https://aclanthology.org/2025.findings-naacl.263)). |
| **Swan-Large** (ArMistral) | ~7B est. | 1024 | 8192 | Research | Arabic-focused (MSA+dialects, 12.5M train examples incl. MIRACL, MMARCO) | **ArabicMTEB medical: 81.64** (best); domain avg 82.49 surpassing OpenAI-3-large 82.20. Cross-lingual Arabic→English trained on MMARCO/XOR-TyDi. ([Swan, NAACL 2025](https://aclanthology.org/2025.findings-naacl.263)) |
| **Arabic-Triplet-Matryoshka-V2** | ? | ? | ? | ? | Arabic STS/retrieval | Used in QIAS 2025 Arabic QA retrieval; **"weaker in MIRACL retrieval setup"** vs bge-m3 ([arabic-retrieval-lab](https://github.com/yinli-systems/arabic-retrieval-lab/blob/main/TECHNICAL_REPORT.md)). |

### 1.2 API-Proprietary (cloud or on-prem deployment)

| Model | Context | Dims | License | On-Prem | Key Evidence |
|-------|---------|------|---------|---------|--------------|
| **Cohere embed-multilingual-v3.0** | 512 tok | 1024 | Proprietary | **Yes** — VPC, air-gapped on-prem ([Cohere private deployments](https://cohere.com/private-deployments)) | 100+ langs incl. Arabic; ArabicMTEB domain avg 73.76 (below OpenAI-3-large) ([Swan paper](https://aclanthology.org/2025.findings-naacl.263)). Strong multilingual general-purpose; good RAG pipeline ecosystem. |
| **Cohere embed-v4** | 128k tok | 256–1536 | Proprietary | **Yes** (same on-prem options) | "Significant relative improvement over Embed 3" per Cohere marketing. Multimodal. Arabic in 100+ langs. |
| **OpenAI text-embedding-3-large** | 8k tok | 3072 | Proprietary | **No direct on-prem** (Azure OpenAI managed service only) | MIRACL avg 54.9; MTEB avg 64.6 ([OpenAI announcement](https://openai.com/index/new-embedding-models-and-api-updates)); ArabicMTEB domain avg 82.20 (including medical) per Swan paper. |
| **Voyage (voyage-3.5 / voyage-4-large)** | 32k tok | 256–2048 | Proprietary | "Deploy anywhere" (claimed); **voyage-4-nano open-weight on HF** (2026) | Vendor claims: voyage-multilingual-2 beat OpenAI v3-large/Cohere/mE5 by 5.6% avg across 27 languages ([Voyage blog](https://blog.voyageai.com/2024/06/10/voyage-multilingual-2-multilingual-embedding-model/)). **Arabic NOT in their primary evaluated languages** (eval covered: en, fr, de, ja, es, ko, bn, pt, ru, + "OTHER"). Arabic-specific evidence thin. |
| **Google Gecko / text-embedding-004** | 2k tok | 768 | Proprietary (Vertex AI) | No | gecko-multilingual-1b: MIRACL avg 56.2 (beats OpenAI 54.9) ([arXiv:2403.20327](https://arxiv.org/pdf/2403.20327)); mE5-large within 0.003 nDCG at 31ms vs GE2 at 231ms ([arXiv:2605.23618](https://arxiv.org/abs/2605.23618)). |

### 1.3 Arabic-Specific Context

| Model | Year | Type | Relevance |
|-------|------|------|-----------|
| AraBERT / AraBERTv2 | 2020/2022 | BERT-base (110M), MSA-pretrained | Base for Swan-Small; strong Arabic NLU but not optimized for sentence embeddings/retrieval. |
| ARBERT / MARBERT | 2021 | BERT-base (163M), MSA / dialectal | ARBERTv2 is Swan-Small base; MARBERT for dialects (tweets). |
| NeoAraBERT | 2026 | NeoBERT architecture, diacritics-aware tokenization | Top Arabic embeddings on 23 tasks including "Muradif" synonym quality task ([ACL 2026 Findings](https://aclanthology.org/2026.findings-acl.1293)); limited retrieval evaluation. |
| GigaBERT | 2020 | BERT + code-switching pretraining | Explicitly designed for Arabic-English mixed text; limited downstream embedding evaluation. |

---

## 2. MTEB / Arabic MTEB Benchmark Evidence

### What the benchmarks say

| Benchmark | Languages | Tasks | What it tells us |
|-----------|-----------|-------|------------------|
| **MTEB v1** (2022) | 112 langs; retrieval English-only | 58 datasets | No Arabic retrieval data. |
| **MMTEB** (ICLR 2025) | 250+ langs | 500+ tasks | `multilingual-e5-large-instruct` is best publicly available model overall; in truly low-resource settings, the XLM-R-based model (560M) consistently beats 7B LLM-based encoders. ([ICLR 2025](https://openreview.net/forum?id=zl3pfz4VCV)) |
| **ArabicMTEB** (NAACL 2025) | Arabic (MSA + dialects) | 94 datasets, 8 tasks incl. 35 retrieval | Swan-Large > OpenAI-3-large > Cohere-v3 > multilingual-e5-large. Domain-specific: medical, news, Wikipedia. ([Swan et al., NAACL 2025](https://aclanthology.org/2025.findings-naacl.263)) |
| **MIRACL** (2023) | 18 langs incl. Arabic | Monolingual retrieval, human-annotated | OpenAI te3-large: 54.9 avg; gecko-multilingual-1b: 56.2; bge-m3 hybrid: **0.801** (Arabic dev). Arabic one of the best-resourced multilingual retrieval benchmarks. |
| **ruMTEB** | Russian | 7 task clusters, 23 datasets | "Best models <1B: BGE-M3 and Multilingual E5-large" ([arXiv:2504.12879](https://arxiv.org/pdf/2504.12879)). Relevant as analog for non-European-script multilingual. |

### Where evidence is thin

- **Voyage Arabic quality:** Not in primary evaluation language lists. Vendor claims strong general multilingual quality; no public Arabic-specific numbers.
- **OpenAI Arabic:** MIRACL avg reported (54.9); per-Arabic-language breakdown not published by OpenAI. ArabicMTEB domain scores available only via Swan et al.'s independent evaluation.
- **Cohere Arabic:** No public Arabic-specific benchmark numbers beyond Swan et al.'s independent evaluation.
- **Code-switched Arabic/English medical retrieval:** No public benchmark exists. This is the single largest blind spot in current evaluation.
- **Arabic-specific models (Swan, AraModernBERT):** Swan evaluated on ArabicMTEB including medical domain but not specifically on code-switched Arabic+English medical text. English generalization of Swan-Large untested in published benchmarks.

---

## 3. Dirty-Realism Factors

### Code-switching (Arabic + English within a sentence)

Arabic medical text routinely mixes Arabic with English/Latin clinical terminology. This is linguistically documented as common in healthcare (COLING 2025 survey: [arXiv:2501.13419](https://arxiv.org/pdf/2501.13419); "CSW in Medical and Educational Domains" section).

**What helps:**
- **bge-m3's sparse lexical head** directly handles Latin drug names (e.g., "دواء Paracetamol 500mg") by computing token-level lexical similarity — the Latin tokens match in the sparse representation even if the Arabic context is unseen.
- **multilingual-e5-large(-instruct)** trained on XLM-R (100-language multilingual) with mixed-language data should handle code-switch at the sentence level, but has no explicit lexical/fallback mechanism.
- **Swan-Large** trained on cross-lingual Arabic→English data (MMARCO, XOR-TyDi) but primarily Arabic-pretrained (ArMistral); English quality within mixed sentences is a question mark.

**Practical mitigation:** Hybrid retrieval (dense + BM25) is the single most effective approach for code-switched text. bge-m3 supports this natively out of the box.

### Drug names in Latin script inside Arabic sentences

The critical issue: a doctor writes "المريض يتناول Metformin 500mg مرتين يومياً". The embedding model must link "Metformin" (Latin) to retrieved passages that contain "ميتفورمين" (Arabic transliteration) or the Latin form directly.

**What helps:**
- bge-m3 sparse retrieval: Latin tokens matched lexically regardless of surrounding Arabic.
- Hybrid BM25 baseline: handles exact-match drug names well.
- ArabicMTEB's cross-lingual retrieval task (Arabic queries → English docs via mMarco) is the closest proxy to this pattern; Swan and multilingual-e5-large were evaluated on it.

### Diacritics

Medical Arabic text is almost always undiacritized. Recent evidence:

- **ABJAD-NLP 2026 (Persian/Arabic analog):** "Removing diacritics, zero-width non-joiners, and normalizing Yeh/Kaf all did not have an effect on F1" for NER — diacritics removal is safe and may simplify.
- **NeoAraBERT (ACL 2026):** Diacritics-aware tokenization improved Arabic embeddings on certain intrinsic tasks, but the evaluation is STS/classification, not retrieval.
- **Practical recommendation:** Strip diacritics consistently at indexing and query time (both sides must match). Normalize Arabic text: remove tatweel (kashida), normalize alef variants (أ/إ/آ → ا), normalize teh marbuta (ة → ه), normalize yaeh (ى → ي). Arabic-specific preprocessing tools (AraBERT's preprocessing pipeline / AraNLP) handle this.

---

## 4. Practical Deployment Considerations

### On-prem capability (required for public deployment)

| Model | On-prem Status | How |
|-------|---------------|-----|
| bge-m3 | ✅ Fully on-prem | Sentence-transformers, TEI (Hugging Face Text Embeddings Inference), vLLM, Ollama |
| multilingual-e5-large(-instruct) | ✅ Fully on-prem | Same stack as bge-m3; also ONNX, OpenVINO, TensorRT |
| Swan-Large | ⚠️ Uncertain | Researchers state "models and benchmarks made publicly accessible for research"; ArMistral-based; weight download available on GitHub; commercial license terms unclear — must verify |
| Cohere embed-v3/v4 | ✅ On-prem via private deployment | Air-gapped, VPC, Model Vault; Cohere-managed but runs in your infra ([Cohere deployments](https://cohere.com/private-deployments)) |
| OpenAI text-embedding-3-large | ⚠️ Azure OpenAI only | Managed Azure service; no on-prem/self-hosted option |
| Voyage | ⚠️ API + voyage-4-nano open-weight | voyage-4-nano on HF (2026) for on-prem; other Voyage models API-only |
| jina-embeddings-v3 | ❌ License-blocked | CC-BY-NC for open weights; commercial use requires separate license |

### GPU requirements (concept build, single-GPU)

| Model | Approx VRAM | Notes |
|-------|-------------|-------|
| bge-m3 | ~3–4 GB (FP16) | XLM-R backbone; 8192 tok; 568M |
| multilingual-e5-large | ~2–3 GB (FP16) | 512 tok; 560M |
| multilingual-e5-large-instruct | ~2–3 GB (FP16) | Same base; instruction-tuned |
| Swan-Large | ~14–16 GB (FP16, ArMistral-based) | 7B class; larger infra needed |
| bge-reranker-v2-m3 (reranker) | ~1–2 GB | Recommended add-on for bge-m3 pipeline |

---

## 5. Synthesis: Why BGE-M3 for the Concept Build

1. **Best verifiable Arabic retrieval among open models:** MIRACL-arabic dev nDCG@10 = 0.829 with bge-reranker-v2-m3 score blend (arabic-retrieval-lab, reproducible). No other open model has published stronger standalone Arabic retrieval numbers in a documented evaluation setup.

2. **Hybrid sparse+dense is essential for code-switched medical text:** The sparse head provides lexical matching for Latin drug names and English terms inside Arabic sentences — a critical requirement no other candidate handles as naturally.

3. **8192-token context:** Clinical notes, prescriptions, and session transcripts can be long. bge-m3 supports full-document embeddings (or long-chunk) without splitting; mE5 is limited to 512 tokens.

4. **Proven in multilingual low-resource settings:** ruMTEB shows bge-m3 alongside mE5 as best <1B model for Russian (another non-Latin-script language with similar tokenization challenges). MIRACL covers 18 languages including Arabic with human annotations.

5. **Fully on-prem, MIT license, production-proven stack:** Runs via TEI, vLLM, Ollama, sentence-transformers. No license risk.

6. **Pair with bge-reranker-v2-m3 for production quality:** The cross-encoder reranker adds 1.6–2.7 nDCG points over first-stage retrieval alone, a well-understood and low-risk pipeline addition.

### When to reconsider

- **If Arabic-only quality ceiling matters most and English code-switch is secondary:** Test Swan-Large (medical domain 81.64 vs bge-m3's untested medical embedding quality — need to build a small Arabic medical retrieval eval set to compare).
- **If multilingual-e5-large-instruct is required for other non-Arabic languages in the system:** It's the MMTEB overall winner; pair with bge-m3 sparse retrieval in a hybrid setup for Arabic.
- **If cloud is acceptable:** Cohere embed-v3/v4 offers on-prem private deployment with strong multilingual quality and an established RAG ecosystem.

---

## 6. Recommended Next Step

Build a small Arabic medical retrieval eval set (50–100 queries from actual Arabic clinical/medical text with Latin drug names mixed in) and measure nDCG@10 for bge-m3 (dense-only, hybrid, and hybrid+reranker) vs multilingual-e5-large-instruct vs Swan-Large. This will provide the only reliable evidence for this specific use case — no public benchmark covers it.

---

## Sources

| Reference | Citation |
|-----------|----------|
| BGE-M3 paper | [arXiv:2402.03216](https://arxiv.org/pdf/2402.03216) |
| BGE-M3 HF model card | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3/blob/main/README.md) |
| Arabic retrieval lab BGE-M3 results | [TECHNICAL_REPORT.md](https://github.com/yinli-systems/arabic-retrieval-lab/blob/main/TECHNICAL_REPORT.md) |
| multilingual-e5 paper | [arXiv:2402.05672](https://arxiv.org/html/2402.05672) |
| multilingual-e5-large HF model card | [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large/blob/main/README.md) |
| MMTEB (ICLR 2025) | [MMTEB, ICLR 2025](https://openreview.net/forum?id=zl3pfz4VCV) |
| Swan + ArabicMTEB (NAACL 2025) | [aclanthology.org/2025.findings-naacl.263](https://aclanthology.org/2025.findings-naacl.263) |
| Swan arXiv | [arXiv:2411.01192](https://arxiv.org/abs/2411.01192) |
| OpenAI text-embedding-3-large | [OpenAI announcement](https://openai.com/index/new-embedding-models-and-api-updates); [API docs](https://developers.openai.com/api/docs/models/text-embedding-3-large) |
| Cohere embed-v3 / on-prem | [Cohere private deployments](https://cohere.com/private-deployments); [Cohere embed-v3 docs](https://docs.cohere.com/docs/cohere-embed) |
| Jina embeddings v3 | [arXiv:2409.10173](https://arxiv.org/pdf/2409.10173); [HF jinaai/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3) (cc-by-nc-4.0) |
| Jina embeddings v2 | [arXiv:2310.19923](https://arxiv.org/html/2310.19923v4) |
| Voyage multilingual-2 | [Voyage blog](https://blog.voyageai.com/2024/06/10/voyage-multilingual-2-multilingual-embedding-model/); [Voyage docs](https://docs.voyageai.com/docs/embeddings) |
| Google Gecko | [arXiv:2403.20327](https://arxiv.org/pdf/2403.20327) |
| Google Embeddings 2 vs open-source | [arXiv:2605.23618](https://arxiv.org/abs/2605.23618) |
| ruMTEB (Russian benchmark) | [arXiv:2504.12879](https://arxiv.org/pdf/2504.12879) |
| MIRACL | [ACL Anthology / TACL](https://aclanthology.org/2023.tacl-1.63.pdf); [arXiv:2210.09984](https://arxiv.org/pdf/2210.09984) |
| Code-switched Arabic NLP survey | [COLING 2025 / arXiv:2501.13419](https://arxiv.org/pdf/2501.13419) |
| Diacritics in Arabic NLP | [EMNLP 2025](https://aclanthology.org/2025.emnlp-main.846); [EACL 2026](https://aclanthology.org/2026.findings-eacl.22.pdf); [ABJAD-NLP 2026](https://aclanthology.org/volumes/2026.abjadnlp-1) |
| NeoAraBERT (diacritics-aware embeddings) | [ACL 2026 Findings](https://aclanthology.org/2026.findings-acl.1293) |
| AraBERT | [arXiv:2003.00104](https://arxiv.org/pdf/2003.00104) |
| Arabic QIAS 2025 hybrid retrieval | [ACL Anthology 2025 ArabicNLP](https://aclanthology.org/2025.arabicnlp-sharedtasks.124.pdf) |
| LaBSE poor retrieval | [arXiv:2605.23618](https://arxiv.org/abs/2605.23618) — 0.188 nDCG@10 avg on BEIR, below dedicated retrieval models |
