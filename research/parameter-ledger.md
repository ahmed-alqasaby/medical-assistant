# Parameter Ledger — validated vs. measured

*Normative companion to `spec/system-design.md` §9 (R1). Every tunable the v1 architecture depends on is listed here with its status. `[prior]` = a reasonable prior with **zero measured data** behind it; a `[prior]` value is **not a lock** until the owning eval records a value. Status flips to `[measured]` with the value and the eval that produced it.*

*rev. 2026-09-13 — created with the validated-vs-measured review pass (spec §9/R1). The eval-gate program (in-room WER, real-ticket HTR, role-label validation, code-switch retrieval set) is the project's actual risk owner, not any architectural choice.*

## Design tunables

| # | Parameter | Role in pipeline | Status | Owning eval | Recorded value |
|---|---|---|---|---|---|
| 1 | **Enrichment on/off** | whether index-time contextual enrichment (§6) is applied to turns | `[prior]` | code-switch retrieval set (A/B variant) | — |
| 2 | **Enrichment prefix budget** | length budget for the generation-written context prefix per turn | `[prior]` | code-switch retrieval set (A/B) | — |
| 3 | **Chunk boundary = turn** | retrieval + audit unit (speaker·start·end) | **locked** (not a tunable; §6) | — | turn-level |
| 4 | **Retrieval top-k** | candidates passed to the reranker | `[prior]` | code-switch retrieval set | — |
| 5 | **Reranker candidate window** | size of the window into `bge-reranker-v2-m3` | `[prior]` | code-switch retrieval set | — |
| 6 | **Fusion mode** | RRF (default) vs weighted-convex combination across collections | `[prior]` | code-switch retrieval set (A/B) | RRF (tentative) |
| 7 | **Signal weighting by query type** | per-type weights over bge-m3 dense / sparse / late-interaction signals | `[prior]` | code-switch retrieval set (A/B) | — |
| 8 | **CAG frame-packing threshold** | how an active session is packed when it approaches the generation context limit | `[prior]` | token count of the longest real session | — |
| 9 | **Generation context headroom** | % of context budget the brief's long-context synthesis may use | `[prior]` | longest-patient-history measurement | — |
| 10 | **Session token length** | tokens per 45–90 min consult (drives CAG feasibility, §7) | `[prior]` | tokenize N real in-room sessions | 8–25K (estimate) |

## Measurement gates (accuracy — the evals that own the `[prior]`s above)

| Gate | Owner | Target scope | Where results land |
|---|---|---|---|
| **In-room WER** | ASR (Cohere primary, en fallback) | ~20–30 real in-room sessions | this ledger + research/asr.md |
| **HTR on real tickets** | prescription engine choice (Azure DI vs trOCR) | N photo + keyed ground-truth prescriptions | this ledger + research/prescription-ocr.md |
| **Role-label validation** | A/B doctor/patient role pass | ~10 sessions, hand-validated | this ledger |
| **Code-switch retrieval set** | retrieval stack (§6) | 50–100 Arabic+English medical queries | this ledger (rows 1–9) |