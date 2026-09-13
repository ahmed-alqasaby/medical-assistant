# Corpus Inventory

The retrieval corpus and the Extended-track prescription image set. All sources
are **permissive-licensed** (Kaggle-declared) so the project's quality × license
discipline holds for the data layer too. Raw data lives under this directory,
which is **gitignored**; only this inventory and `registry.json` (the
machine-readable manifest with per-file sha256 hashes) are tracked.

| id | Source (Kaggle) | License | Approx. size | Intended use |
|----|-----------------|---------|--------------|--------------|
| `ar` | `yassinabdulmahdi/arabic-medical-q-and-a-dataset` — Arabic Medical Q&A (87,930 Q&A) | MIT | ~82 MB | Arabic-primary retrieval corpus for the middle layer (brief / Q&A / lens) |
| `en` | `gpreda/medquad` — MedQuAD (validated medical Q&A) | Apache 2.0 | ~23 MB | English documents that coexist with the Arabic corpus |
| `rx` | `nooralzoghby/bilingual-medical-prescriptions-arabic-and-english` — 561 handwritten/printed bilingual prescription images + structured annotations | MIT | ~101 MB | Extended-track CV: prescription photo → drug/dose/frequency (seed of ARCH-7) |
| `rx_ocr` | `nadaarfaoui/ocr-processed-handwritten-prescriptions` — OCR-processed handwritten prescriptions | Apache 2.0 | ~16 MB | OCR result text for prescription extraction evaluation |

## Why these

- The product is Arabic-primary with English-permitted and Latin drug names
  preserved (spec §5 / user story 18). The `ar` + `en` pair gives that: Arabic
  medical Q&A as the primary retrievable text, English Q&A as the companion.
- The bilingual `rx` source matches the real world the product targets — a
  prescription with Arabic and English/Latin script on the same ticket (user
  story 11). Its structured annotations (medication / dosage / frequency) are
  the exact slots the prescription pipeline must extract (spec §2 step 4).
- `rx_ocr` is the pre-OCR'd companion, useful to compare raw-image OCR against
  the processed baseline during the Extended-track eval.
- All four are declared permissive (MIT / Apache 2.0) on Kaggle, consistent
  with spec risk register R2 (permissive-license scarcity → joint
  quality × license selection).

## How to fetch

```bash
uv run scripts/fetch_corpus.py --all      # download + stamp sha256 into registry.json
uv run scripts/check_corpus.py           # verify present, non-empty, hash-matching, Arabic-primary
```

Each source downloads to `data/corpus/<id>/`. `registry.json`'s `sha256` field
is stamped by the fetch script; `check_corpus.py` fails on any missing, empty,
tampered, or language-contract-violating source.

## Attribution

- Arabic Medical Q&A Dataset — compiled from a public Arabic medical site by
  the dataset author on Kaggle; MIT.
- MedQuAD — NLM/NIH-derived; Apache 2.0 on Kaggle.
- Bilingual medical prescriptions — author-provided annotated prescription
  images; MIT.
- OCR-Processed Handwritten Prescriptions — derived from the *Doctor's
  Handwritten Prescription BD* dataset; Apache 2.0.

*Retrieval results must always cite the source document; this inventory is
what makes that attribution reproducible.*