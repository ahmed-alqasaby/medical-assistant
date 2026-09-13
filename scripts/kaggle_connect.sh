#!/usr/bin/env bash
set -euo pipefail
# medical-assistant::kaggle_connect — push corpus dataset + RAG notebook to Kaggle.
# The ONLY manual step in the whole LLM setup: your API token
#   (a) ~/.kaggle/access_token  (KGAT bearer token — modern CLI), or
#   (b) ~/.kaggle/kaggle.json   (Kaggle -> avatar -> Settings -> API -> Create New Token).
# The CLI stores auth in ~/.kaggle (KAGGLE_CONFIG_DIR).

R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 0) locate the kaggle CLI
if command -v kaggle >/dev/null 2>&1; then CLI=kaggle
elif [ -x "$R/.venv/bin/kaggle" ]; then CLI="$R/.venv/bin/kaggle"
elif [ -x "$HOME/.local/bin/kaggle" ]; then CLI="$HOME/.local/bin/kaggle"
else
  echo "installing kaggle CLI (user-space)..." >&2
  python3 -m pip install --quiet --user "kaggle==2.2.4"
  CLI="$HOME/.local/bin/kaggle"
fi
echo "kaggle CLI: $CLI"

# 1) auth check (either credential file is fine; CLI auto-selects)
kag=~/.kaggle
if [ ! -f "$kag/access_token" ] && [ ! -f "$kag/kaggle.json" ]; then
  echo "BLOCKED: no ~/.kaggle/access_token or ~/.kaggle/kaggle.json" >&2
  echo "1) open https://www.kaggle.com/settings -> API -> token" >&2
  echo "2a) 'Bearer' mode: save the token value to ~/.kaggle/access_token (chmod 600)" >&2
  echo "2b) or save the downloaded kaggle.json at ~/.kaggle/kaggle.json (chmod 600)" >&2
  echo "3) re-run this script" >&2
  exit 2
fi
chmod 600 "$kag"/* 2>/dev/null || true
"$CLI" quota | head -3

if [ ! -d "$R/data/corpus" ]; then echo "no corpus at $R/data/corpus — build it first" >&2; exit 3; fi

# 2) corpus -> Kaggle dataset (needs a dataset-metadata.json with the owner slug)
OWNER="$("$CLI" config view | awk '/username:/{print $2}')"
SLUG="medical-assistant-corpus"
STAGE="$R/data/kaggle_dataset"
mkdir -p "$STAGE"
cp -RN "$R/data/corpus/." "$STAGE/" 2>/dev/null || true
cat > "$STAGE/dataset-metadata.json" <<JSON
{
  "id": "${OWNER}/${SLUG}",
  "title": "medical-assistant corpus",
  "subtitle": "Arabic medical Q&A (bge-m3 eval fixtures) + English MedQuAD + prescription OCR",
  "isPrivate": true,
  "licenses": [{ "name": "other" }]
}
JSON
"$CLI" datasets create -p "$STAGE" >/dev/null 2>&1 || "$CLI" datasets version -p "$STAGE" -m "corpus refresh" 2>&1 | tail -4

# 3) push notebook (M2) — notebook file is committed in notebooks/
"$CLI" kernels push "$R/notebooks/kaggle_rag_pipeline.ipynb" 2>&1 | tail -8
echo "DONE — run the notebook at https://www.kaggle.com/code/ ... in 'Your Work'"
echo "  attach dataset:  ${OWNER}/${SLUG}"
echo "  after run, download /kaggle/working/vector_store/ -> REPO_ROOT/data/vector_store/"
echo "  backend then serves REPO_ROOT/data/vector_store/chroma at startup."