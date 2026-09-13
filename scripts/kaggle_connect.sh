#!/usr/bin/env bash
set -euo pipefail
# medical-assistant::kaggle_connect — push corpus dataset + RAG notebook to Kaggle.
# The ONLY manual step in the whole LLM setup: your API token at ~/.kaggle/kaggle.json
#   (Kaggle -> your avatar -> Settings -> API -> Create New Token; download & save there).
# Local API token for notebook execution only (print low).

R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAG="~/.kaggle/kaggle.json"
if [ ! -f "$KAG" ]; then
  echo "BLOCKED: no $KAG" >&2
  echo "1) open https://www.kaggle.com/settings -> API -> Create New Token" >&2
  echo "2) place the downloaded file at $KAG  (chmod 600)" >&2
  echo "3) re-run this script" >&2
  exit 2
fi
K=$(cat "$KAG")
if command -v kaggle >/dev/null 2>&1; then CLI=kaggle
elif [ -x "$HOME/.local/bin/kaggle" ]; then CLI="$HOME/.local/bin/kaggle"
else
  echo "installing kaggle CLI (user-space)..." >&2
  python3 -m pip install --user --quiet kaggle
  CLI="$HOME/.local/bin/kaggle"
fi
echo "kaggle CLI: $CLI"

if [ ! -d "$R/data/corpus" ]; then echo "no corpus at $R/data/corpus — build it first" >&2; exit 3; fi

# 1) corpus -> Kaggle dataset (needs kaggle.json creds to authenticate)
echo "> push corpus as dataset medical-assistant-corpus (busybox-style: dataset new -p dir)"
mkdir -p "$R/data/kaggle_dataset" 
cp -R "$R/data/corpus"/* "$R/data/kaggle_dataset/" 2>/dev/null || true
"$CLI" datasets create -p "$R/data/kaggle_dataset" -t "Arabic medical RAG corpus (bge-m3 eval fixtures)" >/dev/null 2>&1 || {
  echo "NOTE: dataset create needs a first run of kaggle CLI to cache creds — push now" >&2
  "$CLI" datasets push "$R/data/kaggle_dataset" 2>&1 | tail -5
}

# 2) push notebook
"$CLI" kernels push "$R/notebooks/kaggle_rag_pipeline.ipynb" 2>&1 | tail -8
echo "DONE — check output at https://www.kaggle.com/code/ ... in 'Your Work'"
