"""Validate the assembled corpus against the registry (M1 verification seam).

CLI wrapper over scripts.validate.validate_corpus.  Exits non-zero if any
source is missing, empty, corrupted, or violates the Arabic-primary /
English-presence contract.

Usage:
    uv run scripts/check_corpus.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common import corpus_defaults

DEFAULT_CORPUS_DIR, DEFAULT_MANIFEST = corpus_defaults()


def main() -> int:
    # imported lazily so the module imports cleanly even without a venv
    from scripts.validate import validate_corpus

    result = validate_corpus(DEFAULT_MANIFEST, DEFAULT_CORPUS_DIR)
    for entry, stats in sorted(result.stats.items()):
        ratio = f", arabic_ratio={stats.get('arabic_ratio')}" if "arabic_ratio" in stats else ""
        print(f"  [{entry}] files={stats['file_count']} bytes={stats['total_bytes']}{ratio}")
    if not result.ok:
        print("corpus validation FAILED:")
        for error in result.errors:
            print(f"  - {error}")
        return 1
    print("corpus validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())