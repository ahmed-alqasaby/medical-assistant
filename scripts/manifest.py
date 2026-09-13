"""Kaggle-sourced corpus manifest: load and schema-check.

The manifest (data/corpus/registry.json) is a JSON array of source entries.
Each entry declares where the data came from, under what licence, what it is
used for, and — once fetched — the exact files with their sha256 hashes so
that retrieval results stay attributable to a reproducible corpus.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_KEYS = {"id", "source", "license", "intended_use", "files", "sha256"}


class ManifestError(ValueError):
    """The manifest is malformed or violates the schema."""


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load and schema-check a manifest file.

    Raises FileNotFoundError if missing, ManifestError on invalid shape.
    """
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ManifestError("manifest must be a JSON array of source entries")
    for entry in data:
        if not isinstance(entry, dict):
            raise ManifestError(f"manifest entries must be objects, got {type(entry).__name__}")
        missing = REQUIRED_KEYS - set(entry)
        if missing:
            raise ManifestError(
                f"entry '{entry.get('id', '<no id>')}' missing required keys: {sorted(missing)}"
            )
        if not isinstance(entry["files"], list) or not isinstance(entry["sha256"], dict):
            raise ManifestError(f"entry '{entry.get('id')}': 'files' must be a list and 'sha256' a dict")
    return data