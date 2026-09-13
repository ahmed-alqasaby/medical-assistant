"""Shared helpers for the corpus scripts.

Single home for the pieces three corpus scripts need and would otherwise
copy-paste: hashing a file, walking a tree for non-junk files, and the
default corpus locations.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file, streaming so hashing stays memory-safe."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_extraction_junk(path: Path) -> bool:
    """True for archive cruft that is never a corpus artifact (Apple junk)."""
    return "__MACOSX" in path.parts or ".DS_Store" in path.parts


def iter_files(root: Path) -> list[Path]:
    """Every non-junk file under root, sorted for deterministic ordering."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not is_extraction_junk(path)
    )


def corpus_defaults() -> tuple[Path, Path]:
    """The canonical corpus dir and manifest paths for this repo."""
    corpus_dir = REPO_ROOT / "data" / "corpus"
    return corpus_dir, corpus_dir / "registry.json"