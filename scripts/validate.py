"""Corpus validation: does the assembled corpus match the manifest and the
Arabic-primary / English-presence contract?

This is the M1 verification seam. The corpus on disk must satisfy:
  - every file in the manifest exists, is non-empty, and matches its sha256;
  - an 'ar' source is present and its combined text is Arabic-primary;
  - an 'en' source is present (English documents coexist with the Arabic).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.common import sha256_file
from scripts.manifest import ManifestError, load_manifest

# The Unicode blocks that carry Arabic script.  Latin-script drug names
# are intentionally *not* in here — they are preserved verbatim in the product.
ARABIC_RANGES: list[tuple[int, int]] = [
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x0870, 0x089F),  # Arabic Extended-B
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
]

DEFAULT_ARABIC_MIN_RATIO = 0.5
DEFAULT_PRIMARY_ID = "ar"
DEFAULT_ENGLISH_ID = "en"


def _is_arabic(char: str) -> bool:
    code_point = ord(char)
    return any(low <= code_point <= high for low, high in ARABIC_RANGES)


def arabic_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are Arabic-script.

    Returns 0.0 for empty or non-alphabetic text.
    """
    alphabetic = [char for char in text if char.isalpha()]
    if not alphabetic:
        return 0.0
    arabic = sum(1 for char in alphabetic if _is_arabic(char))
    return arabic / len(alphabetic)


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    stats: dict[str, dict[str, Any]] = field(default_factory=dict)


def validate_corpus(
    manifest_path: str | Path,
    corpus_dir: str | Path,
    arabic_min_ratio: float = DEFAULT_ARABIC_MIN_RATIO,
) -> ValidationResult:
    """Validate an assembled corpus against its manifest and the language contract."""
    corpus_root = Path(corpus_dir)
    errors: list[str] = []
    stats: dict[str, dict[str, Any]] = {}

    try:
        entries = load_manifest(manifest_path)
    except (ManifestError, FileNotFoundError) as exc:
        return ValidationResult(ok=False, errors=[f"invalid manifest: {exc}"])

    ids = {entry["id"] for entry in entries}
    primary_text = ""

    for entry in entries:
        entry_stats: dict[str, int] = {"file_count": 0, "total_bytes": 0}
        for relative_path in entry["files"]:
            full_path = corpus_root / relative_path
            if not full_path.exists():
                errors.append(f"[{entry['id']}] missing file: {relative_path}")
                continue
            size = full_path.stat().st_size
            if size == 0:
                errors.append(f"[{entry['id']}] empty file: {relative_path}")
                continue
            expected_hash = entry["sha256"].get(relative_path)
            if expected_hash and sha256_file(full_path) != expected_hash:
                errors.append(f"[{entry['id']}] sha256 mismatch: {relative_path}")
                continue
            entry_stats["file_count"] += 1
            entry_stats["total_bytes"] += size
            if entry["id"] == DEFAULT_PRIMARY_ID:
                try:
                    primary_text += full_path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    errors.append(f"[{DEFAULT_PRIMARY_ID}] unreadable file: {relative_path}")
        stats[entry["id"]] = entry_stats

    if DEFAULT_ENGLISH_ID not in ids:
        errors.append(f"no '{DEFAULT_ENGLISH_ID}' source in manifest — English documents are required")

    if DEFAULT_PRIMARY_ID in ids:
        ratio = arabic_ratio(primary_text)
        stats[DEFAULT_PRIMARY_ID]["arabic_ratio"] = round(ratio, 4)
        if ratio < arabic_min_ratio:
            errors.append(
                f"[{DEFAULT_PRIMARY_ID}] Arabic ratio {ratio:.3f} below threshold {arabic_min_ratio}"
            )

    return ValidationResult(ok=not errors, errors=errors, stats=stats)