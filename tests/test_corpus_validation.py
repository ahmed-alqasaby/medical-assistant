"""Seam A: corpus validation behavior.

Tests validate_corpus(manifest_path, corpus_dir) against a tiny fixture corpus
built in tmp_path.  Deterministic, no network — the manifest and hashes are
constructed programmatically so the test owns the expected values.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.manifest import load_manifest
from scripts.validate import validate_corpus


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_ARABIC_TEXT = (
    "الدكتور وصف للمريض دواء باراسيتامول ٥٠٠ ملغ مرتين يوميا لمدة خمسة أيام "
    "المريض يعاني من صداع شديد منذ ثلاثة أيام也无法理解中医的疗效 "
    "atient reported mild headache; no fever, cough, or sore throat.\n"
)

_ENGLISH_TEXT = (
    "Patient presents with chronic lower back pain, worse on the left side. "
    "Pain radiates to left leg. No saddle anesthesia. Neurological exam intact.\n"
    "Diagnosis: lumbar radiculopathy L4-L5.\n"
)

_RX_TEXT = (
    "Drug: Amoxicillin, Dose: 500mg, Frequency: TID, Duration: 7 days\n"
    "Drug: Paracetamol, Dose: 1g, Frequency: Q6H PRN, Duration: 3 days\n"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _manifest_entries(
    ar_file: str = "q_and_a.csv",
    en_file: str | None = "medquad.csv",
    rx_file: str = "prescriptions.csv",
) -> list[dict]:
    """Return a *skeleton* manifest without hashes (hashes filled by fixture)."""
    entries = [
        {
            "id": "ar",
            "source": "yassinabdulmahdi/arabic-medical-q-and-a-dataset",
            "license": "MIT",
            "intended_use": "Arabic-primary medical Q&A",
            "files": [f"ar/{ar_file}"],
            "sha256": {},
        }
    ]
    if en_file:
        entries.append(
            {
                "id": "en",
                "source": "gpreda/medquad",
                "license": "Apache-2.0",
                "intended_use": "English medical Q&A",
                "files": [f"en/{en_file}"],
                "sha256": {},
            }
        )
    entries.append(
        {
            "id": "rx",
            "source": "nooralzoghby/bilingual-medical-prescriptions-arabic-and-english",
            "license": "MIT",
            "intended_use": "Handwritten bilingual rx images + annotations",
            "files": [f"rx/{rx_file}"],
            "sha256": {},
        }
    )
    return entries


def _write_corpus(root: Path, entries: list[dict], ar_content: str = _ARABIC_TEXT, en_content: str = _ENGLISH_TEXT, rx_content: str = _RX_TEXT) -> None:
    """Write the small fixture corpus and stamp hashes into the manifest entries."""
    ar_path = root / entries[0]["files"][0]
    _write(ar_path, ar_content)
    entries[0]["sha256"][entries[0]["files"][0]] = _sha256(ar_path)

    en_entry = next((e for e in entries if e["id"] == "en"), None)
    if en_entry:
        en_path = root / en_entry["files"][0]
        _write(en_path, en_content)
        en_entry["sha256"][en_entry["files"][0]] = _sha256(en_path)

    rx_entry = next((e for e in entries if e["id"] == "rx"), None)
    if rx_entry:
        rx_path = root / rx_entry["files"][0]
        _write(rx_path, rx_content)
        rx_entry["sha256"][rx_entry["files"][0]] = _sha256(rx_path)


def _write_manifest(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class TestValidMiniCorpus:
    def test_passes_for_valid_mini_corpus(self, tmp_path: Path) -> None:
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        entries = _manifest_entries()
        _write_corpus(corpus_dir, entries)
        manifest_path = tmp_path / "registry.json"
        _write_manifest(manifest_path, entries)

        result = validate_corpus(manifest_path, corpus_dir)

        assert result.ok is True, f"Expected ok=True, got errors: {result.errors}"
        assert result.errors == []
        assert "ar" in result.stats
        assert "en" in result.stats

    def test_stats_report_file_sizes_and_counts(self, tmp_path: Path) -> None:
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        entries = _manifest_entries()
        _write_corpus(corpus_dir, entries)
        manifest_path = tmp_path / "registry.json"
        _write_manifest(manifest_path, entries)

        result = validate_corpus(manifest_path, corpus_dir)

        assert result.stats["ar"]["file_count"] == 1
        assert result.stats["ar"]["total_bytes"] > 0


class TestMissingOrCorruptFile:
    def test_fails_when_file_missing(self, tmp_path: Path) -> None:
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        entries = _manifest_entries()
        _write_corpus(corpus_dir, entries)
        # Remove the rx file after hash is stamped
        rx_path = corpus_dir / entries[2]["files"][0]
        rx_path.unlink()

        manifest_path = tmp_path / "registry.json"
        _write_manifest(manifest_path, entries)

        result = validate_corpus(manifest_path, corpus_dir)

        assert result.ok is False
        assert any("missing" in e.lower() or "not found" in e.lower() for e in result.errors)

    def test_fails_on_hash_mismatch(self, tmp_path: Path) -> None:
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        entries = _manifest_entries()
        _write_corpus(corpus_dir, entries)
        # Tamper with the Arabic file
        ar_path = corpus_dir / entries[0]["files"][0]
        ar_path.write_text("tampered content", encoding="utf-8")

        manifest_path = tmp_path / "registry.json"
        _write_manifest(manifest_path, entries)

        result = validate_corpus(manifest_path, corpus_dir)

        assert result.ok is False
        assert any("hash" in e.lower() or "mismatch" in e.lower() for e in result.errors)


class TestArabicPrimaryConstraint:
    def test_fails_when_no_english_source(self, tmp_path: Path) -> None:
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        entries = _manifest_entries(en_file=None)
        _write_corpus(corpus_dir, entries)
        manifest_path = tmp_path / "registry.json"
        _write_manifest(manifest_path, entries)

        result = validate_corpus(manifest_path, corpus_dir)

        assert result.ok is False
        assert any("english" in e.lower() or "en" in e.lower() for e in result.errors)

    def test_fails_when_arabic_ratio_below_threshold(self, tmp_path: Path) -> None:
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        entries = _manifest_entries()
        # Write mostly non-Arabic content to the ar file
        _write_corpus(corpus_dir, entries, ar_content="This is English only. " * 100)
        manifest_path = tmp_path / "registry.json"
        _write_manifest(manifest_path, entries)

        result = validate_corpus(manifest_path, corpus_dir)

        assert result.ok is False
        assert any("arabic" in e.lower() or "ratio" in e.lower() for e in result.errors)


class TestManifestSchemaValidation:
    def test_rejects_invalid_manifest_missing_required_keys(self, tmp_path: Path) -> None:
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        bad_entries = [{"id": "ar"}]  # missing source, license, intended_use, files, sha256
        manifest_path = tmp_path / "registry.json"
        _write_manifest(manifest_path, bad_entries)

        result = validate_corpus(manifest_path, corpus_dir)

        assert result.ok is False
        assert any("manifest" in e.lower() or "schema" in e.lower() or "invalid" in e.lower() for e in result.errors)

    def test_load_manifest_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nope.json")


class TestLoadManifest:
    def test_load_manifest_roundtrip(self, tmp_path: Path) -> None:
        entries = _manifest_entries()
        manifest_path = tmp_path / "registry.json"
        _write_manifest(manifest_path, entries)
        loaded = load_manifest(manifest_path)
        assert len(loaded) == len(entries)
        assert loaded[0]["id"] == "ar"
