"""Fetch the corpus sources from Kaggle into data/corpus, stamping hashes.

The registry (data/corpus/registry.json) declares the sources.  After a fetch,
each entry's `files` and `sha256` are stamped with what is actually on disk, so
validation (scripts/check_corpus.py) can verify the corpus deterministically.

Usage:
    uv run scripts/fetch_corpus.py --all       # every source (default)
    uv run scripts/fetch_corpus.py --id ar     # one source
    uv run scripts/fetch_corpus.py --id ar --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common import corpus_defaults, iter_files, sha256_file
from scripts.manifest import load_manifest
DEFAULT_CORPUS_DIR, DEFAULT_MANIFEST = corpus_defaults()
WORK_DIR = DEFAULT_CORPUS_DIR / ".work"


def _stamp_tree(entry_id: str, dest: Path) -> tuple[list[str], dict[str, str]]:
    """Collect every file under dest prefixed with the entry id.

    Validation resolves manifest paths against the corpus root, so each entry's
    files must read ``<id>/<relative-path>`` (e.g. ``ar/test.csv``), matching the
    contract the validation tests assert.
    """
    files: list[str] = []
    sha256: dict[str, str] = {}
    for file in iter_files(dest):
        relative = f"{entry_id}/{file.relative_to(dest)}"
        files.append(relative)
        sha256[relative] = sha256_file(file)
    return files, sha256


def _flatten_extracted(staging: Path, dest: Path) -> None:
    """Move files out of any single top-level wrapper folder into dest."""
    dest.mkdir(parents=True, exist_ok=True)
    top_level = [p for p in staging.iterdir() if p.name != "__MACOSX"]
    if len(top_level) == 1 and top_level[0].is_dir():
        source = top_level[0]
    else:
        source = staging
    for file in iter_files(source):
        relative = file.relative_to(source)
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        file.rename(target)


def _kaggle_bin() -> str:
    """Locate the kaggle CLI: on PATH, else in this venv's bin dir."""
    from shutil import which

    on_path = which("kaggle")
    if on_path:
        return on_path
    candidates = [
        Path(sys.prefix) / "bin" / "kaggle",
        Path(sys.executable).parent / "kaggle",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    sys.exit("kaggle CLI not found — install from requirements.txt (kaggle==2.2.4)")



def _download_entry(source: str, slug: str, dest: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"  [dry-run] would download {source} -> {dest}")
        return
    staging = WORK_DIR / f"staging-{slug}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    click_dir = WORK_DIR / "click"
    click_dir.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {source} …")
    result = subprocess.run(
        [_kaggle_bin(), "datasets", "download", "-d", source, "-p", str(click_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"kaggle datasets download failed for {source}:\n{result.stderr}")
    archived = click_dir / f"{slug}.zip"
    with zipfile.ZipFile(archived) as zf:
        zf.extractall(staging)
    _flatten_extracted(staging, dest)
    archived.unlink()
    shutil.rmtree(staging)


def fetch(manifest_path: Path, corpus_dir: Path, entry_ids: set[str] | None, dry_run: bool) -> None:
    entries = load_manifest(manifest_path)
    if entry_ids:
        entries = [e for e in entries if e["id"] in entry_ids]

    anything_fetched = False
    for entry in entries:
        entry_id = entry["id"]
        source = entry["source"]
        slug = source.rsplit("/", 1)[-1]
        dest = corpus_dir / entry_id

        if not dry_run and dest.exists():
            files = iter_files(dest)
            if files:
                print(f"  [skip] {entry_id}: already present ({len(files)} files)")
                entry["files"], entry["sha256"] = _stamp_tree(entry_id, dest)
                continue

        _download_entry(source, slug, dest, dry_run)
        if dry_run:
            continue
        entry["files"], entry["sha256"] = _stamp_tree(entry_id, dest)
        bytes_ = sum((corpus_dir / f).stat().st_size for f in entry["files"])
        print(f"  stamped {entry_id}: {len(entry['files'])} files, {bytes_ / 1e6:.1f} MB")
        anything_fetched = True

    if not dry_run:
        manifest_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
        print(f"registry updated: {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="fetch every source in the registry (default)")
    parser.add_argument("--id", action="append", help="fetch only the given source id(s)")
    parser.add_argument("--dry-run", action="store_true", help="print what would be fetched without downloading")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    if not args.id and not args.all:
        parser.error("pass --all or --id <source id>")
    if args.all and args.id:
        parser.error("--all and --id are mutually exclusive")

    corpus_dir = args.corpus_dir
    if not corpus_dir.is_absolute():
        corpus_dir = REPO_ROOT / corpus_dir
    corpus_dir.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    target_ids = set(args.id) if args.id else None
    fetch(args.manifest, corpus_dir, target_ids, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())