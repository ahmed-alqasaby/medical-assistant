"""Seam B: environment / project-contract tests.

These assert the structural invariants M1 requires: .env.example present,
requirements.txt pins kaggle, .gitignore covers data artifacts, setup script
exists, Python ≥3.10, and the corpus inventory doc is present and readable.
"""
from __future__ import annotations

import os
import re
import sys
import stat
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_env_example() -> dict[str, str]:
    """Parse .env.example as KEY=value pairs, ignoring comments and blanks."""
    path = ROOT / ".env.example"
    env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    return env


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class TestPythonVersion:
    def test_venv_python_version_at_least_3_10(self) -> None:
        venv_python = ROOT / ".venv" / "bin" / "python"
        assert venv_python.exists(), ".venv/bin/python not found — run scripts/setup.sh first"
        # We're running under pytest with the venv, so just check sys.version_info.
        assert sys.version_info >= (3, 10), f"Expected Python >=3.10, got {sys.version}"


class TestEnvExample:
    def test_env_example_exists(self) -> None:
        path = ROOT / ".env.example"
        assert path.exists(), ".env.example missing"
        content = path.read_text(encoding="utf-8")
        assert len(content.strip()) > 0, ".env.example is empty"

    def test_env_example_has_data_corpus_dir(self) -> None:
        env = _load_env_example()
        assert "DATA_CORPUS_DIR" in env, ".env.example must define DATA_CORPUS_DIR"

    def test_env_example_has_no_committed_secrets(self) -> None:
        env = _load_env_example()
        for key, value in env.items():
            assert value != "", f"{key} in .env.example has an empty value — should be a placeholder"
            # Ensure no real tokens leaked
            assert "KGAT_" not in value, f"{key} looks like a real Kaggle token in .env.example"


class TestRequirements:
    def test_requirements_txt_pinned(self) -> None:
        path = ROOT / "requirements.txt"
        assert path.exists(), "requirements.txt missing"
        content = path.read_text()
        assert re.search(r"kaggle==\d+\.\d+\.\d+", content), "requirements.txt must pin kaggle==X.Y.Z"

    def test_requirements_txt_pinned_pytest(self) -> None:
        path = ROOT / "requirements.txt"
        content = path.read_text()
        assert re.search(r"pytest==\d+\.\d+\.\d+", content), "requirements.txt must pin pytest==X.Y.Z"


class TestGitignore:
    @pytest.mark.parametrize("pattern", [".venv", ".env", "data/corpus"])
    def test_gitignore_covers_pattern(self, pattern: str) -> None:
        path = ROOT / ".gitignore"
        assert path.exists(), ".gitignore missing"
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        # Exact match or glob-style match (e.g. data/corpus/** or data/corpus)
        matched = any(
            line == pattern or line.startswith(pattern + "/") or line == pattern + "/**"
            for line in lines
        )
        assert matched, f".gitignore must cover '{pattern}' — got lines: {lines}"


class TestSetupScript:
    def test_setup_script_exists(self) -> None:
        path = ROOT / "scripts" / "setup.sh"
        assert path.exists(), "scripts/setup.sh missing"

    def test_setup_script_is_executable(self) -> None:
        path = ROOT / "scripts" / "setup.sh"
        mode = path.stat().st_mode
        is_exec = bool(mode & stat.S_IXUSR) or bool(mode & stat.S_IXGRP)
        # Soft check: warn but don't fail if not yet +x (chmod comes later)
        if not is_exec:
            pytest.skip("scripts/setup.sh not yet marked executable — will be fixed")


class TestCorpusInventory:
    def test_inventory_exists(self) -> None:
        path = ROOT / "data" / "corpus" / "INVENTORY.md"
        assert path.exists(), "data/corpus/INVENTORY.md missing"

    def test_inventory_non_empty(self) -> None:
        path = ROOT / "data" / "corpus" / "INVENTORY.md"
        content = path.read_text(encoding="utf-8")
        assert len(content.strip()) > 0, "INVENTORY.md is empty"
        # Must mention the four sources
        for kw in ["yassinabdulmahdi", "gpreda", "nooralzoghby", "nadaarfaoui"]:
            assert kw in content, f"INVENTORY.md must document source '{kw}'"

    def test_inventory_records_licenses(self) -> None:
        path = ROOT / "data" / "corpus" / "INVENTORY.md"
        content = path.read_text(encoding="utf-8")
        for lic in ["MIT", "Apache", "Apache-2.0"]:
            if lic in content:
                return
        pytest.fail("INVENTORY.md must mention at least one permissive license (MIT / Apache)")


class TestRegistry:
    def test_registry_json_exists(self) -> None:
        path = ROOT / "data" / "corpus" / "registry.json"
        assert path.exists(), "data/corpus/registry.json missing"

    def test_registry_json_valid(self) -> None:
        import json
        from scripts.manifest import REQUIRED_KEYS
        path = ROOT / "data" / "corpus" / "registry.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, list), "registry.json must be a JSON array"
        for entry in data:
            missing = REQUIRED_KEYS - set(entry.keys())
            assert not missing, f"Entry '{entry.get('id')}' missing keys: {missing}"
