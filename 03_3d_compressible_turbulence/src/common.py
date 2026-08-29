"""Shared helpers for the PDEBench 3D CFD demonstration."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml


FIELDS = ("D", "Vx", "Vy", "Vz", "P")


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def locate_field_files(raw_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for field in FIELDS:
        matches = sorted(raw_dir.glob(f"*_{field}.npy"))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one *_{field}.npy in {raw_dir}, found {len(matches)}"
            )
        found[field] = matches[0]
    return found

