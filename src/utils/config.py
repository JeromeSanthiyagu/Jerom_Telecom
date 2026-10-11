"""Configuration helpers: project paths and cached YAML loading."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "src" / "config"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"


@lru_cache(maxsize=None)
def load_yaml(name: str) -> dict[str, Any]:
    """Load `src/config/<name>` (cached). Treat the result as read-only."""
    path = CONFIG_DIR / name
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_schema() -> dict[str, Any]:
    return load_yaml("schema.yaml")
