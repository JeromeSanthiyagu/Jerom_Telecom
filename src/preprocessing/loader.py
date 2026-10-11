"""Load the raw dataset (unchanged) and the processed dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.preprocessing.download_dataset import RAW_PATH, download
from src.utils.config import PROCESSED_DIR

PROCESSED_PATH = PROCESSED_DIR / "records.parquet"


def load_raw(path: Path | None = None, verify: bool = True) -> pd.DataFrame:
    """Return the raw CSV exactly as published.

    Everything is read as text (`dtype=str`) so that type conversion happens
    in one visible, logged place (`cleaner.convert_types`) instead of silently
    inside `read_csv`. With `verify`, the pinned file is downloaded if missing
    and its checksum is checked.
    """
    if path is None:
        path = download() if verify else RAW_PATH
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def load_processed(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m src.preprocessing.processor` first.")
    return pd.read_parquet(path)
