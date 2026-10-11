"""Download the 5G NetOps dataset from Hugging Face at a pinned revision.

Source: https://huggingface.co/datasets/fenar/5g_netops (license: apache-2.0)

The revision and SHA-256 are pinned so every run of the project works on
exactly the same bytes. If the upstream file changes, the checksum check
fails loudly instead of silently analysing different data.

Usage:
    python -m src.preprocessing.download_dataset
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

REPO_ID = "fenar/5g_netops"
REVISION = "75f94e94097eb72efcd95ef417d4be171ac31bb6"  # upstream commit, 2024-06-19
FILENAME = "5G_netops_data_100K.csv.xz"
SHA256 = "006b685455531c414fcb4872aa8e10ca205091c41a0cf7395f276b6d86c96e98"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_PATH = RAW_DIR / FILENAME
URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{FILENAME}"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(force: bool = False) -> Path:
    """Download the raw file if missing (or if `force`), then verify its checksum."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if force or not RAW_PATH.exists():
        print(f"Downloading {URL}")
        tmp = RAW_PATH.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.replace(RAW_PATH)

    actual = sha256_of(RAW_PATH)
    if actual != SHA256:
        raise RuntimeError(
            f"Checksum mismatch for {RAW_PATH}: expected {SHA256}, got {actual}. "
            "Delete the file and re-run, or review the upstream change."
        )
    print(f"OK  {RAW_PATH}  sha256={actual}")
    return RAW_PATH


if __name__ == "__main__":
    download(force="--force" in sys.argv)
