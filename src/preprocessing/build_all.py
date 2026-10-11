"""Rebuild every derived artifact from the raw dataset, in order (cross-platform version of build_all.sh).

    python -m src.preprocessing.build_all

Steps: pinned download + checksum -> inspection report -> validated, cleaned records -> threshold check ->
conditions, labels and chunk texts -> BM25 + ChromaDB indexes (about 8 minutes on a CPU) -> ML models.
"""

from __future__ import annotations

import subprocess
import sys
import time

STEPS = [
    ("download dataset (pinned revision, SHA-256 checked)", ["-m", "src.preprocessing.download_dataset"]),
    ("inspect dataset (Phase 1 report)", ["-m", "src.preprocessing.inspect_dataset"]),
    ("preprocess: validate and clean", ["-m", "src.preprocessing.processor"]),
    ("check thresholds still match the data", ["-m", "src.preprocessing.build_thresholds", "--check"]),
    ("derive conditions, labels and chunk texts", ["-m", "src.preprocessing.build_incidents"]),
    ("build BM25 and ChromaDB indexes", ["-m", "src.preprocessing.build_index"]),
    ("train ML models", ["-m", "src.services.models.train"]),
]


def main() -> None:
    for i, (title, args) in enumerate(STEPS, 1):
        print(f"[{i}/{len(STEPS)}] {title} ...", flush=True)
        t = time.time()
        subprocess.run([sys.executable, *args], check=True)
        print(f"      done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
