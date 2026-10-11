"""Build the retrieval indexes (Phases 9-10) from data/processed/incidents.parquet.

    python -m src.preprocessing.build_index              # BM25 + Chroma vector index
    python -m src.preprocessing.build_index --bm25-only
"""

from __future__ import annotations

import logging
import sys
import time

from src.rag.bm25 import build_bm25_index
from src.rag.vector_store import build_vector_index
from src.preprocessing.build_incidents import load_incidents
from src.preprocessing.conditions import load_conditions

logger = logging.getLogger(__name__)


def main(bm25_only: bool = False) -> None:
    inc = load_incidents()
    t = time.time()
    idx = build_bm25_index(inc)
    logger.info("BM25: %d docs, %d terms (%.1fs)", idx.weights.shape[0], idx.weights.shape[1], time.time() - t)
    if not bm25_only:
        t = time.time()
        manifest = build_vector_index(inc, list(load_conditions()))
        logger.info("Chroma: %s (%.1fs)", manifest, time.time() - t)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main(bm25_only="--bm25-only" in sys.argv)
