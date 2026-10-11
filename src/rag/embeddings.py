"""Sentence-Transformers embedding model (loaded once per process)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.utils.config import load_yaml


def model_name() -> str:
    return load_yaml("retrieval.yaml")["embedding"]["model"]


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name(), device="cpu")


def embed(texts: list[str], batch_size: int | None = None, show_progress: bool = False) -> np.ndarray:
    """L2-normalised float32 embeddings (cosine similarity == dot product)."""
    bs = batch_size or load_yaml("retrieval.yaml")["embedding"]["batch_size"]
    vecs = get_model().encode(texts, batch_size=bs, normalize_embeddings=True,
                              show_progress_bar=show_progress, convert_to_numpy=True)
    return vecs.astype("float32")
