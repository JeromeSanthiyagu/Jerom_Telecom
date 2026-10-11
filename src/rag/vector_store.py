"""Vector search over the incident embedding texts with ChromaDB (Phase 9).

Each Chroma item's id IS the record_id, so a vector hit maps directly to a
real record (no separate id-mapping table to drift). Metadata stored with
every vector supports Chroma `where` filtering (Phase 12).
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from src.rag.embeddings import embed, model_name
from src.utils.config import PROJECT_ROOT, load_yaml

logger = logging.getLogger(__name__)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

META_FIELDS = ["season", "weather", "state", "city", "primary_category", "severity", "priority"]


def _cfg() -> dict:
    return load_yaml("retrieval.yaml")["vector_store"]


def _path():
    return PROJECT_ROOT / _cfg()["path"]


def _client():
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(_path()), settings=Settings(anonymized_telemetry=False))


def record_metadata(rec: dict[str, Any], condition_names: list[str]) -> dict[str, Any]:
    meta: dict[str, Any] = {"record_id": rec["record_id"]}
    for f in META_FIELDS:
        v = rec.get(f)
        if v is not None and not (isinstance(v, float) and np.isnan(v)) and v is not pd.NA:
            meta[f] = str(v)
    if rec.get("issue_reported") is not None and rec.get("issue_reported") is not pd.NA:
        meta["issue_reported"] = bool(rec["issue_reported"])
    for c in condition_names:
        meta[f"cond_{c}"] = bool(rec.get(f"cond_{c}", False))
    return meta


def build_vector_index(incidents: pd.DataFrame, condition_names: list[str]) -> dict:
    """Embed every unique embedding_text once and (re)create the Chroma collection."""
    cfg = _cfg()
    texts = incidents["embedding_text"]
    unique = texts.drop_duplicates()
    logger.info("embedding %d unique texts for %d records", len(unique), len(texts))
    vecs = embed(unique.tolist(), show_progress=True)
    lookup = dict(zip(unique.tolist(), range(len(unique))))
    all_vecs = vecs[[lookup[t] for t in texts]]

    client = _client()
    try:
        client.delete_collection(cfg["collection"])
    except Exception:  # collection did not exist
        pass
    col = client.create_collection(cfg["collection"], metadata={"hnsw:space": cfg["distance"]})

    recs = incidents.to_dict("records")
    bs = cfg["add_batch_size"]
    for start in range(0, len(recs), bs):
        chunk = recs[start:start + bs]
        col.add(
            ids=[r["record_id"] for r in chunk],
            embeddings=all_vecs[start:start + bs].tolist(),
            documents=[r["embedding_text"] for r in chunk],
            metadatas=[record_metadata(r, condition_names) for r in chunk],
        )
        logger.info("added %d/%d", min(start + bs, len(recs)), len(recs))

    manifest = {"model": model_name(), "dim": int(all_vecs.shape[1]), "count": len(recs),
                "unique_texts": int(len(unique)), "collection": cfg["collection"]}
    (_path() / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    load_vector_index.cache_clear()
    return manifest


@lru_cache(maxsize=1)
def load_vector_index():
    """Open the persisted collection and check it was built with the current model."""
    manifest_path = _path() / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Vector index not built - run `python -m src.preprocessing.build_index`.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["model"] != model_name():
        raise RuntimeError(f"Index built with {manifest['model']}, config uses {model_name()} - rebuild it.")
    return _client().get_collection(_cfg()["collection"])


def semantic_search(query: str, top_k: int = 5, where: dict | None = None,
                    query_vec: np.ndarray | None = None) -> list[dict]:
    """Top-k records by cosine similarity: [{record_id, semantic_score, metadata}]."""
    col = load_vector_index()
    qv = query_vec if query_vec is not None else embed([query])[0]
    res = col.query(query_embeddings=[qv.tolist()], n_results=top_k, where=where,
                    include=["distances", "metadatas"])
    return [
        {"record_id": rid, "semantic_score": round(1.0 - float(d), 6), "metadata": m}
        for rid, d, m in zip(res["ids"][0], res["distances"][0], res["metadatas"][0])
    ]


def semantic_scores(query_vec: np.ndarray, record_ids: list[str]) -> dict[str, float]:
    """Exact cosine similarity of the query to specific records."""
    if not record_ids:
        return {}
    got = load_vector_index().get(ids=record_ids, include=["embeddings"])
    emb = np.asarray(got["embeddings"], dtype="float32")
    sims = emb @ query_vec.astype("float32")
    return dict(zip(got["ids"], sims.astype(float)))
