"""BM25 keyword retrieval (Phase 10).

Okapi BM25 with the Lucene-style IDF  ln(1 + (N - n + 0.5) / (n + 0.5)),
precomputed as a sparse document x term weight matrix so a query is one
sparse column-sum, and the index persists as a .npz file.

Tokenisation: lower-case words (no pure numbers, no English stop words), plus
word bigrams joined by "_" and the parts of snake_case tokens. Every document
mentions "latency" in its metric line, but only records with the derived
condition contain the bigram `high_latency`, so phrase queries discriminate.

Indexed fields: search_text, which already contains the derived conditions
(by label and name), category, severity, priority, weather, season, location
and issue-reported flag.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from src.utils.config import PROJECT_ROOT, load_yaml

_WORD = re.compile(r"[a-z][a-z0-9_]*")


def tokenize(text: str) -> list[str]:
    words = [w for w in _WORD.findall(text.lower()) if w not in ENGLISH_STOP_WORDS]
    tokens: list[str] = []
    for w in words:
        tokens.append(w)
        if "_" in w:
            parts = [p for p in w.split("_") if p]
            tokens.extend(parts)
            tokens.extend(f"{a}_{b}" for a, b in zip(parts, parts[1:]))
    plain = [w for w in words if "_" not in w]
    tokens.extend(f"{a}_{b}" for a, b in zip(plain, plain[1:]))
    return tokens


class BM25Index:
    def __init__(self, weights: sparse.csc_matrix, vocab: dict[str, int], record_ids: list[str]):
        self.weights = weights            # docs x terms, BM25 term weights
        self.vocab = vocab
        self.record_ids = record_ids

    # ---------------------------------------------------------------- build
    @classmethod
    def build(cls, texts: list[str], record_ids: list[str], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        vec = CountVectorizer(analyzer=tokenize)
        tf = vec.fit_transform(texts).tocsr().astype(np.float64)   # docs x terms
        n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = doc_len.mean()
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        tf = tf.tocoo()
        norm = k1 * (1 - b + b * doc_len[tf.row] / avgdl)
        w = idf[tf.col] * tf.data * (k1 + 1) / (tf.data + norm)
        weights = sparse.csc_matrix((w, (tf.row, tf.col)), shape=tf.shape)
        vocab = {t: int(i) for t, i in vec.vocabulary_.items()}
        return cls(weights, vocab, list(record_ids))

    # ---------------------------------------------------------------- query
    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document (0 for documents without query terms)."""
        ids = sorted({self.vocab[t] for t in tokenize(query) if t in self.vocab})
        if not ids:
            return np.zeros(self.weights.shape[0])
        return np.asarray(self.weights[:, ids].sum(axis=1)).ravel()

    def search(self, query: str, top_k: int = 5, mask: np.ndarray | None = None) -> list[dict]:
        s = self.scores(query)
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        k = min(top_k, int(np.isfinite(s).sum()))
        if k <= 0:
            return []
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx], kind="stable")]
        return [{"record_id": self.record_ids[i], "keyword_score": float(s[i]), "row": int(i)}
                for i in idx if s[i] > 0]

    # ---------------------------------------------------------------- persist
    def save(self, path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(path / "weights.npz", self.weights)
        (path / "vocab.json").write_text(json.dumps(self.vocab), encoding="utf-8")
        (path / "record_ids.json").write_text(json.dumps(self.record_ids), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "BM25Index":
        if not (path / "weights.npz").exists():
            raise FileNotFoundError("BM25 index not built - run `python -m src.preprocessing.build_index`.")
        return cls(
            sparse.load_npz(path / "weights.npz").tocsc(),
            json.loads((path / "vocab.json").read_text(encoding="utf-8")),
            json.loads((path / "record_ids.json").read_text(encoding="utf-8")),
        )


def _cfg() -> dict:
    return load_yaml("retrieval.yaml")["bm25"]


def build_bm25_index(incidents: pd.DataFrame) -> BM25Index:
    cfg = _cfg()
    idx = BM25Index.build(incidents["search_text"].tolist(), incidents["record_id"].tolist(),
                          k1=cfg["k1"], b=cfg["b"])
    idx.save(PROJECT_ROOT / cfg["path"])
    load_bm25_index.cache_clear()
    return idx


@lru_cache(maxsize=1)
def load_bm25_index() -> BM25Index:
    return BM25Index.load(PROJECT_ROOT / _cfg()["path"])


def keyword_search(query: str, top_k: int = 5) -> list[dict]:
    return load_bm25_index().search(query, top_k)
