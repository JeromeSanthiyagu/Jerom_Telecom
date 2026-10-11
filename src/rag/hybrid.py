"""Hybrid retrieval (Phases 11-13): semantic + BM25 + metric similarity, with filters.

    1. validate filters -> mask over the incident table (+ Chroma where clause)
    2. candidates = top `candidate_pool` from each method (inside the filter)
    3. re-score every candidate exactly with all methods
    4. min-max normalise semantic and keyword scores over the candidates
    5. hybrid = weighted sum (weights from retrieval.yaml, re-normalised to
       the methods that are active for this query)
    6. final guarantee: drop anything that violates the filters, return top_k
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from src.rag.bm25 import load_bm25_index
from src.rag.embeddings import embed
from src.rag.filters import Filters, chroma_where, filter_mask, normalize_filters, record_matches
from src.rag.metric_similarity import MetricScorer
from src.rag.query_parser import QueryUnderstanding, parse_query
from src.rag.vector_store import semantic_scores, semantic_search
from src.preprocessing.build_incidents import load_incidents
from src.preprocessing.conditions import load_conditions
from src.utils.config import load_yaml

logger = logging.getLogger(__name__)

METRIC_COLUMNS = [
    "cell_availability_pct", "throughput_mbps", "latency_ms", "packet_loss_pct", "call_drop_pct",
    "handover_success_pct", "signal_strength_dbm", "jitter_ms", "setup_success_pct", "user_count",
    "data_usage_gb", "alarm_count", "critical_alarm_count", "config_success_pct", "parameter_changes",
    "security_incidents", "auth_failures", "mttr_hours", "fault_rate_pct",
]
CONTEXT_COLUMNS = ["season", "weather", "city", "state", "zip", "temperature_c", "humidity_pct", "issue_reported"]


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.ones_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def _py(v: Any) -> Any:
    if v is None or v is pd.NA:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return [_py(x) for x in v.tolist()]
    return v


class HybridRetriever:
    def __init__(self, incidents: pd.DataFrame | None = None):
        self.df = incidents if incidents is not None else load_incidents()
        self.df = self.df.reset_index(drop=True)
        self.row_of = {rid: i for i, rid in enumerate(self.df["record_id"])}
        self.bm25 = load_bm25_index()
        if self.bm25.record_ids != self.df["record_id"].tolist():
            raise RuntimeError("BM25 index is out of sync with incidents - rebuild the index.")
        self.metric = MetricScorer(self.df, METRIC_COLUMNS)
        self.cities = set(self.df["city"].dropna().unique())
        cfg = load_yaml("retrieval.yaml")["hybrid"]
        self.weights = cfg["weights"]
        self.pool = cfg["candidate_pool"]
        self.max_top_k = cfg["max_top_k"]
        self.contradiction_penalty = cfg.get("contradiction_penalty", 0.0)
        self.condition_names = list(load_conditions())

    # ------------------------------------------------------------------ API
    def record(self, record_id: str) -> dict | None:
        i = self.row_of.get(record_id)
        return None if i is None else self.to_result(i)

    def hybrid_search(self, query: str, filters: dict | Filters | None = None, top_k: int = 5,
                      understanding: QueryUnderstanding | None = None) -> dict:
        top_k = max(1, min(int(top_k), self.max_top_k))
        f = filters if isinstance(filters, Filters) else normalize_filters(filters, self.cities)
        q = understanding or parse_query(query)
        mask = filter_mask(self.df, f)
        n_allowed = int(mask.sum())
        if n_allowed == 0:
            return {"results": [], "filters": f.as_dict(), "candidates": 0, "matching_records": 0,
                    "understanding": q.to_dict(), "weights": {}}

        text = q.retrieval_text
        qvec = embed([text])[0]
        # ---- candidates
        cand: set[int] = set()
        for hit in semantic_search(text, top_k=min(self.pool, n_allowed), where=chroma_where(f), query_vec=qvec):
            if hit["record_id"] in self.row_of:
                cand.add(self.row_of[hit["record_id"]])
        kw_all = self.bm25.scores(text)
        kw_masked = np.where(mask, kw_all, -np.inf)
        k = min(self.pool, n_allowed)
        cand.update(np.argpartition(-kw_masked, k - 1)[:k].tolist())
        metric_all = self.metric.score(q)
        if metric_all is not None:
            m_masked = np.where(mask, metric_all, -np.inf)
            cand.update(np.argpartition(-m_masked, k - 1)[:k].tolist())
        rows = np.array(sorted(i for i in cand if mask[i]))

        # ---- exact re-scoring
        ids = self.df["record_id"].to_numpy()[rows].tolist()
        sem_map = semantic_scores(qvec, ids)
        sem = np.array([sem_map[r] for r in ids])
        kw = kw_all[rows]
        weights = {"semantic": self.weights["semantic"], "keyword": self.weights["keyword"]}
        parts = {"semantic": _minmax(sem), "keyword": _minmax(kw) if kw.max() > 0 else np.zeros_like(kw)}
        if metric_all is not None:
            weights["metric"] = self.weights["metric"]
            parts["metric"] = metric_all[rows]
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        hybrid = sum(weights[k] * parts[k] for k in weights)
        # records showing a condition the user explicitly called good/normal are penalised
        favorable = q.favorable_conditions()
        if favorable:
            conflicts = sum(self.df[f"cond_{c}"].to_numpy()[rows].astype(float) for c in favorable)
            hybrid = hybrid - self.contradiction_penalty * conflicts

        order = np.argsort(-hybrid, kind="stable")
        results = []
        for j in order:
            res = self.to_result(int(rows[j]))
            if not record_matches(res["_raw"], f):     # final guarantee (should never trigger)
                logger.error("filter violation dropped: %s", res["record_id"])
                continue
            res.pop("_raw")
            res.update({
                "semantic_score": round(float(sem[j]), 4),
                "keyword_score": round(float(parts["keyword"][j]), 4),
                "keyword_score_raw": round(float(kw[j]), 4),
                "metric_score": round(float(parts["metric"][j]), 4) if "metric" in parts else None,
                "hybrid_score": round(float(hybrid[j]), 4),
            })
            results.append(res)
            if len(results) == top_k:
                break
        return {"results": results, "filters": f.as_dict(), "candidates": int(len(rows)),
                "matching_records": n_allowed, "understanding": q.to_dict(),
                "weights": {k: round(v, 4) for k, v in weights.items()}}

    # ------------------------------------------------------------ formatting
    def to_result(self, i: int) -> dict:
        r = self.df.iloc[i]
        raw = {c: _py(r[c]) for c in ["record_id", "primary_category", "severity", "priority", "city", "state",
                                       "season", "weather", "issue_reported"]}
        raw.update({f"cond_{c}": bool(r[f"cond_{c}"]) for c in self.condition_names})
        return {
            "record_id": r["record_id"],
            "category": r["primary_category"],
            "secondary_categories": _py(r["secondary_categories"]),
            "severity": r["severity"],
            "severity_score": _py(r["severity_score"]),
            "priority": r["priority"],
            "conditions": _py(r["conditions"]),
            "metrics": {c: _py(r[c]) for c in METRIC_COLUMNS},
            "metadata": {c: _py(r[c]) for c in CONTEXT_COLUMNS},
            "search_text": r["search_text"],
            "_raw": raw,
        }


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()


def hybrid_search(query: str, filters: dict | None = None, top_k: int = 5) -> dict:
    return get_retriever().hybrid_search(query, filters, top_k)
