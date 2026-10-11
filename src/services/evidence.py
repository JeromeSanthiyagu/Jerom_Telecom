"""Evidence extraction (Phase 15): observations from the query and statistics
calculated from the actual retrieved records and the dataset baseline.

Nothing here is estimated or invented - every number is computed from rows
that exist in data/processed/incidents.parquet or was supplied by the user.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from src.rag.query_parser import QueryUnderstanding
from src.preprocessing.conditions import evaluate_record, load_conditions

KEY_METRICS = ["latency_ms", "packet_loss_pct", "call_drop_pct", "handover_success_pct", "throughput_mbps",
               "signal_strength_dbm", "jitter_ms", "cell_availability_pct", "setup_success_pct", "user_count",
               "critical_alarm_count", "auth_failures", "fault_rate_pct", "mttr_hours"]


@lru_cache(maxsize=1)
def dataset_baseline() -> dict[str, Any]:
    """Medians over the full dataset + mean fault rate by weather (cached)."""
    from src.rag.hybrid import get_retriever

    df = get_retriever().df
    med = {m: round(float(df[m].astype("float64").median()), 2) for m in KEY_METRICS}
    by_weather = df.groupby("weather")["fault_rate_pct"].mean().round(2).to_dict()
    return {"rows": int(len(df)), "median": med, "mean_fault_rate_by_weather": by_weather}


def observed_state(q: QueryUnderstanding, metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Which conditions are present / absent / unknown for the case being analysed.

    present  - stated as adverse in the query, or a supplied value beyond threshold
    absent   - stated as fine/normal, or a supplied value within threshold
    unknown  - everything else
    """
    values = {**q.stated_values(), **{k: v for k, v in (metrics or {}).items() if v is not None}}
    measured = evaluate_record(values)
    present = set(q.adverse_conditions()) | set(measured["conditions"])
    conds = load_conditions()
    measured_ok = {n for n, c in conds.items() if c.metric in values and n not in measured["conditions"]}
    absent = (set(q.favorable_conditions()) | measured_ok) - present
    unknown = set(conds) - present - absent
    sources = {}
    for n in present:
        sources[n] = "measured" if n in measured["conditions"] else "stated"
    return {
        "values": values,
        "present": sorted(present),
        "absent": sorted(absent),
        "unknown": sorted(unknown),
        "condition_sources": sources,
        "measured_evidence": measured["evidence"],
        "context": dict(q.context),
    }


def historical_statistics(results: list[dict]) -> dict[str, Any]:
    """Statistics over the retrieved records only (the historical evidence)."""
    if not results:
        return {"n_records": 0}
    n = len(results)
    cond_counts = Counter(c for r in results for c in r["conditions"])
    stats: dict[str, Any] = {
        "n_records": n,
        "record_ids": [r["record_id"] for r in results],
        "condition_frequency": {c: {"count": k, "share": round(k / n, 2)} for c, k in cond_counts.most_common()},
        "category_distribution": dict(Counter(r["category"] for r in results).most_common()),
        "severity_distribution": dict(Counter(r["severity"] for r in results).most_common()),
        "weather_distribution": dict(Counter(r["metadata"]["weather"] for r in results).most_common()),
        "metrics": {},
    }
    for m in KEY_METRICS:
        vals = np.array([r["metrics"][m] for r in results if r["metrics"].get(m) is not None], dtype=float)
        if len(vals):
            stats["metrics"][m] = {"mean": round(float(vals.mean()), 2), "median": round(float(np.median(vals)), 2),
                                   "min": round(float(vals.min()), 2), "max": round(float(vals.max()), 2)}
    return stats


def records_for_prompt(results: list[dict], limit: int = 5) -> list[dict]:
    """Compact view of retrieved records for the LLM (values exactly as stored)."""
    out = []
    for r in results[:limit]:
        out.append({
            "record_id": r["record_id"],
            "hybrid_score": r.get("hybrid_score"),
            "category": r["category"],
            "severity": r["severity"],
            "conditions": r["conditions"],
            "weather": r["metadata"]["weather"],
            "season": r["metadata"]["season"],
            "metrics": {m: (round(v, 2) if isinstance(v, float) else v) for m, v in r["metrics"].items() if m in KEY_METRICS},
        })
    return out


def to_frame(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{**r["metrics"], **r["metadata"], "record_id": r["record_id"]} for r in results])
