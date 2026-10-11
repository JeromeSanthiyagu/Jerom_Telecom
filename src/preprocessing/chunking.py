"""Searchable incident representation (Phase 7).

The dataset has no natural-language descriptions, so every record gets a
deterministic text built ONLY from its own raw fields and the documented
derived features. Two texts are produced:

* `search_text`    - the full representation: shown to users and the LLM,
                     indexed by BM25. Numbers are rounded to 2 decimals for
                     display; exact values stay in the structured columns.
* `embedding_text` - the qualitative part (conditions, category, severity,
                     context) without numbers. The sentence-embedding model
                     reads at most 256 word pieces and numbers carry little
                     semantic signal; numeric closeness is handled by the
                     separate metric-similarity score (Phase 13).

Missing values are written as "not available in the current dataset".
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from src.preprocessing.conditions import load_conditions
from src.utils.geo import STATE_NAMES

NA_TEXT = "not available in the current dataset"

# (column, label, unit, decimals)
NETWORK = [
    ("cell_availability_pct", "cell availability", "%", 2),
    ("throughput_mbps", "throughput", " Mbps", 2),
    ("latency_ms", "latency", " ms", 2),
    ("packet_loss_pct", "packet loss", "%", 2),
    ("call_drop_pct", "call drop rate", "%", 2),
    ("handover_success_pct", "handover success", "%", 2),
    ("signal_strength_dbm", "signal strength", " dBm", 2),
    ("jitter_ms", "jitter", " ms", 2),
    ("setup_success_pct", "connection setup success", "%", 2),
]
OPERATIONAL = [
    ("alarm_count", "alarm count", "", 0),
    ("critical_alarm_count", "critical alarm count", "", 2),
    ("parameter_changes", "parameter changes", "", 0),
    ("config_success_pct", "configuration change success", "%", 2),
    ("mttr_hours", "MTTR", " hours", 2),
]
LOAD = [("data_usage_gb", "data usage", " GB", 2), ("user_count", "user count", "", 0)]
SECURITY = [("security_incidents", "security incidents", "", 0),
            ("auth_failures", "authentication failures", "", 0)]
ENVIRONMENT = [("temperature_c", "temperature", " °C", 2), ("humidity_pct", "humidity", "%", 2)]


def _is_missing(v: Any) -> bool:
    return v is None or (not isinstance(v, (list, tuple)) and pd.isna(v))


def fmt_value(v: Any, unit: str = "", decimals: int = 2) -> str:
    if _is_missing(v):
        return NA_TEXT
    return f"{float(v):.{decimals}f}{unit}" if decimals else f"{int(round(float(v)))}{unit}"


def _section(rec: Mapping[str, Any], fields) -> str:
    return "; ".join(f"{label} {fmt_value(rec.get(col), unit, d)}" for col, label, unit, d in fields)


def _text(v: Any) -> str:
    return NA_TEXT if _is_missing(v) else str(v)


def _location(rec: Mapping[str, Any]) -> str:
    state = rec.get("state")
    state_txt = NA_TEXT if _is_missing(state) else f"{state} ({STATE_NAMES.get(state, 'unknown state code')})"
    return f"{_text(rec.get('city'))}, {state_txt}"


def _issue(rec: Mapping[str, Any]) -> str:
    v = rec.get("issue_reported")
    return NA_TEXT if _is_missing(v) else ("yes" if bool(v) else "no")


def _conditions_text(rec: Mapping[str, Any]) -> str:
    labels = {n: c.label.lower() for n, c in load_conditions().items()}
    conds = rec.get("conditions") or []
    return ", ".join(f"{labels[c]} ({c})" for c in conds) if len(conds) else "none (all metrics within thresholds)"


def _derived_lines(rec: Mapping[str, Any]) -> list[str]:
    secondary = list(rec.get("secondary_categories") or [])
    cat = rec.get("primary_category", NA_TEXT) + (f" (secondary: {', '.join(secondary)})" if secondary else "")
    return [
        f"Derived conditions: {_conditions_text(rec)}.",
        f"Derived category: {cat}.",
        f"Derived severity: {rec.get('severity', NA_TEXT)} (score {fmt_value(rec.get('severity_score'), '', 1)}). "
        f"Derived priority: {rec.get('priority', NA_TEXT)}.",
    ]


def build_search_text(rec: Mapping[str, Any]) -> str:
    lines = [f"Network record {rec['record_id']} (description generated from dataset fields)."]
    lines += _derived_lines(rec)
    lines += [
        f"Location: {_location(rec)}. Season: {_text(rec.get('season'))}. Weather: {_text(rec.get('weather'))}.",
        f"Network metrics: {_section(rec, NETWORK)}.",
        f"Operational metrics: {_section(rec, OPERATIONAL)}.",
        f"Load: {_section(rec, LOAD)}.",
        f"Security: {_section(rec, SECURITY)}.",
        f"Environment: {_section(rec, ENVIRONMENT)}.",
        f"Customer issue reported: {_issue(rec)}.",
        f"Historical fault occurrence rate: {fmt_value(rec.get('fault_rate_pct'), '%', 2)}.",
    ]
    return "\n".join(lines)


def build_embedding_text(rec: Mapping[str, Any]) -> str:
    """Qualitative summary for the sentence-embedding model.

    Location and numbers are deliberately left out: location is matched by
    filters/BM25, numbers by metric similarity. This also makes many texts
    identical (about 27k unique for 100k records), so each unique text is
    embedded once.
    """
    secondary = list(rec.get("secondary_categories") or [])
    cat = rec.get("primary_category", NA_TEXT) + (f" (secondary: {', '.join(secondary)})" if secondary else "")
    return (
        f"Derived conditions: {_conditions_text(rec)}. Derived category: {cat}. "
        f"Derived severity: {rec.get('severity', NA_TEXT)}. Derived priority: {rec.get('priority', NA_TEXT)}. "
        f"Weather {_text(rec.get('weather'))}, season {_text(rec.get('season'))}. "
        f"Customer issue reported: {_issue(rec)}."
    )
