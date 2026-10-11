"""Metadata filters (Phase 12).

User filters are validated and normalised BEFORE retrieval, then applied in
two places:
  1. as a boolean mask over the incident table (BM25, metric scoring and a
     final post-check), and
  2. as a Chroma `where` clause for the vector search.
The final post-check guarantees no returned record violates an explicit filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.conditions import load_conditions
from src.preprocessing.taxonomy import all_categories
from src.utils.geo import STATE_NAMES, to_state_code

SEASONS = ["Winter", "Spring", "Summer", "Fall"]
WEATHER = ["Clear", "Clouds", "Rain", "Snow"]
SEVERITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
PRIORITIES = ["P1", "P2", "P3", "P4"]

# filter key -> incident column
COLUMN = {
    "season": "season", "weather": "weather", "state": "state", "city": "city",
    "category": "primary_category", "severity": "severity", "priority": "priority",
    "issue_reported": "issue_reported",
}
SUPPORTED = sorted(list(COLUMN) + ["conditions"])


class FilterError(ValueError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


@dataclass
class Filters:
    values: dict[str, list[Any]] = field(default_factory=dict)   # key -> allowed values (OR)
    conditions: list[str] = field(default_factory=list)          # all must be active (AND)

    def is_empty(self) -> bool:
        return not self.values and not self.conditions

    def as_dict(self) -> dict:
        out: dict[str, Any] = {k: (v[0] if len(v) == 1 else v) for k, v in self.values.items()}
        if self.conditions:
            out["conditions"] = self.conditions
        return out


def _as_list(v: Any) -> list:
    return list(v) if isinstance(v, (list, tuple, set)) else [v]


def _match(value: Any, allowed: list[str], key: str) -> str:
    if not isinstance(value, str):
        raise FilterError(f"Filter '{key}' expects text values.", {"value": value})
    lookup = {a.casefold(): a for a in allowed}
    if key == "season":
        lookup["autumn"] = "Fall"
    hit = lookup.get(value.strip().casefold())
    if hit is None:
        raise FilterError(f"Unknown value '{value}' for filter '{key}'.", {"allowed": allowed})
    return hit


def normalize_filters(raw: dict[str, Any] | None, known_cities: set[str] | None = None) -> Filters:
    """Validate and canonicalise a filter dict. Raises FilterError on bad input."""
    f = Filters()
    if not raw:
        return f
    unknown = set(raw) - set(SUPPORTED)
    if unknown:
        raise FilterError(f"Unsupported filter(s): {sorted(unknown)}.", {"supported": SUPPORTED})

    vocab = {"season": SEASONS, "weather": WEATHER, "severity": SEVERITIES, "priority": PRIORITIES,
             "category": all_categories()}
    for key, value in raw.items():
        if value is None or (isinstance(value, (list, str)) and len(value) == 0):
            continue
        if key == "conditions":
            names = set(load_conditions())
            conds = [str(c).strip().lower() for c in _as_list(value)]
            bad = [c for c in conds if c not in names]
            if bad:
                raise FilterError(f"Unknown condition(s): {bad}.", {"allowed": sorted(names)})
            f.conditions = sorted(set(conds))
        elif key == "issue_reported":
            if isinstance(value, bool):
                f.values[key] = [value]
            elif isinstance(value, str) and value.strip().lower() in ("yes", "no", "true", "false"):
                f.values[key] = [value.strip().lower() in ("yes", "true")]
            else:
                raise FilterError("Filter 'issue_reported' expects true/false or yes/no.")
        elif key == "state":
            codes = []
            for v in _as_list(value):
                code = to_state_code(str(v))
                if code is None:
                    raise FilterError(f"Unknown state '{v}'.", {"allowed": sorted(STATE_NAMES)})
                codes.append(code)
            f.values[key] = sorted(set(codes))
        elif key == "city":
            cities = []
            for v in _as_list(value):
                name = " ".join(str(v).split())
                if known_cities is not None:
                    match = {c.casefold(): c for c in known_cities}.get(name.casefold())
                    if match is None:
                        raise FilterError(f"Unknown city '{v}'.")
                    name = match
                cities.append(name)
            f.values[key] = sorted(set(cities))
        else:
            f.values[key] = sorted({_match(v, vocab[key], key) for v in _as_list(value)})
    return f


def filter_mask(df: pd.DataFrame, f: Filters) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    for key, allowed in f.values.items():
        col = df[COLUMN[key]]
        if key == "issue_reported":
            mask &= col.fillna(False).astype(bool).isin(allowed).to_numpy() & col.notna().to_numpy()
        else:
            mask &= col.isin(allowed).to_numpy()
    for c in f.conditions:
        mask &= df[f"cond_{c}"].to_numpy(dtype=bool)
    return mask


def chroma_where(f: Filters) -> dict | None:
    clauses = []
    meta_key = {"category": "primary_category"}
    for key, allowed in f.values.items():
        mk = meta_key.get(key, key)
        clauses.append({mk: allowed[0]} if len(allowed) == 1 else {mk: {"$in": allowed}})
    for c in f.conditions:
        clauses.append({f"cond_{c}": True})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def record_matches(rec: dict, f: Filters) -> bool:
    """Single-record check used as the final guarantee on returned results."""
    for key, allowed in f.values.items():
        if rec.get(COLUMN[key]) not in allowed:
            return False
    return all(bool(rec.get(f"cond_{c}")) for c in f.conditions)
