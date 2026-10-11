"""Derived network conditions (Phase 3).

A condition is a named, documented boolean derived from ONE raw metric and a
threshold from `src/config/thresholds.yaml`, e.g.

    high_latency  <=>  latency_ms > 129.9218   (dataset P90)

The same vectorised function serves the whole dataset and single records
(API input), so there is exactly one implementation of every rule. A missing
metric never triggers a condition; it is reported as unavailable instead.
"""

from __future__ import annotations

import operator as op
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from src.utils.config import load_yaml

OPERATORS = {"<": op.lt, ">": op.gt, "<=": op.le, ">=": op.ge}


@dataclass(frozen=True)
class Condition:
    name: str
    metric: str
    label: str
    operator: str
    threshold: float
    extreme: float
    method: str
    group: str

    @property
    def adverse_high(self) -> bool:
        return self.operator in (">", ">=")

    def describe(self, value: float | None = None) -> str:
        base = f"{self.metric} {self.operator} {self.threshold:g}"
        return f"{self.label} ({value:g} vs threshold {base})" if value is not None else f"{self.label} ({base})"


def load_conditions() -> dict[str, Condition]:
    cfg = load_yaml("thresholds.yaml")["conditions"]
    return {
        name: Condition(
            name=name, metric=c["metric"], label=c["label"], operator=c["operator"],
            threshold=float(c["threshold"]), extreme=float(c["extreme"]),
            method=c["method"], group=c["group"],
        )
        for name, c in cfg.items()
    }


def condition_frame(df: pd.DataFrame, conditions: dict[str, Condition] | None = None) -> pd.DataFrame:
    """Boolean frame (one column per condition). NaN metric -> False."""
    conditions = conditions or load_conditions()
    out = {}
    for name, c in conditions.items():
        if c.metric not in df:
            out[name] = pd.Series(False, index=df.index)
            continue
        values = pd.to_numeric(df[c.metric], errors="coerce").astype("float64")
        out[name] = OPERATORS[c.operator](values, c.threshold).fillna(False).astype(bool)
    return pd.DataFrame(out, index=df.index)


def condition_lists(flags: pd.DataFrame) -> pd.Series:
    """Row-wise list of active condition names, in config order."""
    names = list(flags.columns)
    arr = flags.to_numpy()
    return pd.Series([[n for n, f in zip(names, row) if f] for row in arr], index=flags.index)


def evaluate_record(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Conditions for one record or one set of user-supplied metrics.

    Returns the active conditions with the evidence behind each one, plus the
    conditions that could not be evaluated because the metric is missing.
    """
    conditions = load_conditions()
    row = pd.DataFrame([{k: v for k, v in metrics.items()}])
    flags = condition_frame(row, conditions).iloc[0]
    active, unavailable = [], []
    for name, c in conditions.items():
        value = metrics.get(c.metric)
        if value is None or pd.isna(value):
            unavailable.append(name)
        elif flags[name]:
            active.append({
                "condition": name,
                "label": c.label,
                "metric": c.metric,
                "value": float(value),
                "operator": c.operator,
                "threshold": c.threshold,
                "method": c.method,
            })
    return {"conditions": [a["condition"] for a in active], "evidence": active,
            "unavailable": unavailable}
