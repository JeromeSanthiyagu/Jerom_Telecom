"""Numerical / attribute similarity between a parsed query and records (Phase 13).

Per constraint, a similarity in [0, 1]:
  stated value (eq)   1 - |x - v| / (P99 - P01)                 clipped at 0
  "above v" (gt)      1 if x > v, else decays over the same range
  "below v" (lt)      mirror of gt
  adverse             0.5 * adverse percentile rank of x  +  0.5 * [record has the condition]
  favorable           0.5 * (1 - adverse percentile rank) +  0.5 * [record lacks the condition]
  normal              1 - 2 * |percentile rank - 0.5|
  context (weather / season / state mentioned in the text)  1 if equal else 0

metric_score = mean of all constraint similarities. The condition indicator makes
the documented threshold (thresholds.yaml) decisive: "low throughput" prefers
records that actually cross the low-throughput threshold over ones just above it. Percentile ranks are
computed against the full dataset, so "high" always means "high relative to
this network's own history".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.rag.query_parser import QueryUnderstanding, _adverse_high
from src.preprocessing.conditions import load_conditions


class MetricScorer:
    def __init__(self, incidents: pd.DataFrame, metrics: list[str]):
        self.values: dict[str, np.ndarray] = {}
        self.ranks: dict[str, np.ndarray] = {}
        self.ranges: dict[str, float] = {}
        for m in metrics:
            s = pd.to_numeric(incidents[m], errors="coerce").astype("float64")
            self.values[m] = s.to_numpy()
            self.ranks[m] = s.rank(pct=True).to_numpy()
            self.ranges[m] = float(s.quantile(0.99) - s.quantile(0.01)) or 1.0
        self.context = {k: incidents[k].astype(str).to_numpy() for k in ("weather", "season", "state")}
        self.flags = {c.metric: incidents[f"cond_{name}"].to_numpy(dtype=float)
                      for name, c in load_conditions().items() if f"cond_{name}" in incidents}

    def score(self, q: QueryUnderstanding, rows: np.ndarray | None = None) -> np.ndarray | None:
        """Mean similarity per row (all rows, or the given row positions). None if no signal."""
        parts = []
        for c in q.constraints:
            if c.metric not in self.values:
                continue
            x = self.values[c.metric] if rows is None else self.values[c.metric][rows]
            r = self.ranks[c.metric] if rows is None else self.ranks[c.metric][rows]
            rng = self.ranges[c.metric]
            if c.value is not None:
                if c.comparator == "gt":
                    sim = np.where(x > c.value, 1.0, 1 - (c.value - x) / rng)
                elif c.comparator == "lt":
                    sim = np.where(x < c.value, 1.0, 1 - (x - c.value) / rng)
                else:
                    sim = 1 - np.abs(x - c.value) / rng
            elif c.direction in ("adverse", "favorable"):
                adverse_rank = r if _adverse_high(c.metric) else 1 - r
                flag = self.flags.get(c.metric)
                flag = (flag if rows is None else flag[rows]) if flag is not None else adverse_rank
                sim = 0.5 * adverse_rank + 0.5 * flag
                if c.direction == "favorable":
                    sim = 1 - sim
            elif c.direction == "normal":
                sim = 1 - 2 * np.abs(r - 0.5)
            else:
                continue
            parts.append(np.nan_to_num(np.clip(sim, 0, 1), nan=0.0))
        for key, val in q.context.items():
            col = self.context[key] if rows is None else self.context[key][rows]
            parts.append((col == val).astype(float))
        if not parts:
            return None
        return np.mean(parts, axis=0)
