"""Derived severity (Phase 5) - a documented, configurable points score.

The vectorised `severity_frame` (whole dataset) and `score_record` (one
record / user input) share the same core, so there is one implementation.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.preprocessing.conditions import Condition, condition_frame, load_conditions
from src.utils.config import load_yaml

LEVEL_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _config() -> dict:
    return load_yaml("severity.yaml")


def _position(values: pd.Series, c: Condition) -> pd.Series:
    """0 at the threshold, 1 at (or beyond) the extreme, in the adverse direction."""
    span = abs(c.extreme - c.threshold) or 1.0
    dist = (values - c.threshold) if c.adverse_high else (c.threshold - values)
    return (dist / span).clip(0, 1)


def _level_from_score(score: np.ndarray, levels: list[dict]) -> np.ndarray:
    out = np.full(score.shape, "LOW", dtype=object)
    for lvl in sorted(levels, key=lambda l: l["min_score"]):
        out[score >= lvl["min_score"]] = lvl["name"]
    return out


def _max_level(a: np.ndarray, b: str, mask: np.ndarray) -> np.ndarray:
    rank = {n: i for i, n in enumerate(LEVEL_ORDER)}
    out = a.copy()
    upgrade = mask & np.array([rank[x] < rank[b] for x in a])
    out[upgrade] = b
    return out


def severity_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Columns: severity, severity_score, plus per-indicator points (pts_*)."""
    cfg = _config()
    conditions = load_conditions()
    flags = condition_frame(df, conditions)
    pts = {}
    at_extreme = {}
    for name, max_pts in cfg["points"].items():
        c = conditions[name]
        values = pd.to_numeric(df[c.metric], errors="coerce").astype("float64")
        pos = _position(values, c).fillna(0)
        badness = np.where(flags[name], 0.5 + 0.5 * pos, 0.0)
        pts[name] = max_pts * badness
        at_extreme[name] = flags[name].to_numpy() & (pos.to_numpy() >= 1.0)

    pts_df = pd.DataFrame(pts, index=df.index)
    issue = df["issue_reported"].fillna(False).astype(bool).to_numpy() if "issue_reported" in df else np.zeros(len(df), bool)
    score = np.minimum(100.0, pts_df.sum(axis=1).to_numpy() + cfg["issue_reported_points"] * issue)
    level = _level_from_score(score, cfg["levels"])

    ov = cfg["overrides"]
    ext = pd.DataFrame(at_extreme, index=df.index)
    big = [n for n, p in cfg["points"].items() if p >= ov["extreme_single_min_points"]]
    level = _max_level(level, ov["extreme_single_level"], ext[big].any(axis=1).to_numpy())
    level = _max_level(level, ov["extreme_multi_level"], (ext.sum(axis=1) >= ov["extreme_multi_count"]).to_numpy())

    out = pd.DataFrame({"severity": level, "severity_score": np.round(score, 1)}, index=df.index)
    return pd.concat([out, pts_df.add_prefix("pts_").round(2), ext.add_prefix("ext_")], axis=1)


def score_record(metrics: Mapping[str, Any]) -> dict:
    """Severity for one record or user-supplied metrics, with human-readable reasons."""
    conditions = load_conditions()
    row = pd.DataFrame([dict(metrics)])
    for c in conditions.values():
        if c.metric not in row:
            row[c.metric] = np.nan
    res = severity_frame(row).iloc[0]
    reasons = []
    for name, c in conditions.items():
        p = res.get(f"pts_{name}", 0.0)
        if p and p > 0:
            value = float(row[c.metric].iloc[0])
            note = ", at dataset extreme" if bool(res.get(f"ext_{name}", False)) else ""
            reasons.append({
                "condition": name,
                "reason": f"{c.label}: {value:g} (threshold {c.operator} {c.threshold:g}{note})",
                "points": round(float(p), 1),
            })
    reasons.sort(key=lambda r: -r["points"])
    if bool(metrics.get("issue_reported")):
        reasons.append({"condition": "issue_reported", "reason": "Customer issue reported",
                        "points": float(_config()["issue_reported_points"])})
    return {
        "severity": res["severity"],
        "severity_score": float(res["severity_score"]),
        "reasons": [r["reason"] for r in reasons],
        "contributions": reasons,
    }


def estimate_from_conditions(stated: list[str], measured: Mapping[str, Any] | None = None) -> dict:
    """Severity when some conditions are only *stated* (no value supplied).

    Measured values are scored exactly as in `score_record`. Each stated-only
    condition earns half its points (the value at the threshold), because how
    far beyond the threshold it is cannot be known. The result is labelled as
    an estimate.
    """
    cfg = _config()
    measured = dict(measured or {})
    base = score_record(measured) if measured else {"severity_score": 0.0, "reasons": [], "contributions": []}
    measured_conds = {c["condition"] for c in base["contributions"]}
    conditions = load_conditions()
    extra = []
    for name in stated:
        if name in measured_conds or name not in cfg["points"]:
            continue
        pts = 0.5 * cfg["points"][name]
        extra.append({"condition": name, "reason": f"{conditions[name].label} (stated, value not supplied)",
                      "points": round(pts, 1)})
    score = min(100.0, base["severity_score"] + sum(e["points"] for e in extra))
    level = _level_from_score(np.array([score]), cfg["levels"])[0]
    if measured and LEVEL_ORDER.index(base.get("severity", "LOW")) > LEVEL_ORDER.index(level):
        level = base["severity"]          # keep overrides from measured extremes
    contributions = sorted(base["contributions"] + extra, key=lambda r: -r["points"])
    return {
        "severity": level,
        "severity_score": round(score, 1),
        "reasons": [c["reason"] for c in contributions],
        "contributions": contributions,
        "basis": "measured" if not extra else ("stated conditions (estimated)" if not measured_conds
                                               else "measured values + stated conditions (estimated)"),
    }
