"""Incident priority P1-P4 (Phase 6): severity plus impact, fully itemised."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.utils.config import load_yaml

ORDER = ["P4", "P3", "P2", "P1"]


def _config() -> dict:
    return load_yaml("priority.yaml")


def priority_components(
    severity_score: float,
    severity: str,
    user_count: float | None,
    issue_reported: bool | None,
    low_availability: bool,
    high_critical_alarms: bool,
    fault_risk_level: str | None = None,
) -> dict[str, Any]:
    cfg = _config()
    parts: list[tuple[str, float]] = [(f"Severity {severity} (score {severity_score:g})", float(severity_score))]
    if user_count is not None and not pd.isna(user_count):
        pts = cfg["users_max_points"] * min(1.0, float(user_count) / cfg["users_reference"])
        parts.append((f"{int(user_count)} associated users", pts))
    if issue_reported:
        parts.append(("Customer issue reported", cfg["issue_reported_points"]))
    if low_availability:
        parts.append(("Cell availability below threshold (service impact)", cfg["availability_points"]))
    if high_critical_alarms:
        parts.append(("Critical alarms above threshold", cfg["critical_alarm_points"]))
    if fault_risk_level:
        parts.append((f"Predicted fault risk {fault_risk_level}", cfg["fault_risk_points"].get(fault_risk_level, 0)))

    score = round(sum(p for _, p in parts), 1)
    level = next(l for l in sorted(cfg["levels"], key=lambda l: -l["min_score"]) if score >= l["min_score"])
    name, label = level["name"], level["label"]
    floor = cfg.get("floors", {}).get(severity)
    if floor and ORDER.index(floor) > ORDER.index(name):
        name = floor
        label = next(l["label"] for l in cfg["levels"] if l["name"] == floor)
    return {
        "priority": name,
        "priority_label": label,
        "priority_score": score,
        "reasons": [f"{t} (+{p:.1f})" for t, p in parts if p > 0],
    }


def priority_frame(df: pd.DataFrame, sev: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Vectorised priority for stored records (no fault-risk prediction term)."""
    cfg = _config()
    users = pd.to_numeric(df["user_count"], errors="coerce").astype("float64")
    score = (
        sev["severity_score"].to_numpy()
        + cfg["users_max_points"] * np.minimum(1.0, users.fillna(0).to_numpy() / cfg["users_reference"])
        + cfg["issue_reported_points"] * df["issue_reported"].fillna(False).astype(bool).to_numpy()
        + cfg["availability_points"] * flags["low_availability"].to_numpy()
        + cfg["critical_alarm_points"] * flags["high_critical_alarm_count"].to_numpy()
    )
    score = np.round(score, 1)
    out = np.full(len(df), "P4", dtype=object)
    for lvl in sorted(cfg["levels"], key=lambda l: l["min_score"]):
        out[score >= lvl["min_score"]] = lvl["name"]
    for sev_level, floor in cfg.get("floors", {}).items():
        mask = (sev["severity"].to_numpy() == sev_level) & np.array([ORDER.index(x) < ORDER.index(floor) for x in out])
        out[mask] = floor
    return pd.DataFrame({"priority": out, "priority_score": score}, index=df.index)


def score_priority(metrics: Mapping[str, Any], severity: Mapping[str, Any], conditions: list[str],
                   fault_risk_level: str | None = None) -> dict:
    return priority_components(
        severity_score=severity["severity_score"],
        severity=severity["severity"],
        user_count=metrics.get("user_count"),
        issue_reported=bool(metrics.get("issue_reported")) if metrics.get("issue_reported") is not None else None,
        low_availability="low_availability" in conditions,
        high_critical_alarms="high_critical_alarm_count" in conditions,
        fault_risk_level=fault_risk_level,
    )
