"""Shared ML helpers: feature-set resolution, encoding, leakage guard, artifacts."""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from src.utils.config import PROJECT_ROOT, load_yaml

SEASONS = ["Winter", "Spring", "Summer", "Fall"]
WEATHER = ["Clear", "Clouds", "Rain", "Snow"]


class LeakageError(ValueError):
    pass


def cfg() -> dict:
    return load_yaml("models.yaml")


def artifacts_dir() -> Path:
    p = PROJECT_ROOT / cfg()["artifacts_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def reports_dir() -> Path:
    p = PROJECT_ROOT / cfg()["reports_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_feature_set(spec: list[str] | str, all_columns: Iterable[str], target: str) -> list[str]:
    """Expand group names into raw column names (before one-hot encoding)."""
    groups = cfg()["feature_groups"]
    if spec == "ALL_EXCEPT_TARGET":
        excluded = cfg()["always_excluded"]
        return [c for c in all_columns if c != target
                and not any(fnmatch.fnmatch(c, pat) for pat in excluded)]
    cols: list[str] = []
    for item in spec:
        cols.extend(groups.get(item, [item]))
    return cols


def check_leakage(features: list[str], task: str, allow_formula_demo: bool = False) -> None:
    """Raise if a feature list contains the target, a derived label or an excluded column."""
    c = cfg()
    excluded_patterns = list(c["always_excluded"])
    task_cfg = c[task]
    target = task_cfg["target"]
    task_excluded = set(task_cfg.get("excluded_features", {}))
    for f in features:
        base = encoded_base(f)
        if base == target:
            raise LeakageError(f"{task}: target '{target}' used as a feature")
        if any(fnmatch.fnmatch(base, pat) for pat in excluded_patterns):
            raise LeakageError(f"{task}: excluded column '{base}' used as a feature")
        if base in task_excluded and not allow_formula_demo:
            raise LeakageError(f"{task}: '{base}' is excluded: {task_cfg['excluded_features'][base]}")


def encoded_base(col: str) -> str:
    for prefix in ("season_", "weather_"):
        if col.startswith(prefix) and col[len(prefix):] in SEASONS + WEATHER:
            return prefix[:-1]
    return col


def encode(df: pd.DataFrame, raw_features: list[str]) -> pd.DataFrame:
    """Numeric matrix. season/weather -> one-hot; unknown category -> NaN in every one-hot column
    (HistGradientBoosting treats NaN as missing instead of as a fake category)."""
    out = {}
    for f in raw_features:
        if f in ("season", "weather"):
            values = SEASONS if f == "season" else WEATHER
            col = df[f] if f in df else pd.Series([None] * len(df), index=df.index)
            known = col.isin(values)
            for v in values:
                out[f"{f}_{v}"] = np.where(known, (col == v).astype(float), np.nan)
        else:
            col = df[f] if f in df else pd.Series([np.nan] * len(df), index=df.index)
            out[f] = pd.to_numeric(col, errors="coerce").astype("float64").to_numpy()
    return pd.DataFrame(out, index=df.index)


def save_model(name: str, model: Any, meta: dict) -> None:
    joblib.dump({"model": model, "meta": meta}, artifacts_dir() / f"{name}.joblib")


def load_model(name: str) -> dict | None:
    path = artifacts_dir() / f"{name}.joblib"
    return joblib.load(path) if path.exists() else None


def write_report(name: str, report: dict) -> None:
    (reports_dir() / f"{name}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def regression_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }
