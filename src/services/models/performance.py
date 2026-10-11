"""KPI performance prediction (Phase 21): throughput from context + radio/transport KPIs.

Chosen because Phase 1 showed throughput is the only KPI with a meaningful,
non-leaking signal (R^2 ~0.33 from context + radio/transport, mostly weather).
Latency (0.13), call drop (0.19) and handover success (0.0) are not
predictable from the other columns. No time-series model: there are no
timestamps.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.services.models.common import (check_leakage, cfg, encode, load_model, regression_metrics,
                               resolve_feature_set, save_model, write_report)

logger = logging.getLogger(__name__)
TASK = "performance"
MODEL_NAME = "throughput"


def train(df: pd.DataFrame) -> dict:
    c = cfg()
    tcfg = c[TASK]
    target = tcfg["target"]
    raw = resolve_feature_set(tcfg["feature_sets"][tcfg["served"]], df.columns, target)
    check_leakage(raw, TASK)
    X, y = encode(df, raw), df[target].astype("float64")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=c["test_size"], random_state=c["random_state"])

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=10, max_features=0.5,
                                               n_jobs=-1, random_state=c["random_state"]),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                                random_state=c["random_state"]),
    }
    results, fitted = {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        results[name] = regression_metrics(yte, m.predict(Xte))
        fitted[name] = m
        logger.info("throughput / %s: %s", name, results[name])
    baseline = regression_metrics(yte, np.full(len(yte), ytr.mean()))
    served = fitted["hist_gradient_boosting"]
    resid = yte.to_numpy() - served.predict(Xte)
    meta = {"target": target, "raw_features": raw, "encoded_features": list(X.columns),
            "algorithm": "HistGradientBoostingRegressor", "test_metrics": results["hist_gradient_boosting"],
            "residual_p10": float(np.quantile(resid, 0.1)), "residual_p90": float(np.quantile(resid, 0.9))}
    save_model(MODEL_NAME, served, meta)
    report = {"target": target, "features": raw, "excluded_features": tcfg["excluded_features"],
              "models": results, "mean_predictor_baseline": baseline, "served": meta,
              "interpretation": "Moderate explanatory power, driven mainly by weather; treat predictions "
                                "as indicative ranges, not precise forecasts."}
    write_report("performance_throughput", report)
    return report


def predict_throughput(inputs: dict[str, Any]) -> dict:
    art = load_model(MODEL_NAME)
    if art is None:
        return {"available": False, "reason": "Throughput model not trained."}
    meta = art["meta"]
    X = encode(pd.DataFrame([inputs]), meta["raw_features"])
    pred = float(max(0.0, art["model"].predict(X)[0]))
    provided = [f for f in meta["raw_features"] if inputs.get(f) is not None]
    return {"available": True, "predicted_throughput_mbps": round(pred, 1),
            "interval_80pct": [round(max(0.0, pred + meta["residual_p10"]), 1), round(pred + meta["residual_p90"], 1)],
            "inputs_used": provided, "test_r2": meta["test_metrics"]["r2"],
            "label": "Predicted throughput (ML prediction)"}
