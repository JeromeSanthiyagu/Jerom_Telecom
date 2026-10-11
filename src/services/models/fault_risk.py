"""Network fault-risk model (Phases 19-20).

Target: `fault_rate_pct` (continuous, 99,976 distinct values) -> REGRESSION,
then the *predicted* rate is mapped to documented risk bands. The historical
fault occurrence rate of a record is never called a prediction.

Feature sets (models.yaml):
  C_primary       context + radio/transport KPIs + throughput      (served)
  D_strict        same without throughput (throughput mirrors critical alarms)
  A_formula_demo  every column except the target and derived labels - shows
                  the target is a near-deterministic formula; never served
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from src.services.models.common import (check_leakage, cfg, encode, load_model, regression_metrics,
                               resolve_feature_set, save_model, write_report)

logger = logging.getLogger(__name__)
TASK = "fault_risk"
MODEL_NAME = "fault_risk"


def risk_level(predicted: float) -> str:
    for band in cfg()[TASK]["risk_bands"]:
        if band["below"] is None or predicted < band["below"]:
            return band["level"]
    return "Critical"


def _models(include_rf: bool):
    models = {
        "linear_regression": LinearRegression(),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                                random_state=cfg()["random_state"]),
    }
    if include_rf:
        models["random_forest"] = RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.5,
                                                        n_jobs=-1, random_state=cfg()["random_state"])
    return models


def train(df: pd.DataFrame) -> dict:
    c = cfg()
    tcfg = c[TASK]
    target = tcfg["target"]
    y = df[target].astype("float64")
    idx_train, idx_test = train_test_split(df.index, test_size=c["test_size"], random_state=c["random_state"])

    report: dict[str, Any] = {"target": target, "task": "regression", "rows_train": len(idx_train),
                              "rows_test": len(idx_test), "feature_sets": {}, "excluded_features": tcfg["excluded_features"]}
    served = None
    for set_name, spec in tcfg["feature_sets"].items():
        raw = resolve_feature_set(spec, df.columns, target)
        check_leakage(raw, TASK, allow_formula_demo=(set_name == "A_formula_demo"))
        X = encode(df, raw)
        check_leakage(list(X.columns), TASK, allow_formula_demo=(set_name == "A_formula_demo"))
        results = {}
        for name, model in _models(include_rf=set_name != "A_formula_demo").items():
            model.fit(X.loc[idx_train], y.loc[idx_train])
            pred = model.predict(X.loc[idx_test])
            results[name] = regression_metrics(y.loc[idx_test], pred)
            logger.info("%s / %s: %s", set_name, name, results[name])
            if set_name == tcfg["served"] and name == "hist_gradient_boosting":
                resid = y.loc[idx_test].to_numpy() - pred
                cv = cross_val_score(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                                   random_state=c["random_state"]),
                                     X.loc[idx_train], y.loc[idx_train], scoring="r2",
                                     cv=KFold(5, shuffle=True, random_state=c["random_state"]))
                served = {
                    "model": model, "raw_features": raw, "encoded_features": list(X.columns),
                    "residual_p10": float(np.quantile(resid, 0.10)), "residual_p90": float(np.quantile(resid, 0.90)),
                    "test_metrics": results[name],
                    "cv_r2_mean": round(float(cv.mean()), 4), "cv_r2_std": round(float(cv.std()), 4),
                }
        report["feature_sets"][set_name] = {"raw_features": raw, "n_encoded": X.shape[1], "models": results}

    meta = {k: v for k, v in served.items() if k != "model"}
    meta.update({"feature_set": tcfg["served"], "algorithm": "HistGradientBoostingRegressor",
                 "risk_bands": tcfg["risk_bands"], "target": target})
    save_model(MODEL_NAME, served["model"], meta)
    report["served"] = {k: v for k, v in meta.items()}
    report["note"] = ("Served model: HistGradientBoosting on the primary feature set. It handles missing "
                      "inputs natively, which matters because queries rarely state every KPI.")
    write_report(MODEL_NAME, report)
    return report


def predict_risk(inputs: dict[str, Any]) -> dict:
    """Predicted fault occurrence risk from whatever inputs are available."""
    art = load_model(MODEL_NAME)
    if art is None:
        return {"available": False, "reason": "Fault-risk model not trained (python -m src.services.models.train)."}
    meta, model = art["meta"], art["model"]
    X = encode(pd.DataFrame([inputs]), meta["raw_features"])
    provided = [f for f in meta["raw_features"] if inputs.get(f) is not None and not pd.isna(inputs.get(f))]
    coverage = len(provided) / len(meta["raw_features"])
    pred = float(np.clip(model.predict(X)[0], 0, 100))
    lo, hi = pred + meta["residual_p10"], pred + meta["residual_p90"]
    return {
        "available": True,
        "predicted_fault_rate_pct": round(pred, 2),
        "risk_level": risk_level(pred),
        "interval_80pct": [round(max(0.0, lo), 2), round(min(100.0, hi), 2)],
        "inputs_used": provided,
        "input_coverage": round(coverage, 2),
        "low_input_coverage": coverage < 0.4,
        "model": {"algorithm": meta["algorithm"], "feature_set": meta["feature_set"],
                  "test_r2": meta["test_metrics"]["r2"], "test_mae": meta["test_metrics"]["mae"]},
        "label": "Predicted fault occurrence risk (ML prediction, not an observed value)",
    }
