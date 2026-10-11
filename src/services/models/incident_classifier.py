"""Incident category classifier (Phase 22).

IMPORTANT: the labels are the rule-derived `primary_category` (taxonomy.yaml),
NOT human-annotated ground truth. Training a model on them measures how well
the rules can be learned from raw metrics; "accuracy" means agreement with
the rules. The rule-based category stays authoritative in the application;
the classifier is reported alongside it as a consistency check and gives
class probabilities when inputs are partial.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.services.models.common import check_leakage, cfg, encode, load_model, resolve_feature_set, save_model, write_report

logger = logging.getLogger(__name__)
TASK = "incident_classifier"
MODEL_NAME = "incident_classifier"


def _eval(y_true, y_pred, labels) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_precision": round(float(p), 4), "macro_recall": round(float(r), 4), "macro_f1": round(float(f), 4)}


def train(df: pd.DataFrame) -> dict:
    c = cfg()
    tcfg = c[TASK]
    target = tcfg["target"]
    raw = resolve_feature_set(tcfg["feature_sets"][tcfg["served"]], df.columns, target)
    check_leakage(raw, TASK)
    X, y = encode(df, raw), df[target].astype(str)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=c["test_size"], random_state=c["random_state"], stratify=y)
    labels = sorted(y.unique())

    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1,
                                                random_state=c["random_state"]),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                                 random_state=c["random_state"]),
    }
    results, preds = {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        preds[name] = m.predict(Xte)
        results[name] = _eval(yte, preds[name], labels)
        logger.info("classifier / %s: %s", name, results[name])

    served_name = "hist_gradient_boosting"
    served = models[served_name]
    cm = confusion_matrix(yte, preds[served_name], labels=labels)
    report = {
        "target": target,
        "label_source": "Rule-derived categories (src/config/taxonomy.yaml) - NOT ground truth. "
                        "Scores measure agreement between the ML model and the rules.",
        "features": raw,
        "class_distribution_test": yte.value_counts().to_dict(),
        "models": results,
        "served": served_name,
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        "per_class": classification_report(yte, preds[served_name], labels=labels, output_dict=True, zero_division=0),
    }
    save_model(MODEL_NAME, served, {"raw_features": raw, "encoded_features": list(X.columns), "labels": labels,
                                     "test_metrics": results[served_name]})
    write_report(MODEL_NAME, report)
    return report


def predict_category(inputs: dict[str, Any]) -> dict:
    art = load_model(MODEL_NAME)
    if art is None:
        return {"available": False, "reason": "Classifier not trained."}
    meta, model = art["meta"], art["model"]
    X = encode(pd.DataFrame([inputs]), meta["raw_features"])
    proba = model.predict_proba(X)[0]
    order = np.argsort(-proba)
    provided = [f for f in meta["raw_features"] if inputs.get(f) is not None]
    return {
        "available": True,
        "predicted_category": str(model.classes_[order[0]]),
        "probability": round(float(proba[order[0]]), 3),
        "top3": [{"category": str(model.classes_[i]), "probability": round(float(proba[i]), 3)} for i in order[:3]],
        "input_coverage": round(len(provided) / len(meta["raw_features"]), 2),
        "label": "ML estimate trained on rule-derived labels (not ground truth)",
    }
