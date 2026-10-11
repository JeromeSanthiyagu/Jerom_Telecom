"""Phases 19-22 tests: fault-risk regression, leakage prevention, throughput model, classifier."""

import json

import pytest

from src.services.models.common import LeakageError, check_leakage, load_model, reports_dir
from src.services.models.fault_risk import predict_risk, risk_level

trained = pytest.mark.skipif(load_model("fault_risk") is None, reason="models not trained")

FORMULA = ["critical_alarm_count", "auth_failures", "security_incidents", "alarm_count", "call_drop_pct",
           "config_success_pct", "parameter_changes"]
DERIVED = ["cond_high_fault_rate", "severity", "severity_score", "priority", "primary_category", "conditions",
           "n_conditions", "search_text", "dq_availability_at_cap"]


# ------------------------------------------------------------------ leakage guard (unit)
@pytest.mark.parametrize("col", ["fault_rate_pct", "mttr_hours"] + FORMULA + DERIVED + ["city", "state", "zip",
                                                                                        "record_id"])
def test_leakage_guard_rejects_forbidden_fault_features(col):
    with pytest.raises(LeakageError):
        check_leakage(["latency_ms", col], "fault_risk")


def test_leakage_guard_accepts_primary_features():
    check_leakage(["season_Winter", "weather_Snow", "latency_ms", "throughput_mbps", "user_count"], "fault_risk")


def test_throughput_model_cannot_use_its_mirror():
    for col in ("throughput_mbps", "critical_alarm_count", "fault_rate_pct"):
        with pytest.raises(LeakageError):
            check_leakage([col], "performance")


def test_risk_bands():
    assert risk_level(10) == "Low" and risk_level(40) == "Medium"
    assert risk_level(50) == "High" and risk_level(70) == "Critical"


# ------------------------------------------------------------------ trained models
@trained
def test_served_fault_model_has_no_leaking_features():
    meta = load_model("fault_risk")["meta"]
    feats = set(meta["raw_features"])
    assert not feats & set(FORMULA + DERIVED + ["fault_rate_pct", "mttr_hours", "city", "state", "zip"])
    check_leakage(meta["encoded_features"], "fault_risk")


@trained
def test_fault_model_report_documents_all_feature_sets():
    rep = json.loads((reports_dir() / "fault_risk.json").read_text())
    sets = rep["feature_sets"]
    assert sets["C_primary"]["models"]["hist_gradient_boosting"]["r2"] > 0.85
    assert 0.4 < sets["D_strict"]["models"]["hist_gradient_boosting"]["r2"] < 0.6
    assert sets["A_formula_demo"]["models"]["linear_regression"]["r2"] > 0.999   # target is formulaic
    for m in ("mae", "rmse", "r2"):
        assert m in sets["C_primary"]["models"]["linear_regression"]
    assert set(sets["C_primary"]["models"]) == {"linear_regression", "hist_gradient_boosting", "random_forest"}


@trained
def test_prediction_is_labelled_and_handles_partial_inputs():
    full = predict_risk({"weather": "Snow", "season": "Winter", "throughput_mbps": 250, "latency_ms": 140})
    empty = predict_risk({})
    assert full["available"] and "ML prediction" in full["label"]
    assert full["predicted_fault_rate_pct"] > empty["predicted_fault_rate_pct"]
    assert empty["input_coverage"] == 0 and empty["low_input_coverage"]
    lo, hi = full["interval_80pct"]
    assert lo <= full["predicted_fault_rate_pct"] <= hi


@trained
def test_unknown_category_is_missing_not_reference_class():
    from src.services.models.common import encode
    import pandas as pd
    X = encode(pd.DataFrame([{"weather": None}]), ["weather"])
    assert X.isna().all(axis=None)


@trained
def test_throughput_model_report():
    rep = json.loads((reports_dir() / "performance_throughput.json").read_text())
    assert rep["models"]["hist_gradient_boosting"]["r2"] > rep["mean_predictor_baseline"]["r2"]
    assert "critical_alarm_count" not in rep["features"] and "fault_rate_pct" not in rep["features"]


@trained
def test_classifier_is_documented_as_derived_labels():
    rep = json.loads((reports_dir() / "incident_classifier.json").read_text())
    assert "NOT ground truth" in rep["label_source"]
    for m in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        assert m in rep["models"]["hist_gradient_boosting"]
    labels = rep["confusion_matrix"]["labels"]
    assert len(rep["confusion_matrix"]["matrix"]) == len(labels)
    assert set(rep["models"]) == {"logistic_regression", "random_forest", "hist_gradient_boosting"}
