"""Phases 4-8 tests: taxonomy, severity, priority, searchable text, record IDs."""

import re

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.build_incidents import INCIDENTS_PATH, derive
from src.preprocessing.chunking import NA_TEXT, build_search_text
from src.preprocessing.priority import priority_components
from src.preprocessing.severity import score_record
from src.preprocessing.taxonomy import all_categories, classify_conditions

built = pytest.mark.skipif(not INCIDENTS_PATH.exists(), reason="incidents not built")


# ----------------------------- taxonomy (Phase 4) ---------------------------- #
def test_no_conditions_is_normal():
    assert classify_conditions([])["primary_category"] == "Normal"


@pytest.mark.parametrize("conds,expected", [
    (["low_handover", "high_call_drop"], "Mobility / Handover"),
    (["weak_signal", "high_call_drop"], "Radio / Signal Quality"),
    (["high_load", "high_latency"], "Congestion / Load"),
    (["high_latency", "high_packet_loss"], "Service Degradation"),
    (["low_availability"], "Availability"),
    (["low_setup_success"], "Access / Connection Setup"),
    (["security_risk"], "Security"),
    (["low_config_success"], "Configuration"),
    (["high_critical_alarm_count", "high_fault_rate"], "Fault / Alarm"),
])
def test_rule_examples(conds, expected):
    assert classify_conditions(conds)["primary_category"] == expected


def test_requires_any_blocks_ineligible_category():
    # high_latency alone must not create a Congestion category (needs load)
    res = classify_conditions(["high_latency"])
    assert "Congestion / Load" not in res["scores"]


def test_tie_break_is_deterministic():
    a = classify_conditions(["security_risk", "low_config_success"])
    b = classify_conditions(["low_config_success", "security_risk"])
    assert a == b and a["primary_category"] == "Security"  # listed before Configuration


def test_reasons_name_contributing_conditions():
    res = classify_conditions(["low_handover", "high_call_drop"])
    assert "low_handover" in res["reasons"][0] and "high_call_drop" in res["reasons"][0]


# ----------------------------- severity (Phase 5) ---------------------------- #
def test_severity_is_low_when_nothing_is_wrong():
    res = score_record({"latency_ms": 50.0, "packet_loss_pct": 2.0, "issue_reported": False})
    assert res["severity"] == "LOW" and res["severity_score"] == 0 and res["reasons"] == []


def test_severity_grows_with_degradation():
    mild = score_record({"latency_ms": 131.0})
    bad = score_record({"latency_ms": 170.0})
    assert 0 < mild["severity_score"] < bad["severity_score"]
    assert "High latency" in bad["reasons"][0]


def test_two_extremes_are_critical():
    res = score_record({"cell_availability_pct": 75.0, "call_drop_pct": 6.0})
    assert res["severity"] == "CRITICAL"


def test_single_extreme_major_indicator_is_at_least_high():
    assert score_record({"cell_availability_pct": 75.0})["severity"] in ("HIGH", "CRITICAL")


def test_severity_ignores_user_count():
    assert score_record({"user_count": 9999})["severity_score"] == 0


# ----------------------------- priority (Phase 6) ---------------------------- #
def test_priority_adds_impact_and_lists_reasons():
    low = priority_components(10, "LOW", 100, False, False, False)
    high = priority_components(10, "LOW", 9900, True, True, False, fault_risk_level="Critical")
    assert low["priority"] == "P4"
    assert high["priority_score"] > low["priority_score"]
    assert any("Customer issue reported" in r for r in high["reasons"])


def test_priority_floor_for_critical_severity():
    assert priority_components(55, "CRITICAL", None, False, False, False)["priority"] in ("P1", "P2")


def test_priority_never_claims_response_time():
    res = priority_components(80, "CRITICAL", 9000, True, True, True, "Critical")
    assert not any(re.search(r"\bwithin\b|\bhour", r, re.I) for r in res["reasons"])


# ------------------------- searchable text (Phase 7) ------------------------- #
def _numbers(text: str) -> list[str]:
    text = re.sub(r"ROW-\d{6}", "", text)
    text = re.sub(r"\bP[1-4]\b", "", text)
    return re.findall(r"-?\d+(?:\.\d+)?", text)


@built
def test_every_number_in_text_comes_from_the_record():
    inc = pd.read_parquet(INCIDENTS_PATH).sample(500, random_state=0)
    numeric = [c for c in inc.columns if c not in ("source_row", "n_conditions", "priority_score")
               and pd.api.types.is_numeric_dtype(inc[c]) and not pd.api.types.is_bool_dtype(inc[c])]
    for rec in inc.to_dict("records"):
        allowed = set()
        for c in numeric:
            v = rec[c]
            if v is None or pd.isna(v):
                continue
            for d in (0, 1, 2):
                allowed.add(f"{float(v):.{d}f}")
            allowed.add(str(int(round(float(v)))))
        for n in _numbers(rec["search_text"]):
            assert n in allowed, (rec["record_id"], n)


def test_missing_values_are_stated_not_invented():
    rec = {"record_id": "ROW-000001", "conditions": [], "primary_category": "Normal",
           "secondary_categories": [], "severity": "LOW", "severity_score": 0.0, "priority": "P4",
           "latency_ms": None, "city": None, "state": "MN"}
    text = build_search_text(rec)
    assert f"latency {NA_TEXT}" in text
    assert "Minnesota" in text


@built
def test_text_labels_generated_content_as_derived():
    text = pd.read_parquet(INCIDENTS_PATH, columns=["search_text"]).iloc[0, 0]
    assert "generated from dataset fields" in text
    for line in ("Derived conditions:", "Derived category:", "Derived severity:"):
        assert line in text


# -------------------------- record IDs (Phase 8) ----------------------------- #
@built
def test_derived_table_keeps_ids_and_raw_values():
    from src.preprocessing.loader import load_processed
    proc = load_processed()
    inc = pd.read_parquet(INCIDENTS_PATH)
    assert inc["record_id"].tolist() == proc["record_id"].tolist()
    for col in ("latency_ms", "fault_rate_pct", "user_count", "weather"):
        assert inc[col].equals(proc[col])


@built
def test_derived_labels_use_documented_vocabulary():
    inc = pd.read_parquet(INCIDENTS_PATH, columns=["primary_category", "severity", "priority"])
    assert set(inc["primary_category"]) <= set(all_categories())
    assert set(inc["severity"]) <= {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert set(inc["priority"]) <= {"P1", "P2", "P3", "P4"}


def test_vectorised_and_single_record_paths_agree():
    from tests.conftest import BASE_ROW  # noqa: F401  (realistic values)
    df = pd.DataFrame([
        {"record_id": "ROW-000001", "source_row": 1, "latency_ms": 150.0, "packet_loss_pct": 13.0,
         "cell_availability_pct": 78.0, "user_count": 9500, "issue_reported": True},
    ])
    for c in ["throughput_mbps", "call_drop_pct", "handover_success_pct", "signal_strength_dbm",
              "jitter_ms", "setup_success_pct", "data_usage_gb", "alarm_count", "critical_alarm_count",
              "config_success_pct", "parameter_changes", "security_incidents", "auth_failures",
              "fault_rate_pct", "city", "state", "season", "weather", "temperature_c", "humidity_pct", "mttr_hours"]:
        df[c] = np.nan
    row = derive(df).iloc[0]
    single = score_record(df.iloc[0].to_dict())
    assert row["severity"] == single["severity"]
    assert row["severity_score"] == single["severity_score"]
