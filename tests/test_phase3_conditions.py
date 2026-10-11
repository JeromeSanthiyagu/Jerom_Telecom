"""Phase 3 tests: derived network conditions."""

import math

import pandas as pd
import pytest

from src.preprocessing.conditions import condition_frame, condition_lists, evaluate_record, load_conditions
from src.preprocessing.loader import PROCESSED_PATH

processed = pytest.mark.skipif(not PROCESSED_PATH.exists(), reason="processed data not built")


def test_every_condition_has_documented_rule():
    for name, c in load_conditions().items():
        assert c.operator in ("<", ">", ">=", "<=")
        assert c.method in ("percentile", "domain")
        assert not math.isnan(c.threshold) and not math.isnan(c.extreme)
        # the extreme is further in the adverse direction than the threshold
        assert (c.extreme >= c.threshold) if c.adverse_high else (c.extreme <= c.threshold), name


def test_threshold_boundary_is_strict():
    c = load_conditions()["high_latency"]
    assert evaluate_record({"latency_ms": c.threshold})["conditions"] == []
    assert evaluate_record({"latency_ms": c.threshold + 0.01})["conditions"] == ["high_latency"]


def test_low_direction_conditions():
    res = evaluate_record({"handover_success_pct": 90.5, "signal_strength_dbm": -112.0})
    assert set(res["conditions"]) == {"low_handover", "weak_signal"}
    ev = {e["condition"]: e for e in res["evidence"]}
    assert ev["weak_signal"]["value"] == -112.0 and ev["weak_signal"]["threshold"] == -110.0


def test_missing_metric_never_triggers_and_is_reported():
    res = evaluate_record({"latency_ms": None, "packet_loss_pct": float("nan")})
    assert res["conditions"] == []
    assert {"high_latency", "high_packet_loss"} <= set(res["unavailable"])


def test_ge_operator_for_integer_columns():
    assert evaluate_record({"security_incidents": 18})["conditions"] == ["security_risk"]
    assert evaluate_record({"security_incidents": 17})["conditions"] == []


def test_condition_lists_follow_flags():
    flags = condition_frame(pd.DataFrame([{"latency_ms": 150.0, "jitter_ms": 49.0}, {"latency_ms": 10.0}]))
    lists = condition_lists(flags)
    assert lists[0] == ["high_latency", "high_jitter"] and lists[1] == []


@processed
def test_thresholds_reproduce_from_data():
    from src.preprocessing.build_thresholds import main
    assert main(check=True) == 0


@processed
def test_condition_shares_are_selective():
    from src.preprocessing.loader import load_processed
    shares = condition_frame(load_processed()).mean()
    assert shares.between(0.05, 0.2).all(), shares.to_dict()
