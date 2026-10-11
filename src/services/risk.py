"""Fault-risk assessment: ML prediction vs. historical evidence, kept separate.

* predicted   - fault-risk model output from the inputs available for this case
* historical  - fault occurrence rate actually recorded in the retrieved
                similar records (an observed statistic, not a prediction)
"""

from __future__ import annotations

from typing import Any

from src.services.models.fault_risk import predict_risk, risk_level


def assess_risk(state: dict[str, Any], history: dict[str, Any]) -> dict:
    inputs = {**state["values"], **{k: v for k, v in state.get("context", {}).items() if k in ("season", "weather")}}
    predicted = predict_risk(inputs)
    hist = history.get("metrics", {}).get("fault_rate_pct")
    historical = None
    if hist:
        historical = {
            "label": "Historical fault occurrence rate of the retrieved similar records (observed, not predicted)",
            **hist,
            "risk_band_of_mean": risk_level(hist["mean"]),
            "n_records": history["n_records"],
        }
    if predicted.get("available") and not predicted["inputs_used"]:
        predicted["caveat"] = ("No model inputs were supplied, so this is the model's baseline expectation, not a "
                               "case-specific prediction. The historical rate of similar records is the stronger "
                               "evidence here.")
    elif predicted.get("available") and predicted["low_input_coverage"]:
        n = len(predicted["inputs_used"])
        predicted["caveat"] = (f"Only {n} of the model's inputs {'was' if n == 1 else 'were'} supplied; the "
                               "missing ones are treated as unknown, so this prediction is less certain than the "
                               "model's test error suggests. Compare it with the historical rate of similar records.")
    return {"predicted": predicted, "historical": historical}
