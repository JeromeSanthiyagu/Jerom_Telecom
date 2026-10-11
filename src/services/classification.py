"""Case classification: category, severity and priority for the analysed case.

* category  - rule-based taxonomy on the case's present conditions (authoritative)
* severity  - points score; exact for supplied values, estimated for conditions
              that were only stated in words
* priority  - severity + impact + predicted fault risk (Phase 6 rules)
* ML        - the category classifier's estimate, shown as a consistency check
* history   - category distribution of the retrieved similar records
"""

from __future__ import annotations

from typing import Any

from src.services.models.incident_classifier import predict_category
from src.preprocessing.priority import score_priority
from src.preprocessing.severity import estimate_from_conditions
from src.preprocessing.taxonomy import classify_conditions


def classify_case(state: dict[str, Any], history: dict[str, Any]) -> dict:
    present = state["present"]
    values = state["values"]
    cat = classify_conditions(present)
    stated_only = [c for c in present if state["condition_sources"].get(c) == "stated"]
    severity = estimate_from_conditions(stated_only, values)
    ml = predict_category(values) if values else {"available": False,
                                                    "reason": "No metric values supplied; the classifier needs values."}
    hist = history.get("category_distribution", {})
    return {
        "category": cat["primary_category"],
        "secondary_categories": cat["secondary_categories"],
        "category_reasons": cat["reasons"],
        "category_method": "rule-based taxonomy (src/config/taxonomy.yaml)",
        "conditions": present,
        "severity": severity["severity"],
        "severity_score": severity["severity_score"],
        "severity_reasons": severity["reasons"],
        "severity_basis": severity["basis"],
        "ml_category": ml,
        "ml_agrees_with_rules": (ml.get("predicted_category") == cat["primary_category"]) if ml.get("available") else None,
        "historical_category_distribution": hist,
    }


def prioritize(classification: dict, state: dict[str, Any], risk_level: str | None) -> dict:
    sev = {"severity": classification["severity"], "severity_score": classification["severity_score"]}
    return score_priority(state["values"], sev, classification["conditions"], fault_risk_level=risk_level)
