"""Corrective recommendations (Phase 18).

Recommendations come only from the playbook entries of the hypotheses that
the evidence supports, and every item names the hypothesis and evidence it
is based on. No KPI improvement is ever promised: monitoring items compare
against the dataset baseline instead.
"""

from __future__ import annotations

from typing import Any

from src.preprocessing.conditions import load_conditions

MONITOR_METRICS = {
    "low_handover": "handover_success_pct", "high_latency": "latency_ms", "high_packet_loss": "packet_loss_pct",
    "high_call_drop": "call_drop_pct", "weak_signal": "signal_strength_dbm", "low_throughput": "throughput_mbps",
    "high_jitter": "jitter_ms", "low_availability": "cell_availability_pct", "low_setup_success": "setup_success_pct",
    "high_fault_rate": "fault_rate_pct", "authentication_failure": "auth_failures",
    "high_critical_alarm_count": "critical_alarm_count",
}


def build_recommendations(rca: dict[str, Any], state: dict[str, Any], baseline: dict[str, Any],
                          min_confidence: float = 0.25, max_actions: int = 8) -> dict:
    hyps = [h for h in rca.get("hypotheses", []) if h["confidence"] >= min_confidence] or rca.get("hypotheses", [])[:1]
    actions: list[dict] = []
    checks: list[dict] = []
    seen_a, seen_c = set(), set()
    for h in hyps:
        evidence = [e.get("condition") or f"{e.get('context')}={e.get('value')}" for e in h["supporting_evidence"]]
        for a in h["actions"]:
            if a.startswith("Monitor"):
                continue        # monitoring is generated below with baselines
            if a not in seen_a:
                seen_a.add(a)
                actions.append({"action": a, "hypothesis": h["id"], "based_on": evidence,
                                "hypothesis_confidence": h["confidence"]})

    # "what to check" covers every reported hypothesis - all checks of the primary one and the
    # top 2 of each alternative: checks are cheap investigations that confirm or rule out causes.
    for i, h in enumerate(rca.get("hypotheses", [])):
        for c in (h["checks"] if i == 0 else h["checks"][:2]):
            if c not in seen_c:
                seen_c.add(c)
                checks.append({"check": c, "hypothesis": h["id"], "hypothesis_confidence": h["confidence"]})

    labels = {n: c.label for n, c in load_conditions().items()}
    monitoring = []
    for cond in state.get("present", []):
        m = MONITOR_METRICS.get(cond)
        if m and m in baseline.get("median", {}):
            monitoring.append({
                "action": f"Monitor {labels[cond].lower()} ({m}) after remediation and compare it with the "
                          f"dataset baseline (median {baseline['median'][m]:g} over {baseline['rows']} records)",
                "metric": m, "baseline_median": baseline["median"][m], "based_on": [cond],
            })
    return {
        "actions": actions[:max_actions],
        "checks": checks,
        "monitoring": monitoring,
        "disclaimer": "Actions are options to evaluate, not guaranteed fixes; no KPI improvement is predicted.",
    }
