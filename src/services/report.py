"""Phase 29 response format (plain text report)."""

from __future__ import annotations

from typing import Any


def _yn(v: bool | None) -> str:
    return "YES" if v else "NO"


def render_report(r: dict[str, Any]) -> str:
    c, pr, risk, rca, recs, val = (r["classification"], r["priority"], r["risk"], r["root_cause"],
                                   r["recommendations"], r["validation"])
    pred, hist = risk["predicted"], risk["historical"]
    obs = r["evidence"]["observed"]
    L: list[str] = []
    L += ["INCIDENT CLASSIFICATION", "",
          f"Category: {c['category']}" + (f" (secondary: {', '.join(c['secondary_categories'])})"
                                          if c["secondary_categories"] else ""),
          f"Severity: {c['severity']} (score {c['severity_score']}, {c['severity_basis']})",
          f"Priority: {pr['priority']} - {pr['priority_label']} (score {pr['priority_score']})", ""]
    L += ["FAULT RISK", ""]
    if pred.get("available"):
        L += [f"Predicted risk: {pred['predicted_fault_rate_pct']}% fault occurrence "
              f"(80% interval {pred['interval_80pct'][0]}-{pred['interval_80pct'][1]}%; ML prediction, "
              f"input coverage {int(pred['input_coverage'] * 100)}%)",
              f"Risk level: {pred['risk_level']}"]
        if pred.get("caveat"):
            L.append(f"Note: {pred['caveat']}")
    else:
        L.append(f"Predicted risk: not available ({pred.get('reason')})")
    if hist:
        L.append(f"Historical fault occurrence of similar records (observed): mean {hist['mean']}%, "
                 f"range {hist['min']}-{hist['max']}% over {hist['n_records']} records")
    L += ["", "SIMILAR HISTORICAL RECORDS", "",
          "Record ID  | Similarity | Key Conditions | Category"]
    for x in r["retrieval"]["results"]:
        L.append(f"{x['record_id']} | {x['hybrid_score']:.2f} | {', '.join(x['conditions'][:4]) or 'none'} | {x['category']}")
    L += ["", "EVIDENCE", ""]
    for e in r["provenance"]:
        if e["source_type"] in ("USER_INPUT", "RAW_DATA", "DERIVED_FEATURE", "HISTORICAL_EVIDENCE", "ML_PREDICTION"):
            L.append(f"• [{e['source_type']}] {e['claim']}")
    L += ["", "LIKELY ROOT CAUSES", ""]
    if not rca.get("hypotheses"):
        L.append(rca["statement"])
    for i, h in enumerate(rca.get("hypotheses", []), 1):
        sup = ", ".join(e.get("condition") or f"{e.get('context')}={e.get('value')}" for e in h["supporting_evidence"])
        L += [f"{i}. {h['title']} (hypothesis)", f"   Confidence: {h['confidence']} (heuristic)",
              f"   Evidence: {sup or 'no direct evidence in the query'}"]
        if h.get("historical_support"):
            L.append(f"   Historical: {h['historical_support']}")
        if h["contradicting_evidence"]:
            L.append(f"   Against: {'; '.join(h['contradicting_evidence'])}")
        if h.get("note"):
            L.append(f"   Note: {h['note']}")
    L += ["", "WHAT TO CHECK", ""]
    L += [f"{i}. {x['check']}" for i, x in enumerate(recs["checks"][:14], 1)]
    L += ["", "RECOMMENDED ACTIONS", ""]
    acts = [a["action"] for a in recs["actions"]] + [m["action"] for m in recs["monitoring"]]
    L += [f"{i}. {a}" for i, a in enumerate(acts, 1)]
    L += ["", "ADDITIONAL EVIDENCE REQUIRED", ""]
    L += [f"• {x}" for x in rca.get("additional_evidence_required", [])]
    L += ["", "VALIDATION", "",
          f"Diagnosis supported by retrieved evidence: {_yn(val['diagnosis_supported'])}",
          f"Recommendation grounded in evidence: {_yn(val['recommendation_grounded'] and val['evidence_grounded'])}",
          f"Hallucination detected: {_yn(val['hallucination_detected'])} ({val['method']})"]
    L += ["", "IMPORTANT LIMITATION", ""]
    L += [f"• {x}" for x in r["limitations"]]
    if not obs["present"]:
        L.insert(0, "NOTE: no degraded network condition was identified in the query.\n")
    return "\n".join(L)
