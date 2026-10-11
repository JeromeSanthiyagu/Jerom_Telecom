"""LLM interpretation of the assembled evidence (Phase 15) with grounding control.

The LLM receives ONLY the evidence package (query, observed conditions,
retrieved records, historical statistics, classification, risk, hypotheses,
recommendations) and writes an explanation. It cannot add hypotheses,
records or numbers: its output is checked by `grounding.check_text`, retried
once with the violations listed, and replaced by a deterministic template if
it still fails. With no LLM configured the template is used directly.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from src.rag.grounding import check_text
from src.rag.llm_provider import get_provider

logger = logging.getLogger(__name__)

NA = "Not available in the current dataset."


class HypothesisNote(BaseModel):
    hypothesis_id: str = Field(description="Must be one of the provided hypothesis ids")
    rationale: str


class Narrative(BaseModel):
    summary: str = Field(description="2-3 sentence overview for a network engineer")
    diagnosis: str = Field(description="Why the evidence points to the ranked hypotheses")
    hypothesis_notes: list[HypothesisNote]
    limitations: list[str]


SYSTEM = """You are a telecom network-operations analyst writing for a network engineer.
You explain an analysis that deterministic tools have already produced. Rules:
1. Use ONLY the facts in the EVIDENCE JSON. Never invent measurements, record IDs, averages,
   user counts, fault rates, MTTR values, baselines or historical incidents.
2. Copy numbers exactly as they appear in the evidence. Do NOT calculate anything new - no averages,
   sums, differences, ratios or unit conversions; use the precomputed statistics instead. Prefer
   describing conditions in words over quoting many numbers. If something is not in the evidence,
   write "Not available in the current dataset."
3. Refer only to record IDs listed in the evidence, always written in full (e.g. ROW-000123, never 000123).
4. Root causes are HYPOTHESES. Discuss only the hypothesis ids provided, in the given order;
   never present one as confirmed.
5. Do not name protocol procedures, interfaces or parameter values (for example specific handover
   events/offsets, random-access procedures, core-network nodes, inter-node interfaces) unless the
   evidence contains them.
6. Never promise an outcome (no "will improve to", no time-to-fix). Say what to monitor instead.
7. Distinguish clearly: values stated by the operator, values from dataset records, derived
   conditions, ML predictions, and hypotheses.
Keep it concise: summary <= 3 sentences, one short rationale per hypothesis."""


def evidence_package(result: dict[str, Any]) -> dict:
    """The exact evidence the LLM may use (also used for grounding checks)."""
    rca = result["root_cause"]
    return {
        "query": result["query"],
        "operator_stated": {"values": result["evidence"]["observed"]["values"],
                            "context": result["evidence"]["observed"]["context"],
                            "conditions_present": result["evidence"]["observed"]["present"],
                            "conditions_reported_normal": result["evidence"]["observed"]["absent"]},
        "retrieved_records": result["evidence"]["records"],
        "historical_statistics_of_retrieved_records": {
            k: v for k, v in result["evidence"]["history"].items() if k != "record_ids"},
        "dataset_baseline": result["evidence"]["baseline"],
        "classification": {k: result["classification"][k] for k in
                           ("category", "secondary_categories", "severity", "severity_score", "severity_basis")},
        "priority": {k: result["priority"][k] for k in ("priority", "priority_label", "priority_score")},
        "fault_risk": result["risk"],
        "hypotheses": [{k: h[k] for k in ("id", "title", "confidence", "supporting_evidence",
                                          "contradicting_evidence", "historical_support", "note")}
                       for h in rca.get("hypotheses", [])],
        "recommended_actions": [a["action"] for a in result["recommendations"]["actions"]],
    }


def template_narrative(result: dict[str, Any]) -> Narrative:
    c, rca, risk = result["classification"], result["root_cause"], result["risk"]
    obs = result["evidence"]["observed"]
    hist = result["evidence"]["history"]
    conds = ", ".join(obs["present"]) or "none"
    pred = risk["predicted"]
    risk_txt = (f"predicted fault occurrence risk {pred['risk_level']} ({pred['predicted_fault_rate_pct']}%)"
                if pred.get("available") else "fault-risk prediction not available")
    summary = (f"For the case described in the query - observed conditions: {conds}; derived category "
               f"{c['category']}, case severity {c['severity']} ({c['severity_basis']}), priority "
               f"{result['priority']['priority']}; {risk_txt}. {hist.get('n_records', 0)} similar historical "
               f"records were retrieved (their own stored severities are listed separately).")
    if rca.get("primary_hypothesis"):
        diagnosis = (f"Leading hypothesis: {rca['primary_hypothesis']} (heuristic confidence "
                     f"{rca['confidence']}). Alternatives: "
                     + (", ".join(a["title"] for a in rca["alternative_hypotheses"]) or "none") + ".")
    else:
        diagnosis = rca["statement"]
    notes = []
    for h in rca.get("hypotheses", []):
        sup = ", ".join(e.get("condition") or f"{e.get('context')}={e.get('value')}" for e in h["supporting_evidence"])
        notes.append(HypothesisNote(hypothesis_id=h["id"], rationale=(
            f"Supported by: {sup or 'no direct evidence (triggered by related conditions)'}"
            + (f"; contradicted by: {'; '.join(h['contradicting_evidence'])}" if h["contradicting_evidence"] else "")
            + (f"; {h['historical_support']}" if h.get("historical_support") else ""))))
    return Narrative(summary=summary, diagnosis=diagnosis, hypothesis_notes=notes,
                     limitations=list(result.get("limitations", [])))


def _narrative_text(n: Narrative) -> str:
    return "\n".join([n.summary, n.diagnosis] + [h.rationale for h in n.hypothesis_notes] + n.limitations)


def write_narrative(result: dict[str, Any], use_llm: bool = True) -> dict:
    evidence = evidence_package(result)
    retrieved = set(result["evidence"]["history"].get("record_ids", []))
    hyp_ids = [h["id"] for h in result["root_cause"].get("hypotheses", [])]
    provider = get_provider() if use_llm else None
    attempts: list[dict] = []

    if provider is not None and provider.name != "offline":
        user = "EVIDENCE:\n" + json.dumps(evidence, indent=1, default=str)
        for attempt in range(2):
            out, info = provider.structured(SYSTEM, user, Narrative, max_tokens=1500)
            entry = {"attempt": attempt + 1, **info}
            if out is None:
                attempts.append(entry)
                break
            issues = check_text(_narrative_text(out), evidence, retrieved)
            bad_ids = [h.hypothesis_id for h in out.hypothesis_notes if h.hypothesis_id not in hyp_ids]
            issues += [f"unknown hypothesis id: {b}" for b in bad_ids]
            entry["grounding_issues"] = issues
            if issues:
                entry["rejected_output"] = out.model_dump()
            attempts.append(entry)
            if not issues:
                return {"source": "llm", "narrative": out.model_dump(), "attempts": attempts,
                        "grounding_issues": []}
            user += ("\n\nYour previous answer was rejected for these grounding problems: "
                     + "; ".join(issues) + ". Rewrite it using only evidence values.")
    tmpl = template_narrative(result)
    return {"source": "template", "narrative": tmpl.model_dump(), "attempts": attempts,
            "grounding_issues": check_text(_narrative_text(tmpl), evidence, retrieved),
            "llm_rejected": bool(attempts and any(a.get("grounding_issues") for a in attempts))}
