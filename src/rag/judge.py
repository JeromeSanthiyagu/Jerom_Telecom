"""Validation layer (Phase 24): deterministic checks + optional LLM-as-judge.

Deterministic checks are authoritative for hallucination: they verify every
number and record ID in the narrative against the evidence. The LLM judge
adds a qualitative review (diagnosis support, relevance, actionability,
completeness). A boolean is only reported True if every available check agrees.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from src.rag.narrative import evidence_package
from src.rag.llm_provider import get_provider


class JudgeVerdict(BaseModel):
    diagnosis_supported: bool
    evidence_grounded: bool
    recommendation_relevant: bool
    recommendation_actionable: bool
    hallucination_detected: bool
    completeness: float
    score: float
    issues: list[str]


JUDGE_SYSTEM = """You are a strict reviewer of telecom network-operations analyses.
Given the EVIDENCE and the ANALYSIS, judge:
- diagnosis_supported: are the ranked hypotheses supported by the evidence conditions?
- evidence_grounded: does every statement in the analysis follow from the evidence?
- recommendation_relevant: do the actions address the observed conditions/hypotheses?
- recommendation_actionable: could an engineer carry the actions out?
- hallucination_detected: does the analysis contain any fact, number, record or claim absent from the evidence?
- completeness (0..1) and an overall score (0..1).
List concrete issues (empty list if none). Judge only against the evidence provided."""


def deterministic_checks(result: dict[str, Any]) -> dict:
    rca, recs = result["root_cause"], result["recommendations"]
    narrative = result["narrative"]
    present = set(result["evidence"]["observed"]["present"])
    records = result["retrieval"]["results"]
    hyps = rca.get("hypotheses", [])
    top = hyps[0] if hyps else None
    hyp_ids = {h["id"] for h in hyps}

    diagnosis_supported = bool(top and any(e.get("condition") in present or e.get("context")
                                           for e in top["supporting_evidence"]) and records)
    recs_grounded = bool(recs["actions"]) and all(a["hypothesis"] in hyp_ids for a in recs["actions"])
    recs_relevant = recs_grounded and all(
        any(c in present for c in a["based_on"]) or a["hypothesis"] == (top or {}).get("id")
        for a in recs["actions"])
    from src.rag.hybrid import get_retriever
    ids_valid = all(r["record_id"] in get_retriever().row_of for r in records)
    sections = {
        "classification": bool(result["classification"].get("category")),
        "risk": bool(result["risk"]["predicted"].get("available")),
        "similar_records": bool(records),
        "hypotheses": bool(hyps),
        "recommendations": bool(recs["actions"]),
        "additional_evidence": bool(rca.get("additional_evidence_required")),
    }
    issues = list(narrative.get("grounding_issues", []))
    if not records:
        issues.append("no similar records retrieved")
    if not present:
        issues.append("no degraded condition identified in the query")
    return {
        "diagnosis_supported": diagnosis_supported,
        "evidence_grounded": not narrative.get("grounding_issues"),
        "recommendation_grounded": recs_grounded,
        "recommendation_relevant": recs_relevant,
        "record_ids_valid": ids_valid,
        "hallucination_detected": bool(narrative.get("grounding_issues")),
        "llm_output_rejected": bool(narrative.get("llm_rejected")),
        "completeness": round(sum(sections.values()) / len(sections), 2),
        "sections": sections,
        "issues": issues,
    }


def llm_judge(result: dict[str, Any]) -> tuple[JudgeVerdict | None, dict]:
    provider = get_provider()
    if provider.name == "offline":
        return None, {"status": "skipped", "reason": "No LLM configured."}
    analysis = {
        "narrative": result["narrative"]["narrative"],
        "primary_hypothesis": result["root_cause"].get("primary_hypothesis"),
        "alternatives": result["root_cause"].get("alternative_hypotheses"),
        "recommended_actions": [a["action"] for a in result["recommendations"]["actions"]],
        "monitoring": [m["action"] for m in result["recommendations"]["monitoring"]],
    }
    user = ("EVIDENCE:\n" + json.dumps(evidence_package(result), indent=1, default=str)
            + "\n\nANALYSIS:\n" + json.dumps(analysis, indent=1, default=str))
    verdict, info = provider.structured(JUDGE_SYSTEM, user, JudgeVerdict, max_tokens=1500)
    if verdict is not None:
        verdict.score = min(1.0, max(0.0, verdict.score))
        verdict.completeness = min(1.0, max(0.0, verdict.completeness))
    return verdict, info


def validate(result: dict[str, Any], use_llm: bool = True) -> dict:
    det = deterministic_checks(result)
    verdict, info = llm_judge(result) if use_llm else (None, {"status": "skipped", "reason": "LLM disabled"})
    out = {
        "deterministic": det,
        "llm_judge": verdict.model_dump() if verdict else None,
        "llm_judge_info": info,
    }
    j = verdict
    out["diagnosis_supported"] = det["diagnosis_supported"] and (j.diagnosis_supported if j else True)
    out["evidence_grounded"] = det["evidence_grounded"] and (j.evidence_grounded if j else True)
    out["recommendation_relevant"] = det["recommendation_relevant"] and (j.recommendation_relevant if j else True)
    out["recommendation_grounded"] = det["recommendation_grounded"]
    out["hallucination_detected"] = det["hallucination_detected"] or (j.hallucination_detected if j else False)
    out["score"] = round((j.score if j else 1.0) * (0.5 + 0.5 * det["completeness"])
                         * (0.0 if out["hallucination_detected"] else 1.0), 2)
    out["issues"] = det["issues"] + (j.issues if j else [])
    out["method"] = "deterministic + LLM judge" if j else "deterministic only"
    return out
