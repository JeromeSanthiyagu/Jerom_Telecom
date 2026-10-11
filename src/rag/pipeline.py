"""Grounded RAG analysis pipeline (Phase 15) - the single-pipeline version.

    query -> guardrails -> query understanding -> metadata filters
          -> hybrid retrieval -> evidence extraction -> classification
          -> fault risk -> priority -> root-cause hypotheses
          -> recommendations -> LLM narrative (grounded) -> validation
          -> provenance + Phase 29 report

The multi-agent orchestrator (src/agents/) runs exactly these steps as agents;
`analyze()` here is kept as the reference implementation and both are tested
to give the same result.
"""

from __future__ import annotations

import time
from typing import Any

from src.services import evidence as ev
from src.services.classification import classify_case, prioritize
from src.rag.judge import validate
from src.rag.narrative import write_narrative
from src.services.provenance import ProvenanceLog, SourceType
from src.services.recommendations import build_recommendations
from src.services.report import render_report
from src.services.risk import assess_risk
from src.services.root_cause import root_cause_analysis
from src.services.guardrails import check_query
from src.rag.hybrid import get_retriever
from src.rag.query_parser import parse_query
from src.preprocessing.conditions import load_conditions

LIMITATIONS = [
    "The dataset contains raw network-performance measurements only; conditions, categories, severity, "
    "priority and search texts are derived during preprocessing and are not ground-truth annotations.",
    "The dataset appears synthetic (uniform distributions, random locations) and has no timestamps, cell IDs, "
    "neighbour relations, transport counters or signalling traces, so root causes cannot be confirmed from it.",
    "Root causes are engineering hypotheses with heuristic confidence scores, not calibrated probabilities.",
    "Fault risk is an ML prediction (primary model test R² about 0.88) and is distinct from the historical "
    "fault occurrence rate of the retrieved records.",
]


class GuardrailRejection(Exception):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category
        self.message = message


# --------------------------------------------------------------------------- steps
def step_guardrails(query: str) -> str:
    g = check_query(query)
    if not g.allowed:
        raise GuardrailRejection(g.category, g.message)
    return g.sanitized


def step_retrieve(query: str, filters: dict | None, top_k: int, metrics: dict | None) -> dict:
    retriever = get_retriever()
    understanding = parse_query(query)
    if metrics:
        understanding.add_supplied_metrics(metrics)      # supplied values steer retrieval too
    search = retriever.hybrid_search(query, filters, top_k, understanding=understanding)
    observed = ev.observed_state(understanding, metrics)
    history = ev.historical_statistics(search["results"])
    return {
        "understanding": understanding.to_dict(),
        "retrieval": search,
        "evidence": {"observed": observed, "history": history, "baseline": ev.dataset_baseline(),
                     "records": ev.records_for_prompt(search["results"])},
    }


def step_classify(result: dict) -> dict:
    return classify_case(result["evidence"]["observed"], result["evidence"]["history"])


def step_risk(result: dict) -> tuple[dict, dict]:
    risk = assess_risk(result["evidence"]["observed"], result["evidence"]["history"])
    level = risk["predicted"].get("risk_level") if risk["predicted"].get("available") else None
    return risk, prioritize(result["classification"], result["evidence"]["observed"], level)


def step_diagnose(result: dict) -> dict:
    return root_cause_analysis(result["evidence"]["observed"], result["evidence"]["history"])


def step_recommend(result: dict) -> dict:
    return build_recommendations(result["root_cause"], result["evidence"]["observed"], result["evidence"]["baseline"])


def build_provenance(result: dict) -> list[dict]:
    log = ProvenanceLog()
    labels = {n: c.label for n, c in load_conditions().items()}
    obs = result["evidence"]["observed"]
    for m, v in obs["values"].items():
        shown = f"{v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)
        log.add(f"Operator-supplied value {m} = {shown}", SourceType.USER_INPUT, {m: v})
    for n in obs["present"]:
        if obs["condition_sources"].get(n) == "stated":
            log.add(f"{labels[n]} reported in the query", SourceType.USER_INPUT, {"condition": n})
    for e in obs["measured_evidence"]:
        log.add(f"{e['label']}: {e['value']:g} {e['operator']} threshold {e['threshold']:g}",
                SourceType.DERIVED_FEATURE, e)
    hist = result["evidence"]["history"]
    if hist.get("n_records"):
        top_conds = list(hist["condition_frequency"].items())[:4]
        for cond, f in top_conds:
            log.add(f"{labels.get(cond, cond)} present in {f['count']} of {hist['n_records']} similar records",
                    SourceType.HISTORICAL_EVIDENCE, {"condition": cond, **f, "record_ids": hist["record_ids"]})
        for x in result["retrieval"]["results"][:3]:
            m = x["metrics"]
            log.add(f"{x['record_id']}: latency {m['latency_ms']:.2f} ms, packet loss {m['packet_loss_pct']:.2f}%, "
                    f"handover success {m['handover_success_pct']:.2f}%, fault rate {m['fault_rate_pct']:.2f}%",
                    SourceType.RAW_DATA, {"record_id": x["record_id"], **{k: m[k] for k in
                                          ("latency_ms", "packet_loss_pct", "handover_success_pct", "fault_rate_pct")}})
    c = result["classification"]
    log.add(f"Derived category {c['category']}, severity {c['severity']} ({c['severity_basis']})",
            SourceType.DERIVED_FEATURE, {"reasons": c["category_reasons"] + c["severity_reasons"]})
    pred = result["risk"]["predicted"]
    if pred.get("available"):
        log.add(f"Predicted fault occurrence risk {pred['risk_level']} ({pred['predicted_fault_rate_pct']}%)",
                SourceType.ML_PREDICTION, {k: pred[k] for k in ("predicted_fault_rate_pct", "interval_80pct",
                                                                "input_coverage", "model")})
    if result["risk"]["historical"]:
        h = result["risk"]["historical"]
        log.add(f"Historical fault occurrence of similar records: mean {h['mean']}%", SourceType.HISTORICAL_EVIDENCE, h)
    for h in result["root_cause"].get("hypotheses", []):
        log.add(f"Possible {h['title'].lower()}", SourceType.ENGINEERING_HYPOTHESIS,
                {"confidence": h["confidence"], "supporting": h["supporting_evidence"],
                 "contradicting": h["contradicting_evidence"]})
    for a in result["recommendations"]["actions"]:
        log.add(a["action"], SourceType.RECOMMENDATION, {"hypothesis": a["hypothesis"], "based_on": a["based_on"]})
    return [c.model_dump(mode="json") for c in log.claims]


def finalize(result: dict, use_llm: bool) -> dict:
    result["narrative"] = write_narrative(result, use_llm=use_llm)
    result["validation"] = validate(result, use_llm=use_llm)
    result["provenance"] = build_provenance(result)
    result["report_text"] = render_report(result)
    return result


# --------------------------------------------------------------------------- pipeline
def analyze(query: str, filters: dict | None = None, top_k: int = 5, metrics: dict | None = None,
            use_llm: bool = True) -> dict[str, Any]:
    t0 = time.time()
    clean = step_guardrails(query)
    result: dict[str, Any] = {"query": clean, "limitations": list(LIMITATIONS)}
    result.update(step_retrieve(clean, filters, top_k, metrics))
    result["classification"] = step_classify(result)
    result["risk"], result["priority"] = step_risk(result)
    result["root_cause"] = step_diagnose(result)
    result["recommendations"] = step_recommend(result)
    finalize(result, use_llm)
    result["timing_s"] = round(time.time() - t0, 2)
    return result
