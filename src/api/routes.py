"""REST endpoints (Phase 27). Mounted by src/main.py; documented in docs/API_Documentation.pdf."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter

from src.agents.orchestrator import run as run_agents
from src.api.errors import NotFound
from src.api.schemas import AnalyzeRequest, FeedbackRequest, Metrics, RecommendRequest, RecordOrMetrics, SearchRequest
from src.preprocessing.conditions import evaluate_record, load_conditions
from src.preprocessing.taxonomy import all_categories
from src.rag.filters import PRIORITIES, SEASONS, SEVERITIES, WEATHER, FilterError
from src.rag.hybrid import get_retriever
from src.rag.llm_provider import get_provider
from src.rag.pipeline import step_guardrails, step_retrieve
from src.rag.query_parser import parse_query
from src.services import evidence as ev
from src.services.classification import classify_case, prioritize
from src.services.feedback import save_feedback
from src.services.guardrails import check_query
from src.services.models.common import load_model
from src.services.models.fault_risk import predict_risk
from src.services.recommendations import build_recommendations
from src.services.risk import assess_risk
from src.services.root_cause import root_cause_analysis
from src.utils.config import load_schema
from src.utils.geo import STATE_NAMES

VERSION = "1.0.0"
router = APIRouter()


# ----------------------------------------------------------------------------- helpers
def _record_values(record_id: str) -> tuple[dict, dict]:
    rec = get_retriever().record(record_id)
    if rec is None:
        raise NotFound(f"Record {record_id} does not exist.")
    values = dict(rec["metrics"])     # the risk model ignores fault_rate_pct/mttr_hours (not features)
    values.update({k: rec["metadata"][k] for k in ("season", "weather", "temperature_c", "humidity_pct",
                                                    "issue_reported")})
    return rec, values


def _inputs(body: RecordOrMetrics) -> tuple[dict | None, dict]:
    if body.record_id and body.metrics:
        raise FilterError("Provide either record_id or metrics, not both.")
    if body.record_id:
        return _record_values(body.record_id)
    if body.metrics and body.metrics.values():
        return None, body.metrics.values()
    raise FilterError("Provide a record_id or at least one metric value.")


# ----------------------------------------------------------------------------- routes
@router.get("/", tags=["meta"])
def root() -> dict:
    return {"service": "AI-Powered Telecom Network Operations Assistant", "version": VERSION, "docs": "/docs",
            "health": "/health",
            "endpoints": ["/api/v1/search", "/api/v1/analyze", "/api/v1/classify", "/api/v1/risk",
                          "/api/v1/recommend", "/api/v1/feedback", "/api/v1/records/{record_id}",
                          "/api/v1/reference"]}


@router.get("/health", tags=["meta"])
def health() -> dict:
    checks: dict[str, Any] = {}
    try:
        r = get_retriever()
        checks["incidents"] = len(r.df)
        checks["bm25_docs"] = r.bm25.weights.shape[0]
    except Exception as exc:  # report, don't crash the health endpoint
        checks["retrieval_error"] = str(exc)
    checks["models"] = {m: load_model(m) is not None for m in ("fault_risk", "throughput", "incident_classifier")}
    p = get_provider()
    checks["llm"] = {"provider": p.name, "model": p.model}
    ok = "retrieval_error" not in checks and all(checks["models"].values())
    return {"status": "ok" if ok else "degraded", "checks": checks}


@router.post("/api/v1/search", tags=["retrieval"])
def search(body: SearchRequest) -> dict:
    q = step_guardrails(body.query)
    res = get_retriever().hybrid_search(q, body.filters, body.top_k)
    return {"query": q, **res}


@router.post("/api/v1/analyze", tags=["analysis"])
def analyze(body: AnalyzeRequest) -> dict:
    metrics = body.metrics.values() if body.metrics else None
    result = run_agents(body.query, body.filters, body.top_k, metrics, use_llm=body.use_llm)
    result["analysis_id"] = uuid.uuid4().hex
    return result


@router.post("/api/v1/classify", tags=["analysis"])
def classify(body: RecordOrMetrics) -> dict:
    rec, values = _inputs(body)
    q = parse_query("")
    state = ev.observed_state(q, values)
    c = classify_case(state, {})
    risk = predict_risk(values)
    pr = prioritize(c, state, risk.get("risk_level") if risk.get("available") else None)
    out = {"source": "dataset record (RAW_DATA)" if rec else "operator-supplied metrics (USER_INPUT)",
           "conditions": evaluate_record(values), "classification": c, "priority": pr}
    if rec:
        out["record_id"] = rec["record_id"]
        out["stored_labels"] = {"category": rec["category"], "severity": rec["severity"], "priority": rec["priority"]}
    return out


@router.post("/api/v1/risk", tags=["analysis"])
def risk(body: RecordOrMetrics) -> dict:
    rec, values = _inputs(body)
    out = {"predicted": predict_risk(values)}
    if rec:
        out["record_id"] = rec["record_id"]
        out["historical_fault_rate_pct"] = {
            "value": rec["metrics"]["fault_rate_pct"],
            "label": "Historical fault occurrence rate recorded in the dataset (observed, not predicted)"}
    return out


@router.post("/api/v1/recommend", tags=["analysis"])
def recommend(body: RecommendRequest) -> dict:
    q = step_guardrails(body.query)
    base = step_retrieve(q, body.filters, body.top_k, body.metrics.values() if body.metrics else None)
    obs, hist = base["evidence"]["observed"], base["evidence"]["history"]
    rca = root_cause_analysis(obs, hist)
    recs = build_recommendations(rca, obs, base["evidence"]["baseline"])
    return {"query": q, "observed": obs, "root_cause": rca, "recommendations": recs,
            "similar_record_ids": hist.get("record_ids", []), "risk": assess_risk(obs, hist)}


@router.post("/api/v1/feedback", tags=["feedback"], status_code=201)
def feedback(body: FeedbackRequest) -> dict:
    retriever = get_retriever()
    unknown = [r for r in body.retrieved_record_ids if r not in retriever.row_of]
    if unknown:
        raise FilterError("Unknown record IDs in feedback.", {"unknown": unknown})
    return save_feedback(body.model_dump())


@router.get("/api/v1/records/{record_id}", tags=["retrieval"])
def record(record_id: str) -> dict:
    rec = get_retriever().record(record_id)
    if rec is None:
        raise NotFound(f"Record {record_id} does not exist.")
    rec.pop("_raw", None)
    rec["note"] = "Raw fields are from the dataset; conditions/category/severity/priority/search_text are derived."
    return rec


@router.get("/api/v1/reference", tags=["meta"])
def reference() -> dict:
    """Read-only vocabulary for clients: field labels/units/ranges, condition thresholds and filter values.
    Everything comes from src/config (schema.yaml, thresholds.yaml, taxonomy.yaml) and the request schema."""
    limits = {}
    for name, f in Metrics.model_fields.items():
        limits[name] = {"min": next((m.ge for m in f.metadata if hasattr(m, "ge")), None),
                        "max": next((m.le for m in f.metadata if hasattr(m, "le")), None)}
    fields = [{"name": c["name"], "label": c["raw"], "unit": c.get("unit"), "role": c.get("role"),
               "type": c["type"], "input": c["name"] in limits, **limits.get(c["name"], {})}
              for c in load_schema()["columns"]]
    conds = [{"name": c.name, "label": c.label, "metric": c.metric, "operator": c.operator,
              "threshold": c.threshold, "method": c.method, "group": c.group} for c in load_conditions().values()]
    return {"fields": fields, "conditions": conds, "categories": all_categories(), "states": STATE_NAMES,
            "seasons": SEASONS, "weather": WEATHER, "severities": SEVERITIES, "priorities": PRIORITIES}


@router.get("/api/v1/guardrails/check", tags=["meta"], include_in_schema=False)
def guardrail_check(query: str) -> dict:
    g = check_query(query)
    return {"allowed": g.allowed, "category": g.category, "message": g.message}
