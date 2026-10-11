"""docs/API_Documentation.pdf — every exposed endpoint with method, request/response format and real samples.

    python -m docs.deliverables.collect_samples      # once, with the API running
    python -m docs.deliverables.api_doc
"""

from __future__ import annotations

import json

from docs.deliverables.kit import P, abridge, build, cover, dumps, mono, samples, section, table
from docs.guide.pdf_kit import bullets, callout, code
from src.api.schemas import AnalyzeRequest, FeedbackRequest, Metrics, RecommendRequest, RecordOrMetrics, SearchRequest
from src.utils.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "docs" / "API_Documentation.pdf"
BASE = "http://localhost:8000"


def _type(prop: dict) -> str:
    if "anyOf" in prop:
        return " | ".join(_type(p) for p in prop["anyOf"] if p.get("type") != "null") or "null"
    if "enum" in prop:
        return " | ".join(f'"{e}"' for e in prop["enum"])
    if prop.get("type") == "array":
        return f"array of {_type(prop.get('items', {}))}"
    if "$ref" in prop:
        return prop["$ref"].split("/")[-1]
    return prop.get("type", "object")


def fields_table(model) -> list:
    schema = model.model_json_schema()
    required = set(schema.get("required", []))
    rows = [["Field", "Type", "Required", "Constraints / default", "Description"]]
    for name, prop in schema["properties"].items():
        inner = next((p for p in prop.get("anyOf", []) if p.get("type") != "null"), prop)
        cons = [f"{k} {inner[k]}" for k in ("minimum", "maximum", "minLength", "maxLength", "pattern") if k in inner]
        cons = [c.replace("minimum", "≥").replace("maximum", "≤") for c in cons]
        if "default" in prop and prop["default"] is not None:
            cons.append(f"default {json.dumps(prop['default'])}")
        rows.append([mono(name), _type(prop), "yes" if name in required else "no", ", ".join(cons) or "–",
                     prop.get("description", "")])
    return [table(rows, widths=[1.5, 1.3, 0.8, 1.5, 1.55])]


def resp_table(rows: list[list[str]]) -> list:
    return [table([["Field", "Description"]] + [[mono(a), b] for a, b in rows], widths=[1.6, 4.4])]


def endpoint(title: str, method: str, path: str, purpose: str, request: list, response: list, sample: dict,
             curl: str, max_list: int = 3) -> list:
    s = [*section(title),
         table([["Endpoint", mono(path)], ["HTTP method", method], ["Purpose", purpose],
                ["Success status", str(sample["status"])]], widths=[1.3, 4.7], header=False, bold_first_col=True)]
    s += [P("Request format", "h2")] + request
    s += [P("Response format", "h2")] + response
    s += [P("Sample request", "h2"), code(curl)]
    if sample.get("request") is not None:
        s.append(code(dumps(sample["request"])))
    s += [P("Sample response (real output; long lists shortened with '… n more')", "h2"),
          code(dumps(abridge(sample["response"], max_list=max_list)), max_lines=75)]
    return s


def story() -> list:
    S = samples()
    s = cover("API Documentation", "REST API reference (submission deliverable 5)", [
        ["Framework", "FastAPI on Uvicorn; request/response validation with Pydantic"],
        ["Base URL (local)", BASE],
        ["Start the API", mono("uvicorn src.main:app --reload --port 8000")],
        ["Interactive docs", f"{BASE}/docs (Swagger UI) and {BASE}/openapi.json"],
        ["Authentication", "none (local evaluation); no secrets are accepted or returned"],
        ["Samples", "every sample response in this document was returned by the running API "
                    "(docs/deliverables/collect_samples.py)"],
    ])

    s += section("1. Conventions")
    s += bullets([
        "Requests and responses are JSON (" + mono("Content-Type: application/json") + ").",
        "Every response carries an " + mono("X-Request-ID") + " header (sent by the client or generated) for tracing.",
        "Errors always have the same shape: " + mono('{"error": {"code", "message", "details", "request_id"}}') + ".",
        "Status codes: 200 OK, 201 created (feedback), 404 unknown record, 422 invalid request or rejected query, "
        "500 server or agent failure.",
        "Derived values (conditions, category, severity, priority) are labelled derived; ML predictions are labelled "
        "predictions; root causes are hypotheses.",
    ])
    s.append(table([["Method", "Endpoint", "Purpose"],
                    ["GET", mono("/"), "service information and endpoint list"],
                    ["GET", mono("/health"), "status of data, indexes, models and LLM provider"],
                    ["POST", mono("/api/v1/search"), "hybrid retrieval of similar historical records"],
                    ["POST", mono("/api/v1/analyze"), "full multi-agent analysis (main endpoint)"],
                    ["POST", mono("/api/v1/classify"), "conditions, category, severity, priority"],
                    ["POST", mono("/api/v1/risk"), "predicted fault-occurrence risk"],
                    ["POST", mono("/api/v1/recommend"), "hypotheses and recommendations without the LLM"],
                    ["POST", mono("/api/v1/feedback"), "store a rating of an analysis"],
                    ["GET", mono("/api/v1/records/{record_id}"), "one full record, raw and derived"],
                    ["GET", mono("/api/v1/reference"), "labels, units, ranges and thresholds for clients"],
                    ["GET", mono("/api/v1/guardrails/check"), "check a query against the input guardrails"]],
                   widths=[0.8, 2.4, 2.8]))

    s += section("2. Shared object: Metrics")
    s.append(P("Optional operator-supplied measurements, used by analyze, recommend, classify and risk. Unknown "
               "field names are rejected; ranges match the dataset schema."))
    s += fields_table(Metrics)

    s += endpoint("3. GET /", "GET", "/", "Service name, version and the list of endpoints.",
                  [P("No parameters.")], resp_table([("service", "service name"), ("version", "API version"),
                                                     ("docs / health", "paths of the docs and health endpoints"),
                                                     ("endpoints", "list of API paths")]),
                  S["root"], f"curl {BASE}/")
    s += endpoint("4. GET /health", "GET", "/health", "Readiness: data, BM25 index, ML models and the LLM provider.",
                  [P("No parameters.")], resp_table([("status", '"ok" or "degraded"'),
                                                     ("checks.incidents", "number of indexed records"),
                                                     ("checks.bm25_docs", "documents in the BM25 index"),
                                                     ("checks.models", "which ML models are loaded"),
                                                     ("checks.llm", "active LLM provider and model")]),
                  S["health"], f"curl {BASE}/health")
    s += endpoint("5. POST /api/v1/search", "POST", "/api/v1/search",
                  "Hybrid retrieval (semantic + BM25 + metric similarity) with strictly enforced filters.",
                  fields_table(SearchRequest) + [P("Filter keys: state (name or code), city, season, weather, "
                                                   "category, severity, priority, issue_reported, conditions "
                                                   "(list, all must be present). A value may be a list (any of).")],
                  resp_table([("query", "the sanitised query"),
                              ("results[]", "records: record_id, category, severity, priority, conditions, metrics "
                                            "(raw), metadata, search_text, semantic/keyword/metric/hybrid scores"),
                              ("filters", "normalised filters that were applied"),
                              ("candidates / matching_records", "re-scored candidates / records passing the filters"),
                              ("understanding", "parsed constraints and context"),
                              ("weights", "weights actually used for the hybrid score")]),
                  S["search"], f"curl -X POST {BASE}/api/v1/search -H \"Content-Type: application/json\" -d @body.json",
                  max_list=2)
    s += endpoint("6. POST /api/v1/analyze", "POST", "/api/v1/analyze",
                  "The main endpoint: guardrails, then six agents (retrieval, classification, risk, diagnosis, "
                  "recommendation with LLM explanation, validation).",
                  fields_table(AnalyzeRequest),
                  resp_table([("analysis_id", "unique ID of this analysis (used by feedback)"),
                              ("retrieval / evidence", "similar records; observed conditions; history statistics"),
                              ("classification", "derived category, severity and reasons"),
                              ("priority", "P1–P4 with score and reasons"),
                              ("risk.predicted / risk.historical", "ML prediction with 80% interval / observed rate in "
                                                                   "similar records"),
                              ("root_cause", "ranked hypotheses with confidence, evidence, checks, data needed"),
                              ("recommendations", "actions (with basis), checks, monitoring, disclaimer"),
                              ("narrative", "explanation; source llm or template; LLM attempts"),
                              ("validation", "grounded, supported, relevant, hallucination; method; issues"),
                              ("provenance", "every statement with one of 7 source types"),
                              ("report_text", "the full text report"),
                              ("agent_trace / timing_s", "per-agent status and duration / total time")]),
                  S["analyze"], f"curl -X POST {BASE}/api/v1/analyze -H \"Content-Type: application/json\" -d @body.json",
                  max_list=2)
    s += [*section("6a. Analyze with measured values"),
          P("When metrics are supplied, conditions are evaluated from the values (source 'measured') and steer retrieval:"),
          code(dumps(S["analyze_metrics"]["request"])),
          code(dumps(abridge({"evidence": {"observed": {k: S["analyze_metrics"]["response"]["evidence"]["observed"][k]
                                                        for k in ("values", "present", "absent", "measured_evidence")}},
                              "classification": {k: S["analyze_metrics"]["response"]["classification"][k]
                                                 for k in ("category", "severity", "severity_basis")}}, max_list=3)))]
    s += endpoint("7. POST /api/v1/classify", "POST", "/api/v1/classify",
                  "Conditions, category, severity and priority for a stored record or for supplied metrics.",
                  fields_table(RecordOrMetrics) + [P("Provide exactly one of record_id or metrics.")],
                  resp_table([("source", "dataset record (RAW_DATA) or operator-supplied metrics (USER_INPUT)"),
                              ("conditions", "triggered conditions with value, operator and threshold"),
                              ("classification", "category, severity, reasons, ML classifier cross-check"),
                              ("priority", "priority with reasons"),
                              ("stored_labels", "labels stored with the record (record_id requests only)")]),
                  S["classify"], f"curl -X POST {BASE}/api/v1/classify -H \"Content-Type: application/json\" "
                                 "-d '{\"record_id\": \"ROW-000123\"}'")
    s += endpoint("8. POST /api/v1/risk", "POST", "/api/v1/risk",
                  "Predicted fault occurrence risk (ML) from a record or from metrics.",
                  fields_table(RecordOrMetrics),
                  resp_table([("predicted", "predicted rate, risk level, 80% interval, inputs used, input coverage, "
                                            "model metadata, label"),
                              ("historical_fault_rate_pct", "the recorded (observed) rate — record_id requests only")]),
                  S["risk"], f"curl -X POST {BASE}/api/v1/risk -H \"Content-Type: application/json\" -d @body.json")
    s += endpoint("9. POST /api/v1/recommend", "POST", "/api/v1/recommend",
                  "Hypotheses and recommendations without classification or LLM explanation (fast path).",
                  fields_table(RecommendRequest),
                  resp_table([("observed", "conditions present/absent and their sources"),
                              ("root_cause", "ranked hypotheses"), ("recommendations", "actions, checks, monitoring"),
                              ("similar_record_ids", "IDs of the retrieved records"), ("risk", "predicted and historical risk")]),
                  S["recommend"], f"curl -X POST {BASE}/api/v1/recommend -H \"Content-Type: application/json\" -d @body.json",
                  max_list=2)
    s += endpoint("10. POST /api/v1/feedback", "POST", "/api/v1/feedback",
                  "Store an engineer's rating with the analysis context (SQLite). Unknown record IDs are rejected.",
                  fields_table(FeedbackRequest),
                  resp_table([("feedback_id", "unique ID of the stored rating"), ("created_at", "UTC timestamp"),
                              ("stored", "true when saved"), ("note", "states that nothing is retrained automatically")]),
                  S["feedback"], f"curl -X POST {BASE}/api/v1/feedback -H \"Content-Type: application/json\" -d @body.json")
    s += endpoint("11. GET /api/v1/records/{record_id}", "GET", "/api/v1/records/{record_id}",
                  "One full record: raw metrics and metadata, derived labels and the search text.",
                  [P("Path parameter " + mono("record_id") + ": format ROW-000001 … ROW-100000.")],
                  resp_table([("record_id / category / severity / priority / conditions", "identity and derived labels"),
                              ("metrics", "raw measured values"), ("metadata", "season, weather, location, flags"),
                              ("search_text", "the record's chunk text"), ("note", "which fields are derived")]),
                  S["record"], f"curl {BASE}/api/v1/records/ROW-000001")
    s += endpoint("12. GET /api/v1/reference", "GET", "/api/v1/reference",
                  "Read-only vocabulary for clients, taken from src/config and the request schema.",
                  [P("No parameters.")],
                  resp_table([("fields[]", "name, dataset label, unit, role, type, input, min, max"),
                              ("conditions[]", "name, label, metric, operator, threshold, method, group"),
                              ("categories / states / seasons / weather / severities / priorities", "filter values")]),
                  S["reference"], f"curl {BASE}/api/v1/reference", max_list=2)
    s += endpoint("13. GET /api/v1/guardrails/check", "GET", "/api/v1/guardrails/check?query=…",
                  "Check a query against the input guardrails without running an analysis.",
                  [P("Query parameter " + mono("query") + " (text).")],
                  resp_table([("allowed", "true if the query would be processed"),
                              ("category", "ok, empty, too_long, prompt_injection, secret_request, unsupported or off_topic"),
                              ("message", "explanation")]),
                  S["guardrails"], f"curl \"{BASE}/api/v1/guardrails/check?query=Ignore%20your%20instructions\"")

    s += section("14. Error responses (real)")
    for name, title in (("guardrail", "Rejected query (prompt injection) — 422"), ("bad_filter", "Unsupported filter — 422"),
                        ("out_of_range", "Value out of range — 422"), ("not_found", "Unknown record — 404")):
        e = S["errors"][name]
        s += [P(title, "h2"), code(f"{e['method']} {e['path']}" + (f"\n{dumps(e['request'])}" if e["request"] else "")),
              code(dumps(abridge(e["response"])))]
    s.append(callout("The API never returns secrets; API keys are read only from the local .env file.", "info"))
    return s


def main():
    return build(OUT, "API Documentation", story())


if __name__ == "__main__":
    main()
