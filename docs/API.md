# API Reference (Phase 27)

```bash
uvicorn src.main:app --port 8000        # interactive OpenAPI docs: http://localhost:8000/docs
```

- Start-up loads the embedding model, Chroma and the models (about 40 s). Set `WARMUP=0` to skip this.
- Every response carries an `X-Request-ID` header.
- Errors are structured:

```json
{"error": {"code": "guardrail_prompt_injection", "message": "…", "details": null, "request_id": "724ec73f0f79"}}
```

| Code | HTTP | When |
|---|---|---|
| `guardrail_empty` / `_too_long` / `_prompt_injection` / `_secret_request` / `_unsupported` / `_off_topic` | 422 | query rejected by guardrails |
| `invalid_filter` | 422 | unknown filter key or value, unknown record IDs in feedback, missing inputs |
| `validation_error` | 422 | request body fails schema validation (e.g. `top_k: 0`, `packet_loss_pct: 150`) |
| `not_found` | 404 | unknown `record_id` |
| `agent_failure` / `internal_error` | 500 | unexpected failure (logged with request id) |

## Endpoints

### `GET /` and `GET /health`
`/health` reports the record count, BM25 size, whether each model loaded, and the LLM provider/model. `status` is `ok` or `degraded`.

### `POST /api/v1/search` — hybrid retrieval
```json
{"query": "high latency and packet loss", "filters": {"state": "Minnesota", "season": "Winter"}, "top_k": 5}
```
**Filters:**

| Filter | Value |
|---|---|
| `season`, `weather`, `state` (name or code), `city`, `category`, `severity`, `priority` | string or list (OR) |
| `issue_reported` | boolean |
| `conditions` | list (AND), e.g. `["high_latency", "low_handover"]` |

**Returns:** `results[]` with `record_id`, `semantic_score`, `keyword_score`, `metric_score`, `hybrid_score`, `category`, `severity`, `priority`, `conditions`, `metrics`, `metadata` and `search_text`. The response also includes `filters` (normalised), `matching_records`, `candidates`, `weights` and `understanding` (the parsed query).

### `POST /api/v1/analyze` — full multi-agent analysis
```json
{"query": "A network region has increasing fault occurrence and declining handover success.",
 "filters": {"weather": ["Rain", "Snow"]}, "top_k": 5, "use_llm": true,
 "metrics": {"handover_success_pct": 90.5, "user_count": 9500}}
```
`metrics` is optional; it carries operator-measured values with validated ranges.

**Returns:**

| Field | Contents |
|---|---|
| `analysis_id`, `query` | |
| `understanding`, `retrieval` | parsed query and search results |
| `evidence` | `observed`, `history`, `baseline`, `records` |
| `classification` | category, severity, ML cross-check |
| `risk` | `predicted` vs `historical` |
| `priority` | |
| `root_cause` | |
| `recommendations` | `actions`, `checks`, `monitoring` |
| `narrative` | source `llm` or `template`, attempts, grounding issues |
| `validation` | |
| `provenance[]` | |
| `report_text` | the Phase 29 text format |
| `limitations`, `agent_trace[]`, `timing_s` | |

Typical latency: under 1 s without the LLM, 10–40 s with the NVIDIA LLM (narrative + judge).

### `POST /api/v1/classify`
`{"record_id": "ROW-000123"}` **or** `{"metrics": {...}}`. Returns:
- the conditions with evidence;
- category with reasons, severity with reasons, ML category with probabilities;
- priority;
- for a record, its `stored_labels`.

### `POST /api/v1/risk`
`{"record_id": "ROW-000001"}` or `{"metrics": {...}}`. Returns `predicted` with these fields:
- `predicted_fault_rate_pct`, `risk_level`, `interval_80pct`;
- `inputs_used`, `input_coverage`, `label` ("ML prediction").

For a record it also returns `historical_fault_rate_pct`, labelled *observed, not predicted*.

### `POST /api/v1/recommend`
The same body as `/search` plus optional `metrics`. Returns `observed`, `root_cause`, `recommendations`, `similar_record_ids` and `risk`, without the LLM.

### `POST /api/v1/feedback` → 201
```json
{"analysis_id": "…", "query": "…", "retrieved_record_ids": ["ROW-000001"],
 "diagnosis": {"primary_hypothesis": "…", "confidence": 0.65}, "recommendations": ["…"],
 "risk_prediction": {"risk_level": "Medium"}, "rating": "useful", "comment": "optional"}
```
`rating` is `useful` or `not_useful`. Unknown record IDs are rejected. Feedback is stored with a timestamp in `data/feedback/feedback.db` and is **not** used for automatic retraining. Export it for review with `python -m src.services.feedback export`.

### `GET /api/v1/records/{record_id}`
Returns the full record: raw metrics, metadata, derived labels and `search_text`, with a note on which fields are derived. Unknown IDs return 404.

### `GET /api/v1/reference`
Read-only vocabulary for clients such as the web UI, taken from `src/config` and the request schema. It returns:
- `fields`: name, dataset label, unit, role, type, whether it is an accepted input, and min/max;
- `conditions`: name, label, metric, operator, threshold, method (`percentile` or `domain`) and group;
- `categories`, `states` (code → name), `seasons`, `weather`, `severities` and `priorities`.

### Web UI
When `web/dist` exists (`npm run build` in `web/`), the React UI is served at `/ui`. Client-side routes such as
`/ui/record/ROW-000123` return the app shell. These routes are excluded from the OpenAPI schema.
