"""Streamlit front end for the Telecom NetOps Assistant (Phase 28).

    streamlit run frontend.py            (expects the API at API_URL, default http://localhost:8000)

The UI talks to the FastAPI backend only. Field labels, units, valid ranges and
condition thresholds come from GET /api/v1/reference, so no number is restated here.
Pages: Analyze a case (default) · Search records · Record details · Assess measurements.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="NetOps Assistant", page_icon=":material/cell_tower:", layout="wide")

EXAMPLES = {
    "Faults rising, handover falling": "A network region has increasing fault occurrence and declining handover "
                                       "success. What corrective actions should be considered?",
    "Latency and loss, good signal": "Cell has high latency and packet loss with good signal strength and normal "
                                     "user load. What similar network conditions occurred before?",
    "Weak signal and drops in snow": "Poor signal strength and frequent call drops during snow",
    "Load up, throughput down": "High user load with reduced throughput and rising latency",
    "Authentication failures": "Many authentication failures and security incidents on a site",
}
SEVERITY_COLOR = {"LOW": "gray", "MEDIUM": "orange", "HIGH": "red", "CRITICAL": "red"}
PRIORITY_COLOR = {"P1": "red", "P2": "red", "P3": "orange", "P4": "gray"}
RISK_COLOR = {"Low": "gray", "Medium": "orange", "High": "red", "Critical": "red"}
SOURCE_LABELS = {"USER_INPUT": "Your input", "RAW_DATA": "Raw dataset value", "DERIVED_FEATURE": "Derived label",
                 "HISTORICAL_EVIDENCE": "Similar records", "ML_PREDICTION": "ML prediction",
                 "ENGINEERING_HYPOTHESIS": "Hypothesis", "RECOMMENDATION": "Recommendation"}
CONDITION_SOURCES = {"stated": "Stated in the description", "measured": "Measured value"}
ROLE_GROUPS = [("kpi", "Radio and service KPIs"), ("load", "Load"), ("operational", "Alarms and configuration"),
               ("security", "Security"), ("environment", "Environment")]
DECIMALS = {"%": 2, "ms": 1, "dBm": 1, "Mbps": 1, "GB": 2, "°C": 1, "hours": 2}
OPERATOR_WORDS = {"<": "below", ">": "above", "<=": "at or below", ">=": "at or above"}
ERROR_HINTS = {
    "connection_error": "Start the API (uvicorn src.main:app) or check the API_URL setting, then try again.",
    "invalid_filter": "Adjust the filters and try again.",
    "validation_error": "One of the values is outside its allowed range. Correct it and try again.",
    "not_found": "Check the record ID. IDs look like ROW-000123.",
    "agent_failure": "An analysis step failed on the server. Try again; if it repeats, check the API log.",
}


# ============================================================================================ API
def api(method: str, path: str, timeout: float = 300, **kw) -> tuple[dict | None, dict | None]:
    """Returns (body, None) on success or (None, error) on failure, with a structured error in both cases."""
    try:
        r = httpx.request(method, f"{API_URL}{path}", timeout=timeout, **kw)
    except httpx.HTTPError as exc:
        return None, {"code": "connection_error", "message": f"Cannot reach the API at {API_URL} ({exc})."}
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    if r.status_code < 400:
        return body, None
    return None, body.get("error", {"code": f"http_{r.status_code}", "message": f"Request failed ({r.status_code})."})


@st.cache_data(ttl=30, show_spinner=False)
def health() -> dict | None:
    return api("GET", "/health", timeout=5)[0]


@st.cache_data(ttl=600, show_spinner=False)
def reference() -> dict | None:
    return api("GET", "/api/v1/reference", timeout=10)[0]


def show_error(err: dict | None) -> None:
    if not err:
        return
    code = err.get("code", "error")
    hint = ("Describe a network observation or question; requests unrelated to network operations are not "
            "processed." if code.startswith("guardrail") else ERROR_HINTS.get(code, ""))
    st.error(f"**{err.get('message', 'The request failed.')}**  \n{hint}  \n`{code}`", icon=":material/error:")


# ====================================================================================== formatting
def fields() -> dict[str, dict]:
    return {f["name"]: f for f in (reference() or {}).get("fields", [])}


def conditions() -> dict[str, dict]:
    return {c["name"]: c for c in (reference() or {}).get("conditions", [])}


def cond_label(name: str) -> str:
    return conditions().get(name, {}).get("label") or name.replace("_", " ")


def fmt(value: Any, metric: str | None = None, unit: bool = False) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "–"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if not isinstance(value, (int, float)):
        return str(value)
    f = fields().get(metric or "", {})
    u = f.get("unit") or ""
    text = f"{int(round(value)):,}" if f.get("type") == "int" else f"{value:,.{DECIMALS.get(u, 2)}f}"
    return text + ((u if u in ("%", "°C") else f" {u}") if unit and u else "")


def threshold_text(c: dict) -> str:
    basis = "worst 10% of records" if c["method"] == "percentile" else "engineering threshold"
    return f"{OPERATOR_WORDS.get(c['operator'], c['operator'])} {fmt(c['threshold'], c['metric'], True)} ({basis})"


def badge(text: str, color: str) -> str:
    return f":{color}-badge[{text}]"


def readable(reason: str | None) -> str | None:
    """Provider errors embed the API's JSON; keep only its human-readable message."""
    m = re.search(r"'message':\s*'([^']+)'", reason or "")
    return m.group(1) if m else reason


def page_header(title: str, description: str) -> None:
    st.markdown(f"## {title}")
    st.caption(description)


# ========================================================================================= inputs
def filter_inputs(prefix: str, extended: bool = False) -> None:
    ref = reference() or {}
    states = ref.get("states", {})
    any_ = ["Any"]
    c1, c2, c3, c4 = st.columns(4)
    c1.selectbox("State or territory", any_ + sorted(states, key=states.get), key=f"{prefix}_state",
                 format_func=lambda s: s if s == "Any" else f"{states[s]} ({s})", persist_state="session")
    c2.text_input("City", key=f"{prefix}_city", placeholder="Exact name", persist_state="session",
                  help="Must match a dataset city; case is ignored.")
    c3.selectbox("Season", any_ + ref.get("seasons", []), key=f"{prefix}_season", persist_state="session")
    c4.selectbox("Weather", any_ + ref.get("weather", []), key=f"{prefix}_weather", persist_state="session")
    if not extended:
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.selectbox("Category", any_ + ref.get("categories", []), key=f"{prefix}_category", persist_state="session",
                 help="Derived category stored with each record.")
    c2.selectbox("Severity", any_ + ref.get("severities", []), key=f"{prefix}_severity", persist_state="session")
    c3.selectbox("Priority", any_ + ref.get("priorities", []), key=f"{prefix}_priority", persist_state="session")
    c4.selectbox("Customer issue reported", ["Any", "Yes", "No"], key=f"{prefix}_issue", persist_state="session")
    st.multiselect("Must show all of these conditions", list(conditions()), format_func=cond_label,
                   key=f"{prefix}_conditions", placeholder="Any conditions", persist_state="session")


def collect_filters(prefix: str) -> dict | None:
    ss = st.session_state
    out = {k: ss.get(f"{prefix}_{k}") for k in ("state", "season", "weather", "category", "severity", "priority")}
    out = {k: v for k, v in out.items() if v and v != "Any"}
    if (ss.get(f"{prefix}_city") or "").strip():
        out["city"] = ss[f"{prefix}_city"].strip()
    if ss.get(f"{prefix}_issue") in ("Yes", "No"):
        out["issue_reported"] = ss[f"{prefix}_issue"] == "Yes"
    if ss.get(f"{prefix}_conditions"):
        out["conditions"] = list(ss[f"{prefix}_conditions"])
    return out or None


def numeric_inputs() -> list[dict]:
    return [f for f in (reference() or {}).get("fields", []) if f.get("input") and f["type"] in ("float", "int")]


def metric_inputs(prefix: str) -> None:
    by_metric: dict[str, list[dict]] = {}
    for c in conditions().values():
        by_metric.setdefault(c["metric"], []).append(c)
    for role, title in ROLE_GROUPS:
        group = [f for f in numeric_inputs() if f["role"] == role]
        if not group:
            continue
        st.markdown(f"**{title}**")
        cols = st.columns(3)
        for i, f in enumerate(group):
            is_int = f["type"] == "int"
            hint = " ".join(f"Flagged as “{c['label']}” {threshold_text(c)}." for c in by_metric.get(f["name"], []))
            cols[i % 3].number_input(
                f["label"], key=f"{prefix}_{f['name']}", value=None, persist_state="session",
                min_value=(int(f["min"]) if is_int else float(f["min"])) if f.get("min") is not None else None,
                max_value=(int(f["max"]) if is_int else float(f["max"])) if f.get("max") is not None else None,
                step=1 if is_int else None, format="%d" if is_int else None, placeholder="Not measured",
                help=hint or None)


def collect_metrics(prefix: str) -> dict | None:
    out = {f["name"]: st.session_state.get(f"{prefix}_{f['name']}") for f in numeric_inputs()}
    out = {k: v for k, v in out.items() if v is not None}
    return out or None


# ======================================================================================== records
def records_table(results: list[dict], key: str, with_similarity: bool = True) -> dict | None:
    rows = []
    for x in results:
        m, md = x["metrics"], x["metadata"]
        row = {"Record": x["record_id"]}
        if with_similarity:
            row["Similarity"] = x.get("hybrid_score")
        row.update({"Category": x["category"], "Severity": x["severity"],
                    "Conditions": [cond_label(c) for c in x["conditions"]],
                    "Latency (ms)": m.get("latency_ms"), "Packet loss (%)": m.get("packet_loss_pct"),
                    "Handover success (%)": m.get("handover_success_pct"), "Signal (dBm)": m.get("signal_strength_dbm"),
                    "Users": m.get("user_count"), "Fault rate (%) observed": m.get("fault_rate_pct"),
                    "Location": f"{md['city']}, {md['state']}", "Season · weather": f"{md['season']} · {md['weather']}"})
        rows.append(row)
    event = st.dataframe(
        pd.DataFrame(rows), hide_index=True, key=key, on_select="rerun", selection_mode="single-row",
        column_config={
            "Record": st.column_config.TextColumn(pinned=True, width=95),
            "Similarity": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f", width="small",
                                                          help="Hybrid score of semantic, keyword and metric similarity"),
            "Conditions": st.column_config.ListColumn(width="medium"),
            "Latency (ms)": st.column_config.NumberColumn(format="%.1f"),
            "Packet loss (%)": st.column_config.NumberColumn(format="%.2f"),
            "Handover success (%)": st.column_config.NumberColumn(format="%.2f"),
            "Signal (dBm)": st.column_config.NumberColumn(format="%.1f"),
            "Users": st.column_config.NumberColumn(format="%d"),
            "Fault rate (%) observed": st.column_config.NumberColumn(format="%.2f"),
        })
    sel = event.selection.rows if event else []
    return results[sel[0]] if sel else None


def record_detail(rec: dict) -> None:
    md, metrics = rec["metadata"], rec["metrics"]
    states = (reference() or {}).get("states", {})
    st.markdown(f"#### {rec['record_id']}")
    st.caption(f"{md['city']}, {states.get(md['state'], md['state'])} ({md['state']}) · ZIP {md['zip']} · "
               f"{md['season']} · {md['weather']} · {fmt(md.get('temperature_c'), 'temperature_c', True)}, humidity "
               f"{fmt(md.get('humidity_pct'), 'humidity_pct', True)} · customer issue reported: "
               f"{fmt(md.get('issue_reported'))}")
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown("**Derived labels** — computed from the raw values; not ground-truth annotations")
        secondary = ", ".join(rec.get("secondary_categories") or []) or "none"
        st.markdown(f"Category: **{rec['category']}** (also: {secondary})  \n"
                    f"Severity: {badge(rec['severity'], SEVERITY_COLOR.get(rec['severity'], 'gray'))} "
                    f"score {rec.get('severity_score', '–')}  \n"
                    f"Priority: {badge(rec['priority'], PRIORITY_COLOR.get(rec['priority'], 'gray'))}")
        conds = rec.get("conditions") or []
        st.markdown("Conditions: " + (" ".join(badge(cond_label(c), "blue") for c in conds) if conds else "none"))
    with right:
        flags: dict[str, list[str]] = {}
        for name in rec.get("conditions") or []:
            c = conditions().get(name)
            if c:
                flags.setdefault(c["metric"], []).append(f"{c['label']} ({threshold_text(c)})")
        meta = fields()
        outcomes = ("fault_rate_pct", "mttr_hours")
        rows = []
        for role, title in ROLE_GROUPS + [("_outcome", "Outcomes (observed)")]:
            for n, v in metrics.items():
                in_group = (n in outcomes) if role == "_outcome" else (meta.get(n, {}).get("role") == role
                                                                        and n not in outcomes)
                if in_group:
                    rows.append({"Group": title, "Measurement": meta.get(n, {}).get("label", n),
                                 "Value (raw)": fmt(v, n, True), "Derived condition": "; ".join(flags.get(n, []))})
        st.dataframe(pd.DataFrame(rows), hide_index=True, height=min(38 + 35 * len(rows), 460))


# ===================================================================================== page: analyze
def analysis_summary_panel(res: dict) -> None:
    c, pr, risk, val = res["classification"], res["priority"], res["risk"], res["validation"]
    pred, hist = risk["predicted"], risk["historical"]
    checks = [val["evidence_grounded"], val["diagnosis_supported"], val["recommendation_relevant"],
              not val["hallucination_detected"]]
    with st.container(border=True):
        st.caption("ASSESSMENT")
        st.markdown(f"### {c['category']}")
        if c["secondary_categories"]:
            st.caption("Also " + ", ".join(c["secondary_categories"]))
        st.markdown(
            f"Severity  {badge(c['severity'], SEVERITY_COLOR.get(c['severity'], 'gray'))}  \n"
            f"Priority  {badge(pr['priority'] + ' · ' + pr['priority_label'], PRIORITY_COLOR.get(pr['priority'], 'gray'))}  \n"
            + (f"Predicted fault risk  **{fmt(pred.get('predicted_fault_rate_pct'))}%** "
               f"{badge(pred.get('risk_level', ''), RISK_COLOR.get(pred.get('risk_level'), 'gray'))}  \n"
               if pred.get("available") else "")
            + (f"Observed in similar records  **{fmt(hist['mean'])}%**  \n" if hist else "")
            + f"Review  **{sum(checks)} of 4 checks passed**")
        st.caption(f"Labels are derived from the evidence; severity basis: {c['severity_basis']}.")


def analysis_tabs(res: dict) -> None:
    c, pr, risk, rca = res["classification"], res["priority"], res["risk"], res["root_cause"]
    recs, val, n = res["recommendations"], res["validation"], res["narrative"]
    pred, hist = risk["predicted"], risk["historical"]
    observed = res["evidence"]["observed"]
    titles = {h["id"]: h["title"] for h in rca["hypotheses"]}
    t1, t2, t3, t4, t5 = st.tabs(["Diagnosis and actions", "Evidence", f"Similar records ({len(res['retrieval']['results'])})",
                                  "Automatic review", "Sources and report"])
    with t1:
        if rca.get("primary_hypothesis"):
            st.caption("LEADING HYPOTHESIS — an engineering hypothesis ranked by evidence, not a confirmed cause")
            st.markdown(f"#### {rca['primary_hypothesis']}")
            st.progress(min(1.0, rca["confidence"] or 0),
                        text=f"Confidence {rca['confidence']:.2f} — heuristic score from the playbook, not a probability")
            sup = [(e.get("label") or (cond_label(e["condition"]) if e.get("condition") else f"{e.get('context')} = {e.get('value')}"))
                   + (f" ({CONDITION_SOURCES.get(e.get('source'), e.get('source'))})" if e.get("source") else "")
                   for e in rca["supporting_evidence"]]
            st.markdown("**Supported by:** " + ("; ".join(sup) or "no direct evidence in the description"))
            if rca["hypotheses"][0].get("historical_support"):
                st.markdown(f"**Similar records:** {rca['hypotheses'][0]['historical_support']}")
            if rca["contradicting_evidence"]:
                st.markdown("**Against it:** " + "; ".join(rca["contradicting_evidence"]))
        else:
            st.write(rca["statement"])
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("##### Recommended actions")
            st.caption(recs["disclaimer"])
            groups: dict[str, list[dict]] = {}
            for a in recs["actions"]:
                groups.setdefault(a["hypothesis"], []).append(a)
            i = 1
            for hid, acts in groups.items():
                conf = badge(f"confidence {acts[0]['hypothesis_confidence']:.2f}", "gray")
                st.markdown(f"**If the cause is: {titles.get(hid, hid)}** {conf}")
                for a in acts:
                    basis = (f"Based on: {', '.join(cond_label(b) for b in a['based_on'])}" if a["based_on"]
                             else "Not tied to a condition observed in this case")
                    st.markdown(f"{i}. {a['action']}  \n:gray[{basis}]")
                    i += 1
            if recs["monitoring"]:
                st.markdown("**Monitor after any change**")
                st.markdown("\n".join(f"- {m['action']}" for m in recs["monitoring"]))
        with right:
            st.markdown("##### Verify before acting")
            st.markdown("\n".join(f"- {x['check']}" for x in recs["checks"][:6]))
            if len(recs["checks"]) > 6:
                with st.expander(f"Show all {len(recs['checks'])} checks"):
                    st.markdown("\n".join(f"- {x['check']}" for x in recs["checks"][6:]))
            if rca.get("additional_evidence_required"):
                st.markdown("##### Data needed to confirm the cause")
                st.caption("Not in the dataset; collect it from the network before acting on a hypothesis.")
                st.markdown("\n".join(f"- {x}" for x in rca["additional_evidence_required"]))
        if len(rca["hypotheses"]) > 1:
            st.markdown("##### Other hypotheses considered")
            st.dataframe(pd.DataFrame([{
                "Hypothesis": h["title"], "Confidence": h["confidence"],
                "Supported by": ", ".join(e.get("label") or e.get("condition") or str(e.get("context")) for e in h["supporting_evidence"]) or "no direct evidence",
                "Similar records": h["historical_support"]} for h in rca["hypotheses"][1:]]),
                hide_index=True, column_config={"Confidence": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f")})
    with t2:
        st.markdown("##### Fault risk — a prediction and an observed rate, kept separate")
        a, b = st.columns(2)
        with a.container(border=True):
            st.caption("PREDICTED · ML MODEL")
            if pred.get("available"):
                st.markdown(f"### {fmt(pred['predicted_fault_rate_pct'])}% "
                            f"{badge(pred['risk_level'], RISK_COLOR.get(pred['risk_level'], 'gray'))}")
                st.caption(f"80% interval {fmt(pred['interval_80pct'][0])}–{fmt(pred['interval_80pct'][1])}% · "
                           f"input coverage {int(pred['input_coverage'] * 100)}% · "
                           f"{pred['model']['algorithm']}, test R² {pred['model']['test_r2']:.3f}")
            else:
                st.write(pred.get("reason", "No prediction available."))
        with b.container(border=True):
            st.caption("OBSERVED · SIMILAR RECORDS")
            if hist:
                st.markdown(f"### {fmt(hist['mean'])}% mean")
                st.caption(f"Median {fmt(hist['median'])}% · range {fmt(hist['min'])}–{fmt(hist['max'])}% · "
                           f"{hist['n_records']} records. Recorded in the dataset, not a prediction.")
            else:
                st.write("No similar records were retrieved.")
        if pred.get("low_input_coverage") and pred.get("caveat"):
            st.info(pred["caveat"], icon=":material/info:")
        st.markdown("##### Conditions — derived from your description and values using documented thresholds")
        rows = []
        for name in observed["present"]:
            m = next((x for x in observed["measured_evidence"] if x["condition"] == name), None)
            cdef = conditions().get(name)
            rows.append({"Condition": cond_label(name), "Status": "Present",
                         "Source": CONDITION_SOURCES.get(observed["condition_sources"].get(name), "Stated"),
                         "Value": fmt(m["value"], m["metric"], True) if m else "Not supplied",
                         "Threshold": threshold_text(cdef) if cdef else "–"})
        for name in observed["absent"]:
            cdef = conditions().get(name)
            v = observed["values"].get(cdef["metric"]) if cdef else None
            rows.append({"Condition": cond_label(name), "Status": "Not present",
                         "Source": "Measured value" if v is not None else "Stated in the description",
                         "Value": fmt(v, cdef["metric"], True) if v is not None else "Not supplied",
                         "Threshold": threshold_text(cdef) if cdef else "–"})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        else:
            st.caption("No conditions were recognised. Add symptoms or measured values for a sharper analysis.")
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown(f"**Category: {c['category']}**")
            st.markdown("\n".join(f"- {r}" for r in c["category_reasons"]))
            ml = c["ml_category"]
            st.caption(f"ML classifier: {ml['predicted_category']} "
                       f"({'agrees' if c['ml_agrees_with_rules'] else 'disagrees'} with the rules). {ml.get('label', '')}"
                       if ml.get("available") else f"ML classifier not used: {ml.get('reason')}")
        with b:
            st.markdown(f"**Severity {c['severity']} and priority {pr['priority']} · {pr['priority_label']}**")
            st.caption(f"Severity basis: {c['severity_basis']}. Severity score {c['severity_score']}; "
                       f"priority score {pr['priority_score']}.")
            st.markdown("\n".join(f"- {r}" for r in c["severity_reasons"] + pr["reasons"]))
    with t3:
        r = res["retrieval"]
        st.caption(f"{r['matching_records']:,} records match the filters · {r['candidates']} candidates re-scored · "
                   f"similarity = semantic {r['weights'].get('semantic', 0):.2f} + keyword {r['weights'].get('keyword', 0):.2f}"
                   f" + metric {r['weights'].get('metric', 0):.2f}. Select a row to see all of its measurements.")
        sel = records_table(r["results"], key="an_records")
        if sel:
            with st.container(border=True):
                record_detail(sel)
                if st.button("Open on the Record details page", icon=":material/open_in_new:", type="tertiary"):
                    st.session_state["id"] = sel["record_id"]
                    st.switch_page(PAGES["record"])
    with t4:
        st.caption(f"Method: {val['method']} · score {val['score']}")
        checks = [(val["evidence_grounded"], "Evidence grounded", "Every statement traces back to the evidence package."),
                  (val["diagnosis_supported"], "Diagnosis supported", "The leading hypothesis has evidence in the case."),
                  (val["recommendation_relevant"], "Recommendations relevant", "Each action addresses an observed condition or the leading hypothesis."),
                  (not val["hallucination_detected"], "No hallucination detected", "No numbers, IDs or claims outside the evidence.")]
        a, b = st.columns(2)
        for i, (ok, label, note) in enumerate(checks):
            (a if i % 2 == 0 else b).markdown(
                f"{':green[:material/check_circle: **Passed**]' if ok else ':red[:material/cancel: **Failed**]'} — "
                f"{label}  \n:gray[{note}]")
        if not val["recommendation_relevant"]:
            st.caption("At least one action belongs to a lower-ranked hypothesis and is not tied to a condition observed "
                       "in this case (marked 'Not tied to a condition observed in this case').")
        if val["issues"]:
            st.markdown("**Reviewer notes**")
            st.markdown("\n".join(f"- {i}" for i in val["issues"]))
        info = val["llm_judge_info"]
        st.caption(f"LLM review: {'ran' if val['llm_judge'] else info.get('status')}"
                   + (f" — {readable(info.get('reason'))}" if info.get("reason") and not val["llm_judge"] else "")
                   + (f" · explanation attempts: {', '.join(str(a.get('status')) + (' (rejected by grounding)' if a.get('grounding_issues') else '') for a in n['attempts'])}"
                      if n["attempts"] else ""))
    with t5:
        types = sorted({p["source_type"] for p in res["provenance"]})
        chosen = st.pills("Filter statements by source", types, selection_mode="multi", format_func=lambda t: SOURCE_LABELS.get(t, t),
                          key="an_prov_filter")
        rows = [{"Statement": p["claim"], "Source": SOURCE_LABELS.get(p["source_type"], p["source_type"])}
                for p in res["provenance"] if not chosen or p["source_type"] in chosen]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button("Download report (.txt)", res["report_text"], file_name=f"netops-report-{res['analysis_id'][:8]}.txt",
                           icon=":material/download:", on_click="ignore")
        d2.download_button("Download full result (.json)", json.dumps(res, indent=2),
                           file_name=f"netops-analysis-{res['analysis_id'][:8]}.json", mime="application/json",
                           icon=":material/download:", on_click="ignore")
        with st.expander("Full text report"):
            st.code(res["report_text"], language=None)
        with st.expander(f"Agent trace · {res['timing_s']} s total"):
            st.dataframe(pd.DataFrame(res["agent_trace"]), hide_index=True)
        st.markdown("**Limitations**")
        st.markdown("\n".join(f"- {l}" for l in res["limitations"]))


def feedback_form(res: dict) -> None:
    with st.container(border=True):
        st.markdown("**Was this analysis useful?**")
        if st.session_state.get("an_feedback_for") == res["analysis_id"]:
            st.success("Thank you — your feedback is stored for evaluation review.", icon=":material/check_circle:")
            return
        with st.form("an_feedback", border=False):
            comment = st.text_input("Comment (optional)", placeholder="What was right or wrong?", max_chars=2000)
            c1, c2, _ = st.columns([1, 1, 4])
            useful = c1.form_submit_button("Useful", use_container_width=True)
            not_useful = c2.form_submit_button("Not useful", use_container_width=True)
        if useful or not_useful:
            _, err = api("POST", "/api/v1/feedback", json={
                "analysis_id": res["analysis_id"], "query": res["query"],
                "retrieved_record_ids": [x["record_id"] for x in res["retrieval"]["results"]],
                "diagnosis": {"primary_hypothesis": res["root_cause"].get("primary_hypothesis"),
                              "confidence": res["root_cause"].get("confidence")},
                "recommendations": [a["action"] for a in res["recommendations"]["actions"]],
                "risk_prediction": {k: res["risk"]["predicted"].get(k) for k in ("predicted_fault_rate_pct", "risk_level")},
                "rating": "useful" if useful else "not_useful", "comment": comment or None})
            if err:
                show_error(err)
            else:
                st.session_state["an_feedback_for"] = res["analysis_id"]
                st.rerun()


def load_example() -> None:
    choice = st.session_state.get("an_example")
    if choice:
        st.session_state["an_query"] = EXAMPLES[choice]


def page_analyze() -> None:
    page_header("Analyze a case", "Describe what you are seeing on the network. The assistant retrieves similar "
                "historical records, classifies the case, ranks likely causes and suggests what to check — every "
                "statement is linked to its source.")
    st.selectbox("Start from an example", list(EXAMPLES), index=None, placeholder="Choose an example…",
                 key="an_example", on_change=load_example)
    with st.form("an_form"):
        st.text_area("Case description", key="an_query", height=100, max_chars=1000, persist_state="session",
                     placeholder="e.g. High latency and packet loss on a cell during rain, signal looks normal.",
                     help="Plain language. Mention symptoms, conditions such as weather or load, and what you want to know.")
        with st.expander("Limit similar records by location, season or weather"):
            filter_inputs("an_f")
        with st.expander("Add measured values (optional)"):
            st.caption("Values you enter are treated as measurements of this case: they decide which conditions are "
                       "present and steer the search.")
            metric_inputs("an_m")
        c1, c2, c3 = st.columns([1, 2, 1], vertical_alignment="bottom")
        c1.selectbox("Similar records", list(range(1, 11)), index=4, key="an_topk", persist_state="session")
        c2.toggle("Use the LLM for the explanation and review", value=True, key="an_llm", persist_state="session",
                  help="If the LLM is unavailable, a fixed-template explanation and rule-based checks are used.")
        submitted = c3.form_submit_button("Analyze case", type="primary", use_container_width=True)
    if submitted:
        query = (st.session_state.get("an_query") or "").strip()
        if not query:
            st.warning("Describe what you are seeing first, for example “high latency and packet loss during rain”.")
        else:
            with st.spinner("Retrieving similar records and running the analysis agents"
                            + (" — with the LLM this can take up to a minute…" if st.session_state.get("an_llm") else "…")):
                res, err = api("POST", "/api/v1/analyze", json={
                    "query": query, "filters": collect_filters("an_f"), "top_k": st.session_state.get("an_topk", 5),
                    "use_llm": bool(st.session_state.get("an_llm")), "metrics": collect_metrics("an_m")})
            st.session_state["an_result"], st.session_state["an_error"] = res, err
            st.session_state["an_ran_llm"] = bool(st.session_state.get("an_llm"))
    show_error(st.session_state.get("an_error"))
    res = st.session_state.get("an_result")
    if not res:
        st.caption("Results appear here: the case assessment first, then the diagnosis, recommended actions, the "
                   "evidence behind them and the automatic review.")
        return
    st.divider()
    main, side = st.columns([3, 1], gap="large")
    with side:
        analysis_summary_panel(res)
    with main:
        n = res["narrative"]
        st.markdown("##### Summary " + (":gray[· written by the LLM, passed the grounding check]" if n["source"] == "llm"
                                        else ":gray[· fixed-template explanation]"))
        if st.session_state.get("an_ran_llm") and n["source"] != "llm":
            last = n["attempts"][-1] if n["attempts"] else {}
            reason = readable(last.get("reason") or res["validation"]["llm_judge_info"].get("reason"))
            st.warning(("The LLM's explanation was rejected by the grounding check, so the fixed-template explanation "
                        "is shown." if n.get("llm_rejected") else
                        f"{reason + '. ' if reason else ''}The fixed-template explanation is shown."),
                       icon=":material/warning:")
        st.write(n["narrative"]["summary"])
        analysis_tabs(res)
    feedback_form(res)


# ====================================================================================== page: search
def page_search() -> None:
    page_header("Search records", "Find historical records that resemble a description. Filters are applied "
                "strictly: a record that does not match every filter is never returned.")
    with st.form("se_form"):
        c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
        c1.text_input("Look for", key="se_query", placeholder="e.g. weak signal and call drops", persist_state="session")
        c2.selectbox("Results", [5, 10, 15, 20], index=1, key="se_topk", persist_state="session")
        with st.expander("Filters", expanded=False):
            filter_inputs("se_f", extended=True)
        submitted = st.form_submit_button("Search", type="primary")
    if submitted:
        q = (st.session_state.get("se_query") or "").strip()
        if not q:
            st.warning("Enter what to look for, for example “weak signal and call drops”.")
        else:
            with st.spinner("Searching…"):
                res, err = api("POST", "/api/v1/search", json={"query": q, "filters": collect_filters("se_f"),
                                                               "top_k": st.session_state.get("se_topk", 10)})
            st.session_state["se_result"], st.session_state["se_error"] = res, err
    show_error(st.session_state.get("se_error"))
    res = st.session_state.get("se_result")
    if not res:
        st.caption("Describe a pattern and press Search. Use the filters to restrict results to a place, season, "
                   "category or set of conditions.")
        return
    st.markdown(f"##### {len(res['results'])} results for “{res['query']}”")
    st.caption(f"{res['matching_records']:,} records match the filters · {res['candidates']} candidates scored · "
               "most similar first. Select a row to see its measurements.")
    if not res["results"]:
        st.info("No records match these filters. Remove a filter or a required condition and search again.")
        return
    sel = records_table(res["results"], key="se_records")
    if sel:
        with st.container(border=True):
            record_detail(sel)


# ====================================================================================== page: record
def page_record() -> None:
    page_header("Record details", "Look up one historical record to see its raw measurements next to the labels "
                "derived from them, plus what the models say about it.")
    rid = st.text_input("Record ID", key="id", bind="query-params", placeholder="ROW-000123",
                        help="Format ROW-000123 (the number alone also works).")
    rid = (rid or "").strip().upper()
    if re.fullmatch(r"\d{1,6}", rid):
        rid = f"ROW-{int(rid):06d}"
    if not rid:
        st.caption("Enter a record ID above, or open one from a similar-records table.")
        return
    if not re.fullmatch(r"ROW-\d{6}", rid):
        st.warning("Use the format ROW-000123.")
        return
    rec, err = api("GET", f"/api/v1/records/{rid}")
    if err:
        show_error(err)
        return
    record_detail(rec)
    st.markdown("##### Model checks — predictions for this record, next to what the dataset recorded")
    risk, _ = api("POST", "/api/v1/risk", json={"record_id": rid})
    cls, _ = api("POST", "/api/v1/classify", json={"record_id": rid})
    a, b = st.columns(2)
    with a.container(border=True):
        st.caption("FAULT OCCURRENCE")
        p = (risk or {}).get("predicted", {})
        if p.get("available"):
            st.markdown(f"### {fmt(p['predicted_fault_rate_pct'])}% {badge(p['risk_level'], RISK_COLOR.get(p['risk_level'], 'gray'))}")
            st.caption(f"ML prediction · 80% interval {fmt(p['interval_80pct'][0])}–{fmt(p['interval_80pct'][1])}% · "
                       f"recorded in the dataset: {fmt(risk['historical_fault_rate_pct']['value'])}% (observed)")
    with b.container(border=True):
        st.caption("CATEGORY · ML CLASSIFIER")
        ml = (cls or {}).get("classification", {}).get("ml_category", {})
        if ml.get("available"):
            st.markdown(f"### {ml['predicted_category']}")
            st.caption(f"Probability {ml['probability'] * 100:.1f}% · "
                       f"{'agrees with' if cls['classification']['ml_agrees_with_rules'] else 'differs from'} the "
                       f"rule-based category ({cls['classification']['category']}). {ml.get('label', '')}")


# ====================================================================================== page: assess
def page_assess() -> None:
    page_header("Assess measurements", "Enter measurements for one cell to see which conditions they trigger, the "
                "derived category, severity and priority, and the predicted fault risk. No similar-record search or "
                "explanation is run here.")
    ref = reference() or {}
    left, right = st.columns([3, 2], gap="large")
    with left, st.form("as_form"):
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Season", ["Not specified"] + ref.get("seasons", []), key="as_season", persist_state="session")
        c2.selectbox("Weather", ["Not specified"] + ref.get("weather", []), key="as_weather", persist_state="session")
        c3.selectbox("Customer issue reported", ["Not specified", "Yes", "No"], key="as_issue", persist_state="session")
        metric_inputs("as_m")
        submitted = st.form_submit_button("Assess", type="primary")
    if submitted:
        metrics = collect_metrics("as_m") or {}
        ss = st.session_state
        if ss.get("as_season") not in (None, "Not specified"):
            metrics["season"] = ss["as_season"]
        if ss.get("as_weather") not in (None, "Not specified"):
            metrics["weather"] = ss["as_weather"]
        if ss.get("as_issue") in ("Yes", "No"):
            metrics["issue_reported"] = ss["as_issue"] == "Yes"
        if not metrics:
            st.session_state["as_result"], st.session_state["as_error"] = None, {
                "code": "validation_error", "message": "Enter at least one measurement."}
        else:
            cls, err1 = api("POST", "/api/v1/classify", json={"metrics": metrics})
            risk, err2 = api("POST", "/api/v1/risk", json={"metrics": metrics})
            st.session_state["as_result"] = {"cls": cls, "risk": risk} if cls and risk else None
            st.session_state["as_error"] = err1 or err2
    with right:
        show_error(st.session_state.get("as_error"))
        out = st.session_state.get("as_result")
        if not out:
            st.caption("Enter the measurements you have — any subset works — and press Assess. The help icon on each "
                       "field shows the threshold at which it is flagged.")
            return
        cls, pred = out["cls"], out["risk"]["predicted"]
        c = cls["classification"]
        with st.container(border=True):
            st.caption("ASSESSMENT OF YOUR VALUES")
            st.markdown(f"### {c['category']}")
            st.markdown(f"Severity {badge(c['severity'], SEVERITY_COLOR.get(c['severity'], 'gray'))} score {c['severity_score']}  \n"
                        f"Priority {badge(cls['priority']['priority'] + ' · ' + cls['priority']['priority_label'], PRIORITY_COLOR.get(cls['priority']['priority'], 'gray'))}  \n"
                        + (f"Predicted fault risk **{fmt(pred['predicted_fault_rate_pct'])}%** "
                           f"{badge(pred['risk_level'], RISK_COLOR.get(pred['risk_level'], 'gray'))}" if pred.get("available") else ""))
            if pred.get("available"):
                st.caption(f"ML prediction, 80% interval {fmt(pred['interval_80pct'][0])}–{fmt(pred['interval_80pct'][1])}%; "
                           f"model inputs covered {int(pred['input_coverage'] * 100)}%.")
            if st.button("Analyze a case with these values", icon=":material/arrow_forward:"):
                for f in numeric_inputs():
                    v = st.session_state.get(f"as_m_{f['name']}")
                    if v is not None:
                        st.session_state[f"an_m_{f['name']}"] = v
                st.switch_page(PAGES["analyze"])
        if pred.get("low_input_coverage") and pred.get("caveat"):
            st.info(pred["caveat"], icon=":material/info:")
        st.markdown(f"**Conditions triggered** ({len(cls['conditions']['conditions'])})")
        ev = cls["conditions"]["evidence"]
        st.markdown("\n".join(f"- **{cond_label(e['condition'])}**: {fmt(e['value'], e['metric'], True)} — flagged "
                              f"{threshold_text(conditions()[e['condition']]) if e['condition'] in conditions() else ''}"
                              for e in ev) if ev else "None of the entered values cross a condition threshold.")
        st.caption(f"{len(cls['conditions']['unavailable'])} conditions not assessed because the measurement was not entered.")
        st.markdown("**Why these labels**")
        st.markdown("\n".join(f"- {r}" for r in c["category_reasons"] + c["severity_reasons"] + cls["priority"]["reasons"]))


# ======================================================================================== navigation
PAGES = {
    "analyze": st.Page(page_analyze, title="Analyze a case", icon=":material/troubleshoot:", default=True),
    "search": st.Page(page_search, title="Search records", icon=":material/search:", url_path="search"),
    "record": st.Page(page_record, title="Record details", icon=":material/description:", url_path="record"),
    "assess": st.Page(page_assess, title="Assess measurements", icon=":material/tune:", url_path="assess"),
}
nav = st.navigation(list(PAGES.values()), position="top")

h = health()
if h is None:
    st.error(f"**The API is not reachable at {API_URL}.**  \nStart it with `uvicorn src.main:app --reload --port 8000` "
             "(or set API_URL), then reload this page.", icon=":material/cloud_off:")
elif reference() is None:
    st.warning("Could not load reference data (labels and thresholds) from the API.", icon=":material/warning:")
else:
    llm = h["checks"].get("llm", {})
    st.caption(f":{'green' if h['status'] == 'ok' else 'orange'}[●] API {'connected' if h['status'] == 'ok' else 'degraded'}"
               f" · {h['checks'].get('incidents', 0):,} records · LLM "
               f"{'off (template explanations)' if llm.get('provider') == 'offline' else llm.get('model')}")
nav.run()
