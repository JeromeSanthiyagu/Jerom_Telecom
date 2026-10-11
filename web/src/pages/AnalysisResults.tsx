import { useMemo, useState, type FormEvent } from "react";
import { api, RequestError } from "../lib/api";
import { formatNumber, formatPercent, formatValue, plural, thresholdText } from "../lib/format";
import { useAppData, useSessionState } from "../lib/store";
import type { AnalyzeResponse } from "../lib/types";
import { RecordsTable } from "../components/domain";
import {
  Badge, Check, Disclosure, ErrorNotice, Icon, Meter, Notice, PriorityBadge, RiskBadge, Section, SeverityBadge, TextField,
} from "../components/ui";

const SOURCE_LABELS: Record<string, string> = {
  USER_INPUT: "Your input",
  RAW_DATA: "Raw dataset value",
  DERIVED_FEATURE: "Derived label",
  HISTORICAL_EVIDENCE: "Similar records",
  ML_PREDICTION: "ML prediction",
  ENGINEERING_HYPOTHESIS: "Hypothesis",
  RECOMMENDATION: "Recommendation",
};
const CONDITION_SOURCES: Record<string, string> = { stated: "Stated in the description", measured: "Measured value" };

function download(name: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  URL.revokeObjectURL(url);
}

/** Provider errors embed the API's JSON body; keep only its human-readable message. */
function readable(reason: string | undefined): string | undefined {
  return reason?.match(/'message':\s*'([^']+)'/)?.[1] ?? reason;
}

function distribution(d: Record<string, number>): string {
  return Object.entries(d).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${v}`).join(", ");
}

export function AnalysisResults({ result, ranWithLlm }: { result: AnalyzeResponse; ranWithLlm: boolean }) {
  const { conditionLabel, condition, field } = useAppData();
  const { classification: c, priority: pr, risk, root_cause: rca, recommendations: recs, validation: val, narrative: n } = result;
  const pred = risk.predicted;
  const hist = risk.historical;
  const records = result.retrieval.results;
  const observed = result.evidence.observed;
  const hyps = rca.hypotheses;
  const titleOf = useMemo(() => new Map(hyps.map((h) => [h.id, h.title])), [hyps]);

  const checks = [
    { pass: val.evidence_grounded, label: "Evidence grounded", note: "Every statement traces back to the evidence package." },
    { pass: val.diagnosis_supported, label: "Diagnosis supported", note: "The leading hypothesis has evidence in the case and similar records exist." },
    { pass: val.recommendation_relevant, label: "Recommendations relevant", note: "Each action addresses an observed condition or the leading hypothesis." },
    { pass: !val.hallucination_detected, label: "No hallucination detected", note: "No numbers, record IDs or claims outside the evidence." },
  ];
  const passed = checks.filter((x) => x.pass).length;

  // LLM status: say plainly when the user asked for it and it was not used.
  const lastAttempt = n.attempts[n.attempts.length - 1];
  const readableReason = readable(lastAttempt?.reason ?? val.llm_judge_info.reason);
  const llmNotice = ranWithLlm && n.source !== "llm"
    ? n.llm_rejected
      ? "The LLM's explanation was rejected by the grounding check (unsupported numbers or claims), so the fixed-template explanation is shown."
      : `${readableReason ? `${readableReason.replace(/\.?$/, ".")} ` : ""}The fixed-template explanation is shown and only rule-based review checks ran.`
    : null;

  // Group actions by the hypothesis they come from, keeping the ranking order.
  const groups: { id: string; confidence: number; actions: typeof recs.actions }[] = [];
  for (const a of recs.actions) {
    let g = groups.find((x) => x.id === a.hypothesis);
    if (!g) groups.push(g = { id: a.hypothesis, confidence: a.hypothesis_confidence, actions: [] });
    g.actions.push(a);
  }

  const sections = [
    ["answer", "Summary"], ["diagnosis", "Diagnosis"], ["actions", "Recommended actions"], ["risk", "Fault risk"],
    ["conditions", "Conditions and labels"], ["similar", `Similar records (${records.length})`], ["review", "Automatic review"],
    ["sources", "Sources"], ["report", "Report and trace"],
  ];

  return (
    <div className="results" id="results" tabIndex={-1} aria-label="Analysis results">
      <div>
        {/* ------------------------------------------------------------------ summary */}
        <Section id="answer" title="Summary" aside={n.source === "llm" ? "Written by the LLM · passed the grounding check" : "Fixed-template explanation"}>
          <div className="stack">
            {llmNotice && <Notice tone="warn" title="LLM explanation unavailable">{llmNotice}</Notice>}
            <p className="answer">{n.narrative.summary}</p>
            {pred.low_input_coverage && pred.caveat && <Notice tone="info" title="Fault risk is a baseline estimate">{pred.caveat}</Notice>}
          </div>
        </Section>

        {/* ------------------------------------------------------------------ diagnosis */}
        <Section id="diagnosis" title="Diagnosis" aside="Engineering hypotheses ranked by evidence — not confirmed causes">
          {rca.primary_hypothesis ? (
            <div className="stack-lg">
              <div className="stack">
                <span className="eyebrow">Leading hypothesis</span>
                <p className="hypothesis-title">{rca.primary_hypothesis}</p>
                <div className="confidence">
                  <Meter value={rca.confidence ?? 0} label={`Confidence ${formatNumber(rca.confidence, 2)}`} />
                  <span className="muted small" title={rca.confidence_method}>Heuristic score from the playbook, not a probability</span>
                </div>
                <dl className="facts">
                  <dt>Supported by</dt>
                  <dd>{rca.supporting_evidence.length
                    ? rca.supporting_evidence.map((e, i) => (
                      <span key={i} className="tag">{e.label ?? (e.condition ? conditionLabel(e.condition) : `${e.context} = ${String(e.value)}`)}
                        {e.source && <span className="muted"> · {CONDITION_SOURCES[e.source] ?? e.source}</span>}</span>))
                    : <span className="muted">No direct evidence in the description</span>}</dd>
                  {hyps[0]?.historical_support && <><dt>Similar records</dt><dd>{hyps[0].historical_support}</dd></>}
                  {rca.contradicting_evidence.length > 0 && <><dt>Against it</dt><dd>{rca.contradicting_evidence.join("; ")}</dd></>}
                </dl>
              </div>
              {hyps.length > 1 && (
                <div className="stack">
                  <h3>Other hypotheses considered</h3>
                  <div className="table-wrap">
                    <table className="table">
                      <caption className="visually-hidden">Alternative hypotheses</caption>
                      <thead><tr><th scope="col">Hypothesis</th><th scope="col">Confidence</th><th scope="col">Supported by</th><th scope="col">Similar records</th></tr></thead>
                      <tbody>
                        {hyps.slice(1).map((h) => (
                          <tr key={h.id}>
                            <td>{h.title}{h.note && <div className="small muted">{h.note}</div>}</td>
                            <td><Meter value={h.confidence} label={formatNumber(h.confidence, 2)} /></td>
                            <td>{h.supporting_evidence.length ? h.supporting_evidence.map((e, i) => <span key={i} className="tag">{e.label ?? e.condition ?? e.context}</span>) : <span className="muted">No direct evidence</span>}</td>
                            <td className="secondary">{h.historical_support}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          ) : <p>{rca.statement}</p>}
        </Section>

        {/* ------------------------------------------------------------------ actions */}
        <Section id="actions" title="Recommended actions" aside="Options to evaluate — no KPI improvement is predicted">
          <div className="grid-2">
            <div>
              {groups.map((g) => (
                <div className="action-group" key={g.id}>
                  <h3>If the cause is: {titleOf.get(g.id) ?? g.id} <Badge tone="neutral">confidence {formatNumber(g.confidence, 2)}</Badge></h3>
                  <ol className="action-list">
                    {g.actions.map((a) => (
                      <li key={a.action}>
                        {a.action}
                        <span className="why">
                          {a.based_on.length ? `Based on: ${a.based_on.map(conditionLabel).join(", ")}` : "Not tied to a condition observed in this case"}
                        </span>
                      </li>
                    ))}
                  </ol>
                </div>
              ))}
              {recs.monitoring.length > 0 && (
                <div className="action-group">
                  <h3>Monitor after any change</h3>
                  <ul>{recs.monitoring.map((m) => <li key={m.action}>{m.action}</li>)}</ul>
                </div>
              )}
            </div>
            <div className="stack-lg">
              <div className="stack">
                <h3>Verify before acting</h3>
                <ul>{recs.checks.slice(0, 6).map((x) => <li key={x.check}>{x.check}</li>)}</ul>
                {recs.checks.length > 6 && (
                  <Disclosure plain summary={`Show all ${recs.checks.length} checks`}>
                    <ul>{recs.checks.slice(6).map((x) => <li key={x.check}>{x.check}</li>)}</ul>
                  </Disclosure>
                )}
              </div>
              {rca.additional_evidence_required.length > 0 && (
                <div className="stack">
                  <h3>Data needed to confirm the cause</h3>
                  <p className="small muted">Not in the dataset; collect it from the network before acting on a hypothesis.</p>
                  <ul>{rca.additional_evidence_required.map((x) => <li key={x}>{x}</li>)}</ul>
                </div>
              )}
            </div>
          </div>
        </Section>

        {/* ------------------------------------------------------------------ risk */}
        <Section id="risk" title="Fault risk" aside="A model prediction and an observed rate — kept separate">
          <div className="panel risk-pair">
            <div>
              <span className="eyebrow">Predicted · ML model</span>
              {pred.available ? (
                <>
                  <p className="risk-figure">{formatPercent(pred.predicted_fault_rate_pct)} <RiskBadge value={pred.risk_level ?? ""} /></p>
                  <p className="small secondary">
                    80% interval {formatPercent(pred.interval_80pct?.[0])} – {formatPercent(pred.interval_80pct?.[1])} ·
                    input coverage {Math.round((pred.input_coverage ?? 0) * 100)}%
                  </p>
                  {pred.model && <p className="small muted">{pred.model.algorithm}, test R² {formatNumber(pred.model.test_r2, 3)}, MAE {formatNumber(pred.model.test_mae, 2)} points</p>}
                </>
              ) : <p className="secondary">{pred.reason ?? "No prediction available."}</p>}
            </div>
            <div>
              <span className="eyebrow">Observed · similar records</span>
              {hist ? (
                <>
                  <p className="risk-figure">{formatPercent(hist.mean)} <span className="small muted">mean</span></p>
                  <p className="small secondary">Median {formatPercent(hist.median)} · range {formatPercent(hist.min)} – {formatPercent(hist.max)} · {plural(hist.n_records, "record")}</p>
                  <p className="small muted">Historical fault occurrence recorded in the dataset, not a prediction.</p>
                </>
              ) : <p className="secondary">No similar records were retrieved.</p>}
            </div>
          </div>
        </Section>

        {/* ------------------------------------------------------------------ conditions */}
        <Section id="conditions" title="Conditions and labels" aside="Derived from your description and values using documented thresholds">
          <div className="stack-lg">
            <div className="table-wrap">
              <table className="table">
                <caption className="visually-hidden">Conditions observed in this case</caption>
                <thead><tr><th scope="col">Condition</th><th scope="col">Status</th><th scope="col">Source</th><th scope="col" className="num">Value</th><th scope="col">Threshold</th></tr></thead>
                <tbody>
                  {observed.present.map((name) => {
                    const m = observed.measured_evidence.find((x) => x.condition === name);
                    const def = condition(name);
                    return (
                      <tr key={name}>
                        <td>{conditionLabel(name)}</td>
                        <td><Badge tone="warn">Present</Badge></td>
                        <td>{CONDITION_SOURCES[observed.condition_sources[name]] ?? observed.condition_sources[name]}</td>
                        <td className="num">{m ? formatValue(m.value, field(m.metric), true) : <span className="muted">Not supplied</span>}</td>
                        <td className="secondary">{def ? thresholdText(def, field(def.metric)) : "–"}</td>
                      </tr>
                    );
                  })}
                  {observed.absent.map((name) => {
                    const def = condition(name);
                    const v = def ? observed.values[def.metric] : undefined;
                    return (
                      <tr key={name}>
                        <td>{conditionLabel(name)}</td>
                        <td><Badge tone="ok">Not present</Badge></td>
                        <td>{v !== undefined ? "Measured value" : "Stated in the description"}</td>
                        <td className="num">{v !== undefined ? formatValue(v, field(def!.metric), true) : <span className="muted">Not supplied</span>}</td>
                        <td className="secondary">{def ? thresholdText(def, field(def.metric)) : "–"}</td>
                      </tr>
                    );
                  })}
                  {observed.present.length + observed.absent.length === 0 && (
                    <tr><td colSpan={5} className="muted">No conditions were recognised in the description. Add symptoms or measured values for a sharper analysis.</td></tr>
                  )}
                </tbody>
              </table>
            </div>
            {observed.unknown.length > 0 && <p className="small muted">{plural(observed.unknown.length, "other condition")} could not be assessed because no value or mention was given.</p>}

            <div className="grid-2">
              <div className="stack">
                <h3>Category: {c.category}</h3>
                {c.secondary_categories.length > 0 && <p className="small secondary">Also: {c.secondary_categories.join(", ")}</p>}
                <ul className="small">{c.category_reasons.map((r) => <li key={r}>{r}</li>)}</ul>
                <p className="small muted">
                  {c.ml_category.available
                    ? <>ML classifier: {c.ml_category.predicted_category} ({c.ml_agrees_with_rules ? "agrees" : "disagrees"} with the rules). {c.ml_category.label}</>
                    : <>ML classifier not used: {c.ml_category.reason}</>}
                </p>
              </div>
              <div className="stack">
                <h3>Severity <SeverityBadge value={c.severity} /> and priority <PriorityBadge value={pr.priority} label={pr.priority_label} /></h3>
                <p className="small secondary">Severity basis: {c.severity_basis}. Severity score {formatNumber(c.severity_score, 1)}; priority score {formatNumber(pr.priority_score, 1)}.</p>
                <ul className="small">{[...c.severity_reasons, ...pr.reasons].map((r) => <li key={r}>{r}</li>)}</ul>
              </div>
            </div>
          </div>
        </Section>

        {/* ------------------------------------------------------------------ similar records */}
        <Section id="similar" title={`Similar records (${records.length})`}
          aside={`${plural(result.retrieval.matching_records, "record")} match the filters · ${plural(result.retrieval.candidates, "candidate")} scored`}>
          {records.length ? (
            <div className="stack">
              <p className="small secondary">
                Categories: {distribution(result.evidence.history.category_distribution)} · weather: {distribution(result.evidence.history.weather_distribution)}.
                Select a record ID to see all of its measurements.
              </p>
              <RecordsTable records={records} caption="Similar historical records, most similar first" />
              <p className="small muted">
                Similarity combines semantic ({formatNumber(result.retrieval.weights.semantic, 2)}), keyword ({formatNumber(result.retrieval.weights.keyword, 2)})
                and metric ({formatNumber(result.retrieval.weights.metric, 2)}) scores.
              </p>
            </div>
          ) : <p className="secondary">No records match the filters. Remove a filter and analyze again.</p>}
        </Section>

        {/* ------------------------------------------------------------------ review */}
        <Section id="review" title="Automatic review" aside={`Method: ${val.method} · score ${formatNumber(val.score, 2)}`}>
          <div className="stack-lg">
            <div className="grid-2">
              {checks.map((x) => <Check key={x.label} state={x.pass ? "pass" : "fail"} label={x.label} note={x.note} />)}
            </div>
            {val.issues.length > 0 && (
              <div className="stack">
                <h3>Reviewer notes</h3>
                <ul className="small">{val.issues.map((i) => <li key={i}>{i}</li>)}</ul>
              </div>
            )}
            {!val.recommendation_relevant && val.issues.length === 0 && (
              <p className="small secondary">
                At least one action belongs to a lower-ranked hypothesis and is not tied to a condition observed in this case
                (marked “Not tied to a condition observed in this case” above). Weigh those actions accordingly.
              </p>
            )}
            <p className="small muted">
              LLM review: {val.llm_judge ? "ran" : `${val.llm_judge_info.status}${val.llm_judge_info.reason ? ` — ${readable(val.llm_judge_info.reason)}` : ""}`}.
              {n.attempts.length > 0 && ` LLM explanation attempts: ${n.attempts.map((a) => `#${a.attempt} ${a.status ?? (a.grounding_issues?.length ? "rejected" : "accepted")}`).join(", ")}.`}
            </p>
          </div>
        </Section>

        {/* ------------------------------------------------------------------ provenance */}
        <SourcesSection provenance={result.provenance} />

        {/* ------------------------------------------------------------------ report */}
        <Section id="report" title="Report and trace">
          <div className="stack-lg">
            <div className="row">
              <button type="button" className="btn" onClick={() => download(`netops-report-${result.analysis_id.slice(0, 8)}.txt`, result.report_text, "text/plain")}>
                <Icon name="download" />Download report (.txt)
              </button>
              <button type="button" className="btn" onClick={() => download(`netops-analysis-${result.analysis_id.slice(0, 8)}.json`, JSON.stringify(result, null, 2), "application/json")}>
                <Icon name="download" />Download full result (.json)
              </button>
            </div>
            <Disclosure summary="Full text report"><pre className="report">{result.report_text}</pre></Disclosure>
            <Disclosure summary={`Agent trace · ${formatNumber(result.timing_s, 2)} s total`}>
              <div className="table-wrap">
                <table className="table">
                  <thead><tr><th scope="col">Agent</th><th scope="col">Status</th><th scope="col" className="num">Time (s)</th><th scope="col">Summary</th></tr></thead>
                  <tbody>{result.agent_trace.map((t) => (
                    <tr key={t.agent}><td>{t.agent}</td><td>{t.status}</td><td className="num">{formatNumber(t.duration_s, 3)}</td><td className="secondary">{t.summary}</td></tr>
                  ))}</tbody>
                </table>
              </div>
            </Disclosure>
            <div className="stack">
              <h3>Limitations</h3>
              <ul className="small secondary">{result.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
            </div>
          </div>
        </Section>

        <FeedbackPanel result={result} />
      </div>

      {/* -------------------------------------------------------------------- summary rail */}
      <aside className="rail" aria-label="Case summary">
        <div className="panel verdict">
          <span className="eyebrow">Assessment</span>
          <div>
            <p className="verdict__category">{c.category}</p>
            {c.secondary_categories.length > 0 && <p className="small muted">Also {c.secondary_categories.join(", ")}</p>}
          </div>
          <dl className="facts">
            <dt>Severity</dt><dd><SeverityBadge value={c.severity} /></dd>
            <dt>Priority</dt><dd><PriorityBadge value={pr.priority} label={pr.priority_label} /></dd>
            <dt>Predicted fault risk</dt><dd>{pred.available ? <>{formatPercent(pred.predicted_fault_rate_pct)} <RiskBadge value={pred.risk_level ?? ""} /></> : "–"}</dd>
            <dt>Observed in similar</dt><dd>{hist ? formatPercent(hist.mean) : "–"}</dd>
            <dt>Review</dt>
            <dd><a href="#review">{passed} of {checks.length} checks passed</a></dd>
          </dl>
          <p className="small muted">Labels are derived from the evidence; severity basis: {c.severity_basis}.</p>
        </div>
        <nav className="panel" aria-label="Sections of this analysis">
          <ul className="jump">
            {sections.map(([id, label]) => <li key={id}><a href={`#${id}`}>{label}</a></li>)}
          </ul>
        </nav>
      </aside>
    </div>
  );
}

function SourcesSection({ provenance }: { provenance: AnalyzeResponse["provenance"] }) {
  const [active, setActive] = useState<string | null>(null);
  const counts = provenance.reduce<Record<string, number>>((acc, p) => ({ ...acc, [p.source_type]: (acc[p.source_type] ?? 0) + 1 }), {});
  const rows = active ? provenance.filter((p) => p.source_type === active) : provenance;
  return (
    <Section id="sources" title="Sources" aside="Where each statement in this analysis comes from">
      <div className="stack">
        <div className="row" role="group" aria-label="Filter statements by source">
          <button type="button" className="btn btn--sm" aria-pressed={active === null} onClick={() => setActive(null)}
            style={active === null ? { borderColor: "var(--accent)", color: "var(--accent)" } : undefined}>
            All ({provenance.length})
          </button>
          {Object.entries(counts).map(([type, count]) => (
            <button key={type} type="button" className="btn btn--sm" aria-pressed={active === type} onClick={() => setActive(active === type ? null : type)}
              style={active === type ? { borderColor: "var(--accent)", color: "var(--accent)" } : undefined}>
              {SOURCE_LABELS[type] ?? type} ({count})
            </button>
          ))}
        </div>
        <div className="table-wrap">
          <table className="table">
            <caption className="visually-hidden">Statements and their sources</caption>
            <thead><tr><th scope="col">Statement</th><th scope="col">Source</th></tr></thead>
            <tbody>{rows.map((p, i) => <tr key={i}><td>{p.claim}</td><td className="nowrap">{SOURCE_LABELS[p.source_type] ?? p.source_type}</td></tr>)}</tbody>
          </table>
        </div>
      </div>
    </Section>
  );
}

function FeedbackPanel({ result }: { result: AnalyzeResponse }) {
  const [sentFor, setSentFor] = useSessionState<string | null>("analyze.feedbackFor", null);
  const [comment, setComment] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<RequestError | null>(null);
  const done = sentFor === result.analysis_id;

  async function send(rating: "useful" | "not_useful", e?: FormEvent) {
    e?.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.feedback({
        analysis_id: result.analysis_id, query: result.query,
        retrieved_record_ids: result.retrieval.results.map((x) => x.record_id),
        diagnosis: { primary_hypothesis: result.root_cause.primary_hypothesis, confidence: result.root_cause.confidence },
        recommendations: result.recommendations.actions.map((a) => a.action),
        risk_prediction: { predicted_fault_rate_pct: result.risk.predicted.predicted_fault_rate_pct, risk_level: result.risk.predicted.risk_level },
        rating, comment: comment.trim() || null,
      });
      setSentFor(result.analysis_id);
    } catch (err) {
      setError(err as RequestError);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="section" aria-labelledby="feedback-title">
      <div className="panel">
        <div className="panel__body stack">
          <h2 id="feedback-title">Was this analysis useful?</h2>
          {done ? (
            <Notice tone="ok" role="status" title="Thank you — your feedback is stored for evaluation review." />
          ) : (
            <>
              <p className="small muted">Ratings are stored with the retrieved record IDs and the diagnosis, and used to review quality.</p>
              <form className="feedback" onSubmit={(e) => send("useful", e)}>
                <TextField label="Comment (optional)" value={comment} onChange={setComment} maxLength={2000}
                  placeholder="What was right or wrong?" />
                <div className="row">
                  <button type="submit" className="btn" disabled={busy}>Useful</button>
                  <button type="button" className="btn" disabled={busy} onClick={() => send("not_useful")}>Not useful</button>
                </div>
              </form>
              <ErrorNotice error={error} />
            </>
          )}
        </div>
      </div>
    </section>
  );
}
