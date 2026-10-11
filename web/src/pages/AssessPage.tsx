import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { api, RequestError, type Metrics } from "../lib/api";
import { buildMetrics, formatNumber, formatPercent, formatValue, hasInvalidMetric, plural, thresholdText } from "../lib/format";
import { useAppData, useSessionState, writeSession } from "../lib/store";
import type { ClassifyResponse, RiskResponse } from "../lib/types";
import { MetricFields } from "../components/domain";
import { EmptyState, ErrorNotice, Notice, PageHeader, PriorityBadge, RiskBadge, SelectField, SeverityBadge, Spinner } from "../components/ui";

interface Context { season: string; weather: string; issue: string }
const NO_CONTEXT: Context = { season: "", weather: "", issue: "" };

export function AssessPage() {
  const { reference, field, condition, conditionLabel } = useAppData();
  const navigate = useNavigate();
  const [values, setValues] = useSessionState<Record<string, string>>("assess.metrics", {});
  const [context, setContext] = useSessionState<Context>("assess.context", NO_CONTEXT);
  const [result, setResult] = useSessionState<{ cls: ClassifyResponse; risk: RiskResponse } | null>("assess.result", null);
  const [error, setError] = useState<RequestError | null>(null);
  const [formError, setFormError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const invalid = hasInvalidMetric(values, field);

  function payload(): Metrics | null {
    const m: Metrics = { ...(buildMetrics(values) ?? {}) };
    if (context.season) m.season = context.season;
    if (context.weather) m.weather = context.weather;
    if (context.issue) m.issue_reported = context.issue === "yes";
    return Object.keys(m).length ? m : null;
  }

  async function submit(e: FormEvent) {
    e.preventDefault();
    const metrics = payload();
    if (!metrics) {
      setFormError("Enter at least one measurement.");
      return;
    }
    if (invalid) return;
    setFormError(null);
    setError(null);
    setLoading(true);
    try {
      const [cls, risk] = await Promise.all([api.classify({ metrics }), api.risk({ metrics })]);
      setResult({ cls, risk });
    } catch (err) {
      setError(err as RequestError);
    } finally {
      setLoading(false);
    }
  }

  function analyzeWithValues() {
    writeSession("analyze.metrics", Object.fromEntries(Object.entries(values).filter(([, v]) => v.trim() !== "")));
    navigate("/");
  }

  const opts = (xs: string[]) => [{ value: "", label: "Not specified" }, ...xs.map((x) => ({ value: x, label: x }))];
  const cls = result?.cls;
  const pred = result?.risk.predicted;

  return (
    <>
      <PageHeader title="Assess measurements">
        Enter measurements for one cell to see which conditions they trigger, the derived category, severity and
        priority, and the predicted fault risk. No similar-record search or explanation is run here.
      </PageHeader>

      <div className="split" style={{ gridTemplateColumns: "minmax(0, 1.25fr) minmax(0, 1fr)" }}>
        <form className="panel" onSubmit={submit} noValidate aria-label="Measurements">
          <div className="panel__body stack-lg">
            <fieldset className="fieldset">
              <legend>Context</legend>
              <div className="grid-form">
                <SelectField label="Season" value={context.season} onChange={(v) => setContext({ ...context, season: v })} options={opts(reference?.seasons ?? [])} />
                <SelectField label="Weather" value={context.weather} onChange={(v) => setContext({ ...context, weather: v })} options={opts(reference?.weather ?? [])} />
                <SelectField label="Customer issue reported" value={context.issue} onChange={(v) => setContext({ ...context, issue: v })}
                  options={[{ value: "", label: "Not specified" }, { value: "yes", label: "Yes" }, { value: "no", label: "No" }]} />
              </div>
            </fieldset>
            <MetricFields values={values} onChange={(n, v) => setValues({ ...values, [n]: v })} />
            {formError && <p className="small" role="alert" style={{ color: "var(--crit)" }}>{formError}</p>}
            <div className="row" style={{ justifyContent: "flex-end" }}>
              <button type="button" className="btn btn--ghost" onClick={() => { setValues({}); setContext(NO_CONTEXT); setResult(null); }}>Clear all</button>
              <button type="submit" className="btn btn--primary" disabled={loading || invalid}>
                {loading ? <><Spinner />Assessing…</> : "Assess"}
              </button>
            </div>
          </div>
        </form>

        <div className="split__aside stack" aria-live="polite">
          <ErrorNotice error={error} />
          {cls && pred ? (
            <>
              <div className="panel verdict">
                <span className="eyebrow">Assessment of your values</span>
                <p className="verdict__category">{cls.classification.category}</p>
                {cls.classification.secondary_categories.length > 0 && <p className="small muted">Also {cls.classification.secondary_categories.join(", ")}</p>}
                <dl className="facts">
                  <dt>Severity</dt><dd><SeverityBadge value={cls.classification.severity} /> <span className="small muted">score {formatNumber(cls.classification.severity_score, 1)}</span></dd>
                  <dt>Priority</dt><dd><PriorityBadge value={cls.priority.priority} label={cls.priority.priority_label} /></dd>
                  <dt>Predicted fault risk</dt>
                  <dd>{pred.available ? <>{formatPercent(pred.predicted_fault_rate_pct)} <RiskBadge value={pred.risk_level ?? ""} /></> : "–"}</dd>
                </dl>
                {pred.available && (
                  <p className="small muted">
                    ML prediction, 80% interval {formatPercent(pred.interval_80pct?.[0])} – {formatPercent(pred.interval_80pct?.[1])};
                    model inputs covered {Math.round((pred.input_coverage ?? 0) * 100)}%.
                  </p>
                )}
                <button type="button" className="btn" onClick={analyzeWithValues}>Analyze a case with these values</button>
              </div>
              {pred.low_input_coverage && pred.caveat && <Notice tone="info" title="Few model inputs supplied">{pred.caveat}</Notice>}

              <div className="panel">
                <div className="panel__head"><h2>Conditions triggered</h2><span className="small muted">{plural(cls.conditions.conditions.length, "condition")}</span></div>
                <div className="panel__body stack">
                  {cls.conditions.evidence.length ? (
                    <ul>
                      {cls.conditions.evidence.map((ev) => {
                        const def = condition(ev.condition);
                        return (
                          <li key={ev.condition}>
                            <strong>{conditionLabel(ev.condition)}</strong>: {formatValue(ev.value, field(ev.metric), true)}
                            <span className="small muted"> — flagged {def ? thresholdText(def, field(def.metric)) : `${ev.operator} ${ev.threshold}`}</span>
                          </li>
                        );
                      })}
                    </ul>
                  ) : <p className="secondary">None of the entered values cross a condition threshold.</p>}
                  <p className="small muted">
                    {plural(cls.conditions.unavailable.length, "condition")} not assessed because the measurement was not entered.
                  </p>
                </div>
              </div>

              <div className="panel">
                <div className="panel__head"><h2>Why these labels</h2></div>
                <div className="panel__body">
                  <ul className="small">
                    {[...cls.classification.category_reasons, ...cls.classification.severity_reasons, ...cls.priority.reasons].map((r) => <li key={r}>{r}</li>)}
                  </ul>
                </div>
              </div>
            </>
          ) : !error && (
            <EmptyState title="No assessment yet">
              Enter the measurements you have — any subset works — and press Assess. Hints under each field show the
              threshold at which it is flagged.
            </EmptyState>
          )}
        </div>
      </div>
    </>
  );
}
