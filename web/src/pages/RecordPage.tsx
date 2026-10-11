import { useEffect, useState, type FormEvent } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, RequestError } from "../lib/api";
import { formatPercent } from "../lib/format";
import type { ClassifyResponse, NetworkRecord, RiskResponse } from "../lib/types";
import { RecordDetail } from "../components/domain";
import { EmptyState, ErrorNotice, PageHeader, RiskBadge, Spinner, TextField } from "../components/ui";

const ID_PATTERN = /^ROW-\d{6}$/;

function normalise(raw: string): string {
  const v = raw.trim().toUpperCase();
  if (/^\d{1,6}$/.test(v)) return `ROW-${v.padStart(6, "0")}`;
  return v;
}

export function RecordPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [input, setInput] = useState(id ?? "");
  const [inputError, setInputError] = useState<string | null>(null);
  const [record, setRecord] = useState<NetworkRecord | null>(null);
  const [risk, setRisk] = useState<RiskResponse | null>(null);
  const [cls, setCls] = useState<ClassifyResponse | null>(null);
  const [error, setError] = useState<RequestError | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setInput(id ?? "");
    setRecord(null);
    setRisk(null);
    setCls(null);
    setError(null);
    if (!id) return;
    let live = true;
    setLoading(true);
    api.record(id)
      .then(async (rec) => {
        if (!live) return;
        setRecord(rec);
        // Model checks are secondary; a failure here should not hide the record.
        const [r, c] = await Promise.allSettled([api.risk({ record_id: id }), api.classify({ record_id: id })]);
        if (!live) return;
        if (r.status === "fulfilled") setRisk(r.value);
        if (c.status === "fulfilled") setCls(c.value);
      })
      .catch((e) => live && setError(e as RequestError))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
  }, [id]);

  function submit(e: FormEvent) {
    e.preventDefault();
    const v = normalise(input);
    if (!ID_PATTERN.test(v)) {
      setInputError("Use the format ROW-000123 (or just the number).");
      return;
    }
    setInputError(null);
    navigate(`/record/${v}`);
  }

  const ml = cls?.classification.ml_category;
  return (
    <>
      <PageHeader title="Record details">
        Look up one historical record to see its raw measurements next to the labels derived from them, plus what the
        models say about it.
      </PageHeader>

      <form className="lookup" onSubmit={submit} noValidate role="search" aria-label="Look up a record">
        <TextField label="Record ID" value={input} onChange={setInput} placeholder="ROW-000123" error={inputError}
          autoComplete="off" spellCheck={false} />
        <button type="submit" className="btn btn--primary">Look up</button>
      </form>

      <div style={{ marginTop: "var(--space-5)" }} aria-live="polite">
        {loading && !record && <p className="row muted"><Spinner label="Loading record" /> Loading {id}…</p>}
        <ErrorNotice error={error} />
        {!id && (
          <EmptyState title="No record selected">
            Enter a record ID above, or open one from the similar-records table on the Analyze or Search page.
          </EmptyState>
        )}
        {record && (
          <div className="stack-lg">
            <RecordDetail record={record} />
            <section className="section" aria-labelledby="model-checks">
              <div className="section__head">
                <h2 id="model-checks">Model checks</h2>
                <p>Predictions for this record, shown next to what the dataset recorded</p>
              </div>
              {loading && !risk ? <p className="row muted"><Spinner label="Running model checks" /> Running model checks…</p> : (
                <div className="panel risk-pair">
                  <div>
                    <span className="eyebrow">Fault occurrence</span>
                    {risk?.predicted.available ? (
                      <>
                        <p className="risk-figure">{formatPercent(risk.predicted.predicted_fault_rate_pct)} <RiskBadge value={risk.predicted.risk_level ?? ""} /></p>
                        <p className="small secondary">
                          ML prediction · 80% interval {formatPercent(risk.predicted.interval_80pct?.[0])} – {formatPercent(risk.predicted.interval_80pct?.[1])}
                        </p>
                        {risk.historical_fault_rate_pct && (
                          <p className="small secondary">Recorded in the dataset: <strong>{formatPercent(risk.historical_fault_rate_pct.value)}</strong> (observed)</p>
                        )}
                      </>
                    ) : <p className="secondary">{risk?.predicted.reason ?? "Prediction unavailable."}</p>}
                  </div>
                  <div>
                    <span className="eyebrow">Category · ML classifier</span>
                    {ml?.available ? (
                      <>
                        <p className="risk-figure" style={{ fontSize: "1.125rem" }}>{ml.predicted_category}</p>
                        <p className="small secondary">
                          Probability {formatPercent((ml.probability ?? 0) * 100, 1)} ·{" "}
                          {cls?.classification.ml_agrees_with_rules ? "agrees with" : "differs from"} the rule-based category ({cls?.classification.category})
                        </p>
                        {ml.label && <p className="small muted">{ml.label}</p>}
                      </>
                    ) : <p className="secondary">{ml?.reason ?? "Classifier unavailable."}</p>}
                  </div>
                </div>
              )}
            </section>
          </div>
        )}
      </div>
    </>
  );
}
