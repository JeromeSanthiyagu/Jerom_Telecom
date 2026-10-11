import { useRef, useState, type FormEvent } from "react";
import { api, RequestError } from "../lib/api";
import { buildFilters, buildMetrics, countFilters, EMPTY_FILTERS, hasInvalidMetric, type FilterState } from "../lib/format";
import { useAppData, useSessionState } from "../lib/store";
import type { AnalyzeResponse } from "../lib/types";
import { FilterFields, MetricFields } from "../components/domain";
import { Disclosure, ErrorNotice, PageHeader, SelectField, Spinner, Switch } from "../components/ui";
import { AnalysisResults } from "./AnalysisResults";

export const EXAMPLES: { label: string; query: string }[] = [
  { label: "Faults rising, handover falling", query: "A network region has increasing fault occurrence and declining handover success. What corrective actions should be considered?" },
  { label: "Latency and loss, good signal", query: "Cell has high latency and packet loss with good signal strength and normal user load. What similar network conditions occurred before?" },
  { label: "Weak signal and drops in snow", query: "Poor signal strength and frequent call drops during snow" },
  { label: "Load up, throughput down", query: "High user load with reduced throughput and rising latency" },
  { label: "Authentication failures", query: "Many authentication failures and security incidents on a site" },
];

const BASIC_FILTERS: FilterState = { state: "", city: "", season: "", weather: "" };

export function AnalyzePage() {
  const { field } = useAppData();
  const [query, setQuery] = useSessionState("analyze.query", "");
  const [filters, setFilters] = useSessionState<FilterState>("analyze.filters", BASIC_FILTERS);
  const [metrics, setMetrics] = useSessionState<Record<string, string>>("analyze.metrics", {});
  const [topK, setTopK] = useSessionState("analyze.topK", "5");
  const [useLlm, setUseLlm] = useSessionState("analyze.useLlm", true);
  const [result, setResult] = useSessionState<AnalyzeResponse | null>("analyze.result", null);
  const [ranWithLlm, setRanWithLlm] = useSessionState("analyze.ranWithLlm", false);
  const [error, setError] = useState<RequestError | null>(null);
  const [loading, setLoading] = useState(false);
  const [queryError, setQueryError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  const filterCount = countFilters({ ...EMPTY_FILTERS, ...filters });
  const metricCount = Object.keys(buildMetrics(metrics) ?? {}).length;
  const invalidMetrics = hasInvalidMetric(metrics, field);

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!query.trim()) {
      setQueryError("Describe what you are seeing, for example “high latency and packet loss during rain”.");
      return;
    }
    if (invalidMetrics) return;
    setQueryError(null);
    setError(null);
    setLoading(true);
    abort.current = new AbortController();
    try {
      const res = await api.analyze({
        query: query.trim(), filters: buildFilters({ ...EMPTY_FILTERS, ...filters }), top_k: Number(topK),
        use_llm: useLlm, metrics: buildMetrics(metrics),
      }, abort.current.signal);
      setResult(res);
      setRanWithLlm(useLlm);
      requestAnimationFrame(() => document.getElementById("results")?.focus());
    } catch (err) {
      if ((err as Error).name !== "AbortError") setError(err as RequestError);
    } finally {
      setLoading(false);
      abort.current = null;
    }
  }

  function reset() {
    setQuery("");
    setFilters(BASIC_FILTERS);
    setMetrics({});
    setResult(null);
    setError(null);
    setQueryError(null);
  }

  return (
    <>
      <PageHeader title="Analyze a case">
        Describe what you are seeing on the network. The assistant retrieves similar historical records, classifies the
        case, ranks likely causes and suggests what to check — every statement is linked to its source.
      </PageHeader>

      <form className="panel composer" onSubmit={submit} noValidate aria-busy={loading}>
        <div className="panel__body">
          <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-end", marginBottom: "var(--space-2)" }}>
            <label htmlFor="case-query" className="field" style={{ fontWeight: 600, fontSize: "0.875rem" }}>Case description</label>
            <SelectField label="Start from an example" value="" className="example-picker"
              onChange={(v) => v && (setQuery(EXAMPLES[Number(v)].query), setQueryError(null))}
              options={[{ value: "", label: "Choose an example…" }, ...EXAMPLES.map((x, i) => ({ value: String(i), label: x.label }))]} />
          </div>
          <textarea id="case-query" className="textarea" rows={3} value={query} maxLength={5000}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. High latency and packet loss on a cell during rain, signal looks normal. What should we check?"
            aria-invalid={queryError ? true : undefined} aria-describedby="case-query-desc" />
          <span id="case-query-desc" className={queryError ? "small" : "small muted"} style={queryError ? { color: "var(--crit)" } : undefined}>
            {queryError ?? "Plain language. Mention symptoms, conditions such as weather or load, and what you want to know."}
          </span>

          <div className="composer__extras">
            <Disclosure summary={<>Limit similar records by location, season or weather {filterCount > 0 && <span className="badge badge--info">{filterCount} active</span>}</>}>
              <FilterFields value={{ ...EMPTY_FILTERS, ...filters }} onChange={(f) => setFilters({ state: f.state, city: f.city, season: f.season, weather: f.weather })} />
            </Disclosure>
            <Disclosure summary={<>Add measured values <span className="muted">(optional)</span> {metricCount > 0 && <span className="badge badge--info">{metricCount} entered</span>}
              {invalidMetrics && <span className="badge badge--crit">Check values</span>}</>}>
              <p className="small muted" style={{ marginBottom: "var(--space-4)" }}>
                Values you enter are treated as measurements of this case: they decide which conditions are present and steer the search.
              </p>
              <MetricFields values={metrics} onChange={(n, v) => setMetrics({ ...metrics, [n]: v })} />
            </Disclosure>
          </div>

          <div className="composer__footer">
            <div className="composer__options">
              <SelectField label="Similar records to retrieve" value={topK} onChange={setTopK}
                options={Array.from({ length: 10 }, (_, i) => ({ value: String(i + 1), label: String(i + 1) }))} />
              <Switch label="Use the LLM for the explanation and review" checked={useLlm} onChange={setUseLlm}
                hint="If the LLM is unavailable, a fixed-template explanation and rule-based checks are used." />
            </div>
            <div className="row">
              {loading
                ? <button type="button" className="btn" onClick={() => abort.current?.abort()}>Cancel</button>
                : (query || result) && <button type="button" className="btn btn--ghost" onClick={reset}>Clear</button>}
              <button type="submit" className="btn btn--primary" disabled={loading || invalidMetrics}>
                {loading ? <><Spinner />Analyzing…</> : "Analyze case"}
              </button>
            </div>
          </div>
          {loading && (
            <p className="small muted" role="status" style={{ marginTop: "var(--space-3)" }}>
              Retrieving similar records and running the analysis agents{useLlm ? ". With the LLM enabled this can take up to a minute." : "."}
            </p>
          )}
        </div>
      </form>

      {error && <div style={{ marginTop: "var(--space-4)" }}><ErrorNotice error={error} /></div>}

      {result
        ? <AnalysisResults result={result} ranWithLlm={ranWithLlm} />
        : !loading && !error && (
          <p className="muted" style={{ marginTop: "var(--space-5)" }}>
            Results appear here: the case assessment first, then the diagnosis, recommended actions, the evidence behind them
            and the automatic review.
          </p>
        )}
    </>
  );
}
