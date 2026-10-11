import { useState, type FormEvent } from "react";
import { api, RequestError } from "../lib/api";
import { buildFilters, countFilters, EMPTY_FILTERS, formatNumber, plural, type FilterState } from "../lib/format";
import { useSessionState } from "../lib/store";
import type { SearchResponse } from "../lib/types";
import { FilterFields, RecordsTable } from "../components/domain";
import { EmptyState, ErrorNotice, PageHeader, SelectField, Spinner, TextField } from "../components/ui";

export function SearchPage() {
  const [query, setQuery] = useSessionState("search.query", "");
  const [filters, setFilters] = useSessionState<FilterState>("search.filters", EMPTY_FILTERS);
  const [topK, setTopK] = useSessionState("search.topK", "10");
  const [result, setResult] = useSessionState<SearchResponse | null>("search.result", null);
  const [error, setError] = useState<RequestError | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const active = countFilters(filters);

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!query.trim()) {
      setQueryError("Enter what to look for, for example “weak signal and call drops”.");
      return;
    }
    setQueryError(null);
    setError(null);
    setLoading(true);
    try {
      setResult(await api.search({ query: query.trim(), filters: buildFilters(filters), top_k: Number(topK) }));
    } catch (err) {
      setError(err as RequestError);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <PageHeader title="Search records">
        Find historical records that resemble a description. Filters are applied strictly: a record that does not match
        every filter is never returned.
      </PageHeader>

      <div className="split">
        <form className="panel split__aside" onSubmit={submit} noValidate aria-label="Search and filters">
          <div className="panel__body stack-lg">
            <TextField label="Look for" value={query} onChange={setQuery} placeholder="e.g. high latency and packet loss"
              error={queryError} maxLength={5000} />
            <SelectField label="Results to show" value={topK} onChange={setTopK}
              options={[5, 10, 15, 20].map((n) => ({ value: String(n), label: String(n) }))} />
            <div className="stack">
              <div className="row" style={{ justifyContent: "space-between" }}>
                <h2 style={{ fontSize: "0.875rem" }}>Filters {active > 0 && <span className="badge badge--info">{active} active</span>}</h2>
                {active > 0 && <button type="button" className="btn btn--ghost btn--sm" onClick={() => setFilters(EMPTY_FILTERS)}>Clear filters</button>}
              </div>
              <div className="search-filters">
                <FilterFields value={filters} onChange={setFilters} extended />
              </div>
            </div>
            <button type="submit" className="btn btn--primary" disabled={loading}>
              {loading ? <><Spinner />Searching…</> : "Search"}
            </button>
          </div>
        </form>

        <div className="stack" aria-live="polite">
          <ErrorNotice error={error} />
          {result ? (
            <>
              <div className="section__head">
                <h2>{plural(result.results.length, "result")} for “{result.query}”</h2>
                <p>{plural(result.matching_records, "record")} match the filters · {plural(result.candidates, "candidate")} scored</p>
              </div>
              {result.results.length ? (
                <>
                  <RecordsTable records={result.results} caption={`Search results for ${result.query}`} />
                  <p className="small muted">
                    Most similar first. Similarity combines semantic ({formatNumber(result.weights.semantic, 2)}), keyword
                    ({formatNumber(result.weights.keyword, 2)}) and metric ({formatNumber(result.weights.metric, 2)}) scores.
                    Select a record ID to see its measurements.
                  </p>
                </>
              ) : (
                <EmptyState title="No records match these filters">
                  Remove a filter or a required condition and search again.
                </EmptyState>
              )}
            </>
          ) : !error && (
            <EmptyState title="No search yet">
              Describe a pattern on the left and press Search. Use the filters to restrict results to a place, season,
              category or set of conditions.
            </EmptyState>
          )}
        </div>
      </div>
    </>
  );
}
