import { Fragment, useState } from "react";
import { Link } from "react-router-dom";
import { formatNumber, formatValue, metricError, type FilterState } from "../lib/format";
import { useAppData } from "../lib/store";
import type { NetworkRecord } from "../lib/types";
import { Icon, PriorityBadge, SelectField, SeverityBadge, TextField } from "./ui";

export const ROLE_GROUPS: [string, string][] = [
  ["kpi", "Radio and service KPIs"],
  ["load", "Load"],
  ["operational", "Alarms and configuration"],
  ["security", "Security"],
  ["environment", "Environment"],
];
const OUTCOMES = ["fault_rate_pct", "mttr_hours"];
const SHORT_OP: Record<string, string> = { "<": "below", ">": "above", "<=": "at or below", ">=": "at or above" };

// ---------------------------------------------------------------------------- filters
export function FilterFields({ value, onChange, extended = false }: { value: FilterState; onChange: (f: FilterState) => void; extended?: boolean }) {
  const { reference, conditionLabel } = useAppData();
  const set = (patch: Partial<FilterState>) => onChange({ ...value, ...patch });
  const any = { value: "", label: "Any" };
  const opts = (xs: string[]) => [any, ...xs.map((x) => ({ value: x, label: x }))];
  const states = Object.entries(reference?.states ?? {}).sort((a, b) => a[1].localeCompare(b[1]))
    .map(([code, name]) => ({ value: code, label: `${name} (${code})` }));

  return (
    <div className="stack">
      <div className="grid-form">
        <SelectField label="State or territory" value={value.state} onChange={(v) => set({ state: v })} options={[any, ...states]} />
        <TextField label="City" value={value.city} onChange={(v) => set({ city: v })} placeholder="Exact name"
          hint="Must match a dataset city; case is ignored." />
        <SelectField label="Season" value={value.season} onChange={(v) => set({ season: v })} options={opts(reference?.seasons ?? [])} />
        <SelectField label="Weather" value={value.weather} onChange={(v) => set({ weather: v })} options={opts(reference?.weather ?? [])} />
        {extended && (
          <>
            <SelectField label="Category" value={value.category ?? ""} onChange={(v) => set({ category: v })}
              options={opts(reference?.categories ?? [])} hint="Derived category stored with each record." />
            <SelectField label="Severity" value={value.severity ?? ""} onChange={(v) => set({ severity: v })} options={opts(reference?.severities ?? [])} />
            <SelectField label="Priority" value={value.priority ?? ""} onChange={(v) => set({ priority: v })} options={opts(reference?.priorities ?? [])} />
            <SelectField label="Customer issue reported" value={value.issue ?? ""} onChange={(v) => set({ issue: v })}
              options={[any, { value: "yes", label: "Yes" }, { value: "no", label: "No" }]} />
          </>
        )}
      </div>
      {extended && (
        <fieldset className="fieldset">
          <legend>Must show all of these conditions</legend>
          <div className="condition-list">
            {(reference?.conditions ?? []).map((c) => {
              const checked = value.conditions?.includes(c.name) ?? false;
              return (
                <label key={c.name} className="checkbox">
                  <input type="checkbox" checked={checked} onChange={() => set({
                    conditions: checked ? value.conditions!.filter((x) => x !== c.name) : [...(value.conditions ?? []), c.name],
                  })} />
                  {conditionLabel(c.name)}
                </label>
              );
            })}
          </div>
        </fieldset>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------- measurements
export function MetricFields({ values, onChange, roles }: { values: Record<string, string>; onChange: (name: string, v: string) => void; roles?: string[] }) {
  const { reference, conditionsFor, field } = useAppData();
  const inputs = (reference?.fields ?? []).filter((f) => f.input && (f.type === "float" || f.type === "int"));
  return (
    <div className="stack-lg">
      {ROLE_GROUPS.filter(([role]) => !roles || roles.includes(role)).map(([role, title]) => {
        const group = inputs.filter((f) => f.role === role);
        if (!group.length) return null;
        return (
          <fieldset className="fieldset" key={role}>
            <legend>{title}</legend>
            <div className="grid-form">
              {group.map((f) => {
                const raw = values[f.name] ?? "";
                const hint = conditionsFor(f.name)
                  .map((c) => `Flagged ${SHORT_OP[c.operator]} ${formatValue(c.threshold, field(c.metric), true)}${c.method === "percentile" ? " (worst 10%)" : ""}`)
                  .join("; ");
                return (
                  <TextField key={f.name} label={f.label} value={raw} inputMode="decimal" autoComplete="off"
                    placeholder="Not measured" onChange={(v) => onChange(f.name, v)}
                    error={metricError(raw, f)} hint={hint || undefined} />
                );
              })}
            </div>
          </fieldset>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------- records table
const COLUMNS: { key: string; label: string; unit: string }[] = [
  { key: "latency_ms", label: "Latency", unit: "ms" },
  { key: "packet_loss_pct", label: "Pkt loss", unit: "%" },
  { key: "handover_success_pct", label: "Handover", unit: "% success" },
  { key: "signal_strength_dbm", label: "Signal", unit: "dBm" },
  { key: "user_count", label: "Users", unit: "count" },
  { key: "fault_rate_pct", label: "Fault rate", unit: "% observed" },
];

export function RecordsTable({ records, withSimilarity = true, caption }: { records: NetworkRecord[]; withSimilarity?: boolean; caption: string }) {
  const { conditionLabel, field } = useAppData();
  const [open, setOpen] = useState<string | null>(null);
  const span = 4 + COLUMNS.length + (withSimilarity ? 1 : 0);
  return (
    <div className="table-wrap">
      <table className="table">
        <caption className="visually-hidden">{caption}</caption>
        <thead>
          <tr>
            <th scope="col">Record</th>
            {withSimilarity && <th scope="col" className="num" title="Hybrid score combining semantic, keyword and metric similarity (0–1)">Similarity<span className="unit">0–1</span></th>}
            <th scope="col">Category<span className="unit">severity</span></th>
            <th scope="col">Conditions</th>
            {COLUMNS.map((c) => <th key={c.key} scope="col" className="num">{c.label}<span className="unit">{c.unit}</span></th>)}
            <th scope="col">Location<span className="unit">season · weather</span></th>
          </tr>
        </thead>
        <tbody>
          {records.map((r) => {
            const isOpen = open === r.record_id;
            return (
              <Fragment key={r.record_id}>
                <tr className={isOpen ? "is-open" : undefined}>
                  <td>
                    <button type="button" className="row-toggle" aria-expanded={isOpen}
                      aria-controls={`detail-${r.record_id}`} onClick={() => setOpen(isOpen ? null : r.record_id)}>
                      <Icon name="chevron" />{r.record_id}
                      <span className="visually-hidden">{isOpen ? "Hide details" : "Show details"}</span>
                    </button>
                  </td>
                  {withSimilarity && <td className="num">{formatNumber(r.hybrid_score, 2)}</td>}
                  <td className="cell-category">{r.category}<br /><SeverityBadge value={r.severity} /></td>
                  <td className="cell-conditions">{r.conditions.length ? r.conditions.map((c) => <span className="tag" key={c}>{conditionLabel(c)}</span>) : <span className="muted">None</span>}</td>
                  {COLUMNS.map((c) => <td key={c.key} className="num">{formatValue(r.metrics[c.key], field(c.key))}</td>)}
                  <td className="nowrap">{r.metadata.city}, {r.metadata.state}<div className="small muted">{r.metadata.season} · {r.metadata.weather}</div></td>
                </tr>
                {isOpen && (
                  <tr className="detail-row" id={`detail-${r.record_id}`}>
                    <td colSpan={span}>
                      <div className="detail-row__inner"><RecordDetail record={r} compact /></div>
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------- record detail
export function RecordDetail({ record, compact = false }: { record: NetworkRecord; compact?: boolean }) {
  const { field, condition, conditionLabel, stateName } = useAppData();
  const md = record.metadata;
  const flags = new Map<string, string[]>();
  for (const name of record.conditions) {
    const c = condition(name);
    if (c) flags.set(c.metric, [...(flags.get(c.metric) ?? []), c.label]);
  }
  const groups = [
    ...ROLE_GROUPS.map(([role, title]) => [title, Object.keys(record.metrics).filter((m) => field(m)?.role === role && !OUTCOMES.includes(m))] as const),
    ["Outcomes (observed)", OUTCOMES.filter((m) => m in record.metrics)] as const,
  ].filter(([, names]) => names.length);

  return (
    <div className="stack-lg">
      <div className={compact ? "row" : "record-head"} style={compact ? { justifyContent: "space-between" } : undefined}>
        <div className="record-head">
          {!compact && <h2>{record.record_id}</h2>}
          <p className="secondary">
            {md.city}, {stateName(md.state)} · ZIP {md.zip} · {md.season} · {md.weather} · {formatValue(md.temperature_c, field("temperature_c"), true)},
            humidity {formatValue(md.humidity_pct, field("humidity_pct"), true)} · customer issue reported: {formatValue(md.issue_reported)}
          </p>
        </div>
        {compact && (
          <Link className="btn btn--sm" to={`/record/${record.record_id}`}>
            <Icon name="open" />Open record page
          </Link>
        )}
      </div>

      <div className="grid-2">
        <div className="stack record-labels">
          <div>
            <h3>Derived labels</h3>
            <p className="small muted">Computed from the raw values during preprocessing — not ground-truth annotations.</p>
          </div>
          <dl className="facts">
            <dt>Category</dt>
            <dd>{record.category}{record.secondary_categories.length > 0 && <span className="muted"> · also {record.secondary_categories.join(", ")}</span>}</dd>
            <dt>Severity</dt>
            <dd><SeverityBadge value={record.severity} /> <span className="muted small">score {formatNumber(record.severity_score, 1)}</span></dd>
            <dt>Priority</dt>
            <dd><PriorityBadge value={record.priority} /></dd>
            <dt>Conditions</dt>
            <dd>{record.conditions.length ? record.conditions.map((c) => <span className="tag" key={c}>{conditionLabel(c)}</span>) : "None"}</dd>
          </dl>
        </div>
        <div className="stack">
          <div>
            <h3>Measurements</h3>
            <p className="small muted">Raw values from the dataset. Flags show which derived condition a value triggered.</p>
          </div>
          <div className="table-wrap">
            <table className="table">
              <caption className="visually-hidden">Measurements for {record.record_id}</caption>
              <thead><tr><th scope="col">Measurement</th><th scope="col" className="num">Value</th><th scope="col">Flag</th></tr></thead>
              <tbody>
                {groups.map(([title, names]) => (
                  <Fragment key={title}>
                    <tr><th scope="rowgroup" colSpan={3} className="eyebrow" style={{ position: "static" }}>{title}</th></tr>
                    {names.map((m) => (
                      <tr key={m}>
                        <td>{field(m)?.label ?? m}</td>
                        <td className="num">{formatValue(record.metrics[m], field(m), true)}</td>
                        <td>{flags.get(m)?.map((l) => <span className="tag" key={l}>{l}</span>)}</td>
                      </tr>
                    ))}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
