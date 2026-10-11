import { useId, type InputHTMLAttributes, type ReactNode } from "react";
import type { RequestError } from "../lib/api";
import type { ApiError } from "../lib/types";

// ---------------------------------------------------------------------------- icons (inline, decorative)
const PATHS = {
  info: "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm0 4.5a1.25 1.25 0 1 1 0 2.5 1.25 1.25 0 0 1 0-2.5ZM13.5 17h-3v-1.5h.75v-4.5h-.75V9.5h2.25v6h.75V17Z",
  warn: "M12 2.5 1.5 21h21L12 2.5Zm-1 6.5h2v6h-2V9Zm0 7.5h2v2h-2v-2Z",
  error: "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm3.54 12.12-1.42 1.42L12 13.41l-2.12 2.13-1.42-1.42L10.59 12 8.46 9.88l1.42-1.42L12 10.59l2.12-2.13 1.42 1.42L13.41 12l2.13 2.12Z",
  pass: "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm-1.5 14.5-4-4 1.41-1.41 2.59 2.58 5.59-5.58L17.5 9.5l-7 7Z",
  minus: "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20Zm5 11H7v-2h10v2Z",
  chevron: "M8.6 5.4 15.2 12l-6.6 6.6L7.2 17.2 12.4 12 7.2 6.8z",
  download: "M5 20h14v-2H5v2Zm14-11h-4V3H9v6H5l7 7 7-7Z",
  open: "M14 3v2h3.59l-9.83 9.83 1.41 1.41L19 6.41V10h2V3h-7Zm5 16H5V5h7V3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7h-2v7Z",
  up: "M2 20h20L12 4 2 20Z",
} as const;

export type IconName = keyof typeof PATHS;

export function Icon({ name, className }: { name: IconName; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" focusable="false" className={className}>
      <path d={PATHS[name]} />
    </svg>
  );
}

export function Spinner({ label }: { label?: string }) {
  return <span className="spinner" role={label ? "status" : undefined} aria-label={label} />;
}

// ---------------------------------------------------------------------------- layout
export function PageHeader({ title, children }: { title: string; children: ReactNode }) {
  return (
    <header className="page-header">
      <h1>{title}</h1>
      <p>{children}</p>
    </header>
  );
}

export function Section({ id, title, aside, children }: { id: string; title: string; aside?: ReactNode; children: ReactNode }) {
  return (
    <section className="section" id={id} aria-labelledby={`${id}-title`}>
      <div className="section__head">
        <h2 id={`${id}-title`}>{title}</h2>
        {aside && <p>{aside}</p>}
      </div>
      {children}
    </section>
  );
}

export function EmptyState({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="empty">
      <h2>{title}</h2>
      <p>{children}</p>
    </div>
  );
}

export function Disclosure({ summary, children, open, plain }: { summary: ReactNode; children: ReactNode; open?: boolean; plain?: boolean }) {
  return (
    <details className={`disclosure${plain ? " disclosure--plain" : ""}`} open={open}>
      <summary>{summary}</summary>
      <div className="disclosure__body">{children}</div>
    </details>
  );
}

// ---------------------------------------------------------------------------- status
type Tone = "crit" | "crit-solid" | "warn" | "ok" | "info" | "neutral";

export function Badge({ tone, children, title }: { tone: Tone; children: ReactNode; title?: string }) {
  return <span className={`badge badge--${tone}`} title={title}>{children}</span>;
}

const SEVERITY_TONE: Record<string, Tone> = { LOW: "neutral", MEDIUM: "warn", HIGH: "crit", CRITICAL: "crit-solid" };
const PRIORITY_TONE: Record<string, Tone> = { P1: "crit-solid", P2: "crit", P3: "warn", P4: "neutral" };
const RISK_TONE: Record<string, Tone> = { Low: "neutral", Medium: "warn", High: "crit", Critical: "crit-solid" };

export const SeverityBadge = ({ value }: { value: string }) => <Badge tone={SEVERITY_TONE[value] ?? "neutral"}>{value}</Badge>;
export const PriorityBadge = ({ value, label }: { value: string; label?: string }) => (
  <Badge tone={PRIORITY_TONE[value] ?? "neutral"}>{label ? `${value} · ${label}` : value}</Badge>
);
export const RiskBadge = ({ value }: { value: string }) => <Badge tone={RISK_TONE[value] ?? "neutral"}>{value}</Badge>;

export function Notice({ tone, title, children, role }: { tone: "crit" | "warn" | "info" | "ok"; title?: ReactNode; children?: ReactNode; role?: "alert" | "status" }) {
  const icon: IconName = tone === "crit" ? "error" : tone === "warn" ? "warn" : tone === "ok" ? "pass" : "info";
  return (
    <div className={`notice notice--${tone}`} role={role}>
      <Icon name={icon} />
      <div className="notice__body">
        {title && <strong>{title}</strong>}
        {children && <div className="secondary">{children}</div>}
      </div>
    </div>
  );
}

const ERROR_HINTS: Record<string, string> = {
  connection_error: "Make sure the API server is running, then try again.",
  invalid_filter: "Adjust the filters and try again.",
  validation_error: "One of the values is outside its allowed range. Correct it and try again.",
  not_found: "Check the record ID. IDs look like ROW-000123.",
  agent_failure: "An analysis step failed on the server. Try again; if it repeats, check the API log.",
};

export function ErrorNotice({ error }: { error: RequestError | ApiError | null }) {
  if (!error) return null;
  const e: ApiError = "error" in error ? error.error : error;
  const hint = e.code.startsWith("guardrail")
    ? "Describe a network observation or question. Requests unrelated to network operations are not processed."
    : ERROR_HINTS[e.code];
  return (
    <Notice tone="crit" role="alert" title={e.message}>
      {hint} <span className="muted small">({e.code})</span>
    </Notice>
  );
}

export function Check({ state, label, note }: { state: "pass" | "fail" | "skip"; label: ReactNode; note?: ReactNode }) {
  const word = state === "pass" ? "Passed" : state === "fail" ? "Failed" : "Not run";
  return (
    <div className={`check check--${state}`}>
      <Icon name={state === "pass" ? "pass" : state === "fail" ? "error" : "minus"} />
      <div>
        <div><strong>{word}</strong> — {label}</div>
        {note && <div className="small muted">{note}</div>}
      </div>
    </div>
  );
}

export function Meter({ value, label }: { value: number; label: string }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <span className="meter">
      <span className="meter__track" aria-hidden="true"><span className="meter__fill" style={{ width: `${pct}%`, display: "block" }} /></span>
      <span>{label}</span>
    </span>
  );
}

// ---------------------------------------------------------------------------- form fields
interface BaseFieldProps { label: string; hint?: ReactNode; error?: string | null; className?: string }

export function TextField({ label, hint, error, className, value, onChange, ...rest }:
  BaseFieldProps & { value: string; onChange: (v: string) => void } & Omit<InputHTMLAttributes<HTMLInputElement>, "value" | "onChange">) {
  const id = useId();
  return (
    <div className={`field ${className ?? ""}`}>
      <label htmlFor={id}>{label}</label>
      <input id={id} className="input" value={value} onChange={(e) => onChange(e.target.value)}
        aria-invalid={error ? true : undefined} aria-describedby={hint || error ? `${id}-desc` : undefined} {...rest} />
      {(error || hint) && <span id={`${id}-desc`} className={error ? "error-text" : "hint"}>{error ?? hint}</span>}
    </div>
  );
}

export function SelectField({ label, hint, value, onChange, options, className }:
  BaseFieldProps & { value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }) {
  const id = useId();
  return (
    <div className={`field ${className ?? ""}`}>
      <label htmlFor={id}>{label}</label>
      <select id={id} className="select" value={value} onChange={(e) => onChange(e.target.value)}
        aria-describedby={hint ? `${id}-desc` : undefined}>
        {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
      {hint && <span id={`${id}-desc`} className="hint">{hint}</span>}
    </div>
  );
}

export function Switch({ label, checked, onChange, hint }: { label: string; checked: boolean; onChange: (v: boolean) => void; hint?: string }) {
  const id = useId();
  return (
    <span className="field">
      <label className="switch" htmlFor={id}>
        <input id={id} type="checkbox" role="switch" checked={checked} onChange={(e) => onChange(e.target.checked)}
          aria-describedby={hint ? `${id}-desc` : undefined} />
        {label}
      </label>
      {hint && <span id={`${id}-desc`} className="hint">{hint}</span>}
    </span>
  );
}
