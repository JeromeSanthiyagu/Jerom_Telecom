import type { ConditionDef, Field } from "./types";

const DECIMALS: Record<string, number> = { "%": 2, ms: 1, dBm: 1, Mbps: 1, GB: 2, "°C": 1, hours: 2 };
const OPERATOR_WORDS: Record<string, string> = { "<": "below", ">": "above", "<=": "at or below", ">=": "at or above" };

export function unitSuffix(unit: string | null | undefined): string {
  if (!unit) return "";
  return unit === "%" || unit === "°C" ? unit : ` ${unit}`;
}

/** Format a measurement with the precision its unit warrants; integers get thousands separators. */
export function formatValue(value: unknown, field?: Pick<Field, "unit" | "type">, withUnit = false): string {
  if (value === null || value === undefined || (typeof value === "number" && Number.isNaN(value))) return "–";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value !== "number") return String(value);
  const integer = field?.type === "int";
  const digits = integer ? 0 : DECIMALS[field?.unit ?? ""] ?? 2;
  const text = value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  return withUnit ? text + unitSuffix(field?.unit) : text;
}

export function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined) return "–";
  return value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function formatPercent(value: number | null | undefined, digits = 2): string {
  return value === null || value === undefined ? "–" : `${formatNumber(value, digits)}%`;
}

export function thresholdText(cond: ConditionDef, field?: Field): string {
  const basis = cond.method === "percentile" ? "worst 10% of records" : "engineering threshold";
  return `${OPERATOR_WORDS[cond.operator] ?? cond.operator} ${formatValue(cond.threshold, field, true)} (${basis})`;
}

export function plural(n: number, word: string, pluralWord = `${word}s`): string {
  return `${n.toLocaleString("en-US")} ${n === 1 ? word : pluralWord}`;
}

export interface FilterState {
  state: string;
  city: string;
  season: string;
  weather: string;
  category?: string;
  severity?: string;
  priority?: string;
  issue?: string;
  conditions?: string[];
}

export const EMPTY_FILTERS: FilterState = {
  state: "", city: "", season: "", weather: "", category: "", severity: "", priority: "", issue: "", conditions: [],
};

/** Convert form state into the API's filter object, dropping anything left at "Any". */
export function buildFilters(f: FilterState): Record<string, string | boolean | string[]> | null {
  const out: Record<string, string | boolean | string[]> = {};
  for (const key of ["state", "season", "weather", "category", "severity", "priority"] as const) {
    const v = f[key];
    if (v) out[key] = v;
  }
  if (f.city.trim()) out.city = f.city.trim();
  if (f.issue === "yes" || f.issue === "no") out.issue_reported = f.issue === "yes";
  if (f.conditions?.length) out.conditions = f.conditions;
  return Object.keys(out).length ? out : null;
}

export function countFilters(f: FilterState): number {
  return Object.keys(buildFilters(f) ?? {}).length;
}

/** Keep only measurements the user actually entered, parsed as numbers. */
export function buildMetrics(values: Record<string, string>): Record<string, number> | null {
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(values)) {
    if (v.trim() === "") continue;
    const n = Number(v);
    if (Number.isFinite(n)) out[k] = n;
  }
  return Object.keys(out).length ? out : null;
}

/** Validation message for one measurement input, or null when it is valid or empty. */
export function metricError(raw: string, field: Field): string | null {
  if (raw.trim() === "") return null;
  const n = Number(raw);
  if (!Number.isFinite(n)) return "Enter a number.";
  if (field.type === "int" && !Number.isInteger(n)) return "Enter a whole number.";
  if (field.min !== null && field.min !== undefined && n < field.min) return `Must be at least ${field.min}.`;
  if (field.max !== null && field.max !== undefined && n > field.max) return `Must be at most ${field.max}.`;
  return null;
}

export function hasInvalidMetric(values: Record<string, string>, lookup: (name: string) => Field | undefined): boolean {
  return Object.entries(values).some(([k, v]) => {
    const f = lookup(k);
    return f !== undefined && metricError(v, f) !== null;
  });
}
