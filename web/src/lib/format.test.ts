import { describe, expect, it } from "vitest";
import { buildFilters, buildMetrics, countFilters, EMPTY_FILTERS, formatValue, metricError, thresholdText } from "./format";
import type { ConditionDef, Field } from "./types";

const latency: Field = { name: "latency_ms", label: "Latency (ms)", unit: "ms", role: "kpi", type: "float", input: true, min: 0, max: null };
const users: Field = { name: "user_count", label: "User Count", unit: null, role: "load", type: "int", input: true, min: 0, max: null };
const signal: Field = { name: "signal_strength_dbm", label: "Signal Strength (dBm)", unit: "dBm", role: "kpi", type: "float", input: true, min: -140, max: -44 };

describe("formatValue", () => {
  it("uses unit precision and strips float noise", () => {
    expect(formatValue(430.00100000000003, { unit: "Mbps", type: "float" }, true)).toBe("430.0 Mbps");
    expect(formatValue(4.155, { unit: "%", type: "float" }, true)).toBe("4.16%");
  });
  it("formats integers with separators and handles missing values", () => {
    expect(formatValue(6259, users)).toBe("6,259");
    expect(formatValue(null, latency)).toBe("–");
    expect(formatValue(true)).toBe("Yes");
  });
});

describe("thresholdText", () => {
  it("describes the operator and the threshold basis", () => {
    const c: ConditionDef = { name: "weak_signal", label: "Weak signal strength", metric: "signal_strength_dbm", operator: "<", threshold: -110, method: "domain", group: "radio" };
    expect(thresholdText(c, signal)).toBe("below -110.0 dBm (engineering threshold)");
  });
});

describe("buildFilters", () => {
  it("drops empty values and maps issue_reported to a boolean", () => {
    expect(buildFilters(EMPTY_FILTERS)).toBeNull();
    const f = { ...EMPTY_FILTERS, state: "MN", city: "  Schneiderville ", issue: "no", conditions: ["weak_signal"] };
    expect(buildFilters(f)).toEqual({ state: "MN", city: "Schneiderville", issue_reported: false, conditions: ["weak_signal"] });
    expect(countFilters(f)).toBe(4);
  });
});

describe("measurements", () => {
  it("keeps only entered numbers", () => {
    expect(buildMetrics({ latency_ms: "150", packet_loss_pct: "", user_count: "12" })).toEqual({ latency_ms: 150, user_count: 12 });
    expect(buildMetrics({ latency_ms: " " })).toBeNull();
  });
  it("validates against the schema range and type", () => {
    expect(metricError("-150", signal)).toBe("Must be at least -140.");
    expect(metricError("-40", signal)).toBe("Must be at most -44.");
    expect(metricError("1.5", users)).toBe("Enter a whole number.");
    expect(metricError("abc", latency)).toBe("Enter a number.");
    expect(metricError("", latency)).toBeNull();
  });
});
