// Shapes of the backend responses the UI reads. Only the fields used are typed.

export type Severity = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

export interface Field {
  name: string;
  label: string;
  unit: string | null;
  role: string | null;
  type: string;
  input: boolean;
  min?: number | null;
  max?: number | null;
}

export interface ConditionDef {
  name: string;
  label: string;
  metric: string;
  operator: "<" | ">" | "<=" | ">=";
  threshold: number;
  method: "percentile" | "domain";
  group: string;
}

export interface Reference {
  fields: Field[];
  conditions: ConditionDef[];
  categories: string[];
  states: Record<string, string>;
  seasons: string[];
  weather: string[];
  severities: Severity[];
  priorities: string[];
}

export interface Health {
  status: "ok" | "degraded";
  checks: {
    incidents?: number;
    models?: Record<string, boolean>;
    llm?: { provider: string; model: string | null };
    retrieval_error?: string;
  };
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown> | null;
  request_id?: string | null;
}

export interface RecordMetadata {
  season: string;
  weather: string;
  city: string;
  state: string;
  zip: string;
  temperature_c: number | null;
  humidity_pct: number | null;
  issue_reported: boolean | null;
}

export interface NetworkRecord {
  record_id: string;
  category: string;
  secondary_categories: string[];
  severity: Severity;
  severity_score: number;
  priority: string;
  conditions: string[];
  metrics: Record<string, number>;
  metadata: RecordMetadata;
  hybrid_score?: number;
  semantic_score?: number;
  keyword_score?: number;
  metric_score?: number;
}

export interface SearchResponse {
  query: string;
  results: NetworkRecord[];
  filters: Record<string, unknown>;
  candidates: number;
  matching_records: number;
  weights: { semantic: number; keyword: number; metric: number };
}

export interface RiskPrediction {
  available: boolean;
  predicted_fault_rate_pct?: number;
  risk_level?: string;
  interval_80pct?: [number, number];
  inputs_used?: string[];
  input_coverage?: number;
  low_input_coverage?: boolean;
  model?: { algorithm: string; feature_set: string; test_r2: number; test_mae: number };
  label?: string;
  caveat?: string;
  reason?: string;
}

export interface Evidence {
  condition?: string;
  label?: string;
  source?: string;
  context?: string;
  value?: unknown;
}

export interface Hypothesis {
  id: string;
  title: string;
  confidence: number;
  confidence_breakdown: Record<string, number>;
  supporting_evidence: Evidence[];
  contradicting_evidence: string[];
  historical_support: string;
  additional_evidence_required: string[];
  note: string | null;
}

export interface Classification {
  category: string;
  secondary_categories: string[];
  category_reasons: string[];
  category_method: string;
  conditions: string[];
  severity: Severity;
  severity_score: number;
  severity_reasons: string[];
  severity_basis: string;
  ml_category: {
    available: boolean;
    predicted_category?: string;
    probability?: number;
    reason?: string;
    label?: string;
  };
  ml_agrees_with_rules: boolean | null;
}

export interface Priority {
  priority: string;
  priority_label: string;
  priority_score: number;
  reasons: string[];
}

export interface Action {
  action: string;
  hypothesis: string;
  based_on: string[];
  hypothesis_confidence: number;
}

export interface MeasuredEvidence {
  condition: string;
  label: string;
  metric: string;
  value: number;
  operator: string;
  threshold: number;
}

export interface LlmAttempt {
  attempt: number;
  status?: string;
  reason?: string;
  grounding_issues?: string[];
}

export interface Validation {
  deterministic: Record<string, unknown> & { issues: string[]; completeness: number };
  llm_judge: (Record<string, unknown> & { issues: string[]; score: number }) | null;
  llm_judge_info: { status: string; reason?: string };
  diagnosis_supported: boolean;
  evidence_grounded: boolean;
  recommendation_relevant: boolean;
  recommendation_grounded: boolean;
  hallucination_detected: boolean;
  score: number;
  issues: string[];
  method: string;
}

export interface AnalyzeResponse {
  analysis_id: string;
  query: string;
  limitations: string[];
  retrieval: SearchResponse;
  evidence: {
    observed: {
      values: Record<string, number | string>;
      present: string[];
      absent: string[];
      unknown: string[];
      condition_sources: Record<string, string>;
      measured_evidence: MeasuredEvidence[];
    };
    history: {
      n_records: number;
      record_ids: string[];
      condition_frequency: Record<string, { count: number; share: number }>;
      category_distribution: Record<string, number>;
      weather_distribution: Record<string, number>;
    };
  };
  classification: Classification;
  risk: {
    predicted: RiskPrediction;
    historical: {
      label: string;
      mean: number;
      median: number;
      min: number;
      max: number;
      risk_band_of_mean: string;
      n_records: number;
    } | null;
  };
  priority: Priority;
  root_cause: {
    primary_hypothesis: string | null;
    primary_hypothesis_id: string | null;
    confidence: number | null;
    confidence_method: string;
    supporting_evidence: Evidence[];
    contradicting_evidence: string[];
    alternative_hypotheses: { id: string; title: string; confidence: number }[];
    additional_evidence_required: string[];
    hypotheses: Hypothesis[];
    statement: string;
  };
  recommendations: {
    actions: Action[];
    checks: { check: string; hypothesis: string; hypothesis_confidence: number }[];
    monitoring: { action: string; metric: string; baseline_median: number; based_on: string[] }[];
    disclaimer: string;
  };
  narrative: {
    source: "llm" | "template";
    narrative: {
      summary: string;
      diagnosis: string;
      hypothesis_notes: { hypothesis_id: string; rationale: string }[];
      limitations: string[];
    };
    attempts: LlmAttempt[];
    llm_rejected?: boolean;
  };
  validation: Validation;
  provenance: { claim: string; source_type: string; evidence: Record<string, unknown> }[];
  report_text: string;
  agent_trace: { agent: string; status: string; duration_s: number; summary: string }[];
  timing_s: number;
}

export interface ClassifyResponse {
  source: string;
  record_id?: string;
  conditions: { conditions: string[]; evidence: MeasuredEvidence[]; unavailable: string[] };
  classification: Classification;
  priority: Priority;
  stored_labels?: { category: string; severity: Severity; priority: string };
}

export interface RiskResponse {
  predicted: RiskPrediction;
  record_id?: string;
  historical_fault_rate_pct?: { value: number; label: string };
}
